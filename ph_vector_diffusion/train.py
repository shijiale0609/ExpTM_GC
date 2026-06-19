#!/usr/bin/env python3
"""Train an ExpTM-style conditional MLP diffusion model on LP1 CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import ConditionalGaussianDDPM, ConditionalNoiseMLP


FEATURE_COLUMNS = ("x_top10mean", "y_rank450mean_ascending")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        action="append",
        nargs=2,
        metavar=("VALUE", "CSV"),
        required=True,
        help="Control value and its CSV file; repeat for each condition",
    )
    parser.add_argument(
        "--control-name",
        default="LP1",
        help="Name stored with the conditional variable in the checkpoint and outputs",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ph_vector_ddpm"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_condition(path: Path, control_value: float) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    values = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite feature values")
    return values, np.full((len(values), 1), control_value, dtype=np.float32)


def normalize_control(control: np.ndarray, low: float, high: float) -> np.ndarray:
    return 2.0 * (control - low) / (high - low) - 1.0


def fit_prior(features: np.ndarray, condition: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit m(c)=intercept+slope*c for the ExpTM-style conditional Gaussian prior."""
    design = np.concatenate((np.ones_like(condition), condition), axis=1)
    weights, _, _, _ = np.linalg.lstsq(design, features, rcond=None)
    return weights[0].astype(np.float32), weights[1].astype(np.float32)


def evaluate(
    model: ConditionalNoiseMLP,
    diffusion: ConditionalGaussianDDPM,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for state_zero, condition in loader:
            state_zero = state_zero.to(device)
            condition = condition.to(device)
            timestep = torch.randint(0, diffusion.num_timesteps, (state_zero.shape[0],), device=device)
            noise = torch.randn_like(state_zero)
            state_t = diffusion.q_sample_with_condition(state_zero, timestep, condition, noise)
            losses.append(F.mse_loss(model(state_t, timestep, condition), noise).item())
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    if len(args.condition) < 2:
        raise ValueError("Provide at least two --condition VALUE CSV pairs")
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = [(float(control), Path(path)) for control, path in args.condition]
    if len({control for control, _ in sources}) != len(sources):
        raise ValueError("Each control value must be supplied only once")
    loaded = [load_condition(path, control) for control, path in sorted(sources)]
    features = np.concatenate([values for values, _ in loaded], axis=0)
    control = np.concatenate([values for _, values in loaded], axis=0)
    control_low, control_high = float(control.min()), float(control.max())
    if control_low == control_high:
        raise ValueError("Control values must span more than one value")
    condition = normalize_control(control, control_low, control_high)

    indices = np.random.permutation(len(features))
    val_size = int(len(indices) * args.val_fraction)
    val_indices, train_indices = indices[:val_size], indices[val_size:]
    train_features = features[train_indices]
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    if np.any(feature_std == 0):
        raise ValueError("A feature has zero variance and cannot be standardized")
    standardized = (features - feature_mean) / feature_std

    prior_intercept, prior_slope = fit_prior(standardized[train_indices], condition[train_indices])
    train_dataset = TensorDataset(
        torch.from_numpy(standardized[train_indices]), torch.from_numpy(condition[train_indices])
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if val_size:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(standardized[val_indices]), torch.from_numpy(condition[val_indices])),
            batch_size=args.batch_size,
            shuffle=False,
        )

    model = ConditionalNoiseMLP(width=args.width).to(device)
    diffusion = ConditionalGaussianDDPM(
        args.timesteps,
        torch.from_numpy(prior_intercept),
        torch.from_numpy(prior_slope),
        device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history_path = args.output_dir / "history.csv"

    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_noise_mse", "val_noise_mse"))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            batch_losses: list[float] = []
            for state_zero, batch_condition in train_loader:
                state_zero = state_zero.to(device)
                batch_condition = batch_condition.to(device)
                timestep = torch.randint(0, args.timesteps, (state_zero.shape[0],), device=device)
                noise = torch.randn_like(state_zero)
                state_t = diffusion.q_sample_with_condition(state_zero, timestep, batch_condition, noise)
                loss = F.mse_loss(model(state_t, timestep, batch_condition), noise)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())

            validation_loss = evaluate(model, diffusion, val_loader, device) if val_loader else float("nan")
            record = {
                "epoch": epoch,
                "train_noise_mse": float(np.mean(batch_losses)),
                "val_noise_mse": validation_loss,
            }
            writer.writerow(record)
            handle.flush()
            if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
                print(
                    f"epoch={epoch:4d} train_noise_mse={record['train_noise_mse']:.5f} "
                    f"val_noise_mse={record['val_noise_mse']:.5f}"
                )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {"state_dim": 2, "width": args.width, "time_dim": 32},
        "diffusion_config": {"num_timesteps": args.timesteps},
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "prior_intercept": prior_intercept.tolist(),
        "prior_slope": prior_slope.tolist(),
        "control_name": args.control_name,
        "control_normalization": {"low": control_low, "high": control_high},
    }
    torch.save(checkpoint, args.output_dir / "model.pt")
    (args.output_dir / "training_config.json").write_text(
        json.dumps(
            {
                "condition_sources": [
                    {args.control_name: control, "csv": str(path)}
                    for control, path in sorted(sources)
                ],
                "control_name": args.control_name,
                "num_rows": int(len(features)),
                "num_train_rows": int(len(train_indices)),
                "num_validation_rows": int(val_size),
                "seed": args.seed,
                "device": str(device),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved checkpoint to {args.output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
