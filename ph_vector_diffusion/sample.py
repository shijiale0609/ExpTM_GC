#!/usr/bin/env python3
"""Generate two-feature samples at one or more LP1 values from a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from model import ConditionalGaussianDDPM, ConditionalNoiseMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--control", type=float, nargs="+", required=True, metavar="LP1")
    parser.add_argument("--num-samples-per-control", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=11)
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


def normalize_control(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return 2.0 * (values - low) / (high - low) - 1.0


def main() -> None:
    args = parse_args()
    if args.num_samples_per_control <= 0 or args.batch_size <= 0:
        raise ValueError("sample counts must be positive")
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = ConditionalNoiseMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    diffusion = ConditionalGaussianDDPM(
        checkpoint["diffusion_config"]["num_timesteps"],
        torch.tensor(checkpoint["prior_intercept"]),
        torch.tensor(checkpoint["prior_slope"]),
        device,
    )
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    control_range = checkpoint["control_normalization"]
    control_name = checkpoint.get("control_name", "LP1")
    feature_columns = checkpoint["feature_columns"]

    rows: list[list[float]] = []
    for control_value in args.control:
        normalized_condition = normalize_control(
            np.full((args.num_samples_per_control, 1), control_value, dtype=np.float32),
            control_range["low"],
            control_range["high"],
        )
        generated_batches = []
        for start in range(0, args.num_samples_per_control, args.batch_size):
            condition = torch.from_numpy(normalized_condition[start : start + args.batch_size]).to(device)
            generated_batches.append(diffusion.sample(model, condition).cpu().numpy())
        generated = np.concatenate(generated_batches, axis=0) * feature_std + feature_mean
        rows.extend([[control_value, *item] for item in generated.tolist()])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((control_name, *feature_columns))
        writer.writerows(rows)
    print(f"wrote {len(rows)} samples to {args.output_csv}")


if __name__ == "__main__":
    main()
