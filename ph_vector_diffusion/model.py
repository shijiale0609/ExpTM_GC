"""Conditional vector DDPM components for the two-feature LP1 dataset."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings for a batch of integer timesteps."""
    half_dimension = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half_dimension, device=timesteps.device, dtype=torch.float32)
        / max(half_dimension - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ConditionalNoiseMLP(nn.Module):
    """Predicts DDPM noise from a 2D state, timestep, and normalized control."""

    def __init__(self, state_dim: int = 2, width: int = 128, time_dim: int = 32) -> None:
        super().__init__()
        self.time_dim = time_dim
        input_dim = state_dim + time_dim + 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, state_dim),
        )

    def forward(
        self, state: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        time_features = timestep_embedding(timestep, self.time_dim)
        return self.network(torch.cat((state, time_features, condition), dim=1))


class ConditionalGaussianDDPM:
    """DDPM whose terminal prior is N(a + b * control_normalized, I)."""

    def __init__(
        self,
        num_timesteps: int,
        prior_intercept: torch.Tensor,
        prior_slope: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.device = device
        self.prior_intercept = prior_intercept.to(device).float().view(1, -1)
        self.prior_slope = prior_slope.to(device).float().view(1, -1)

        betas = torch.linspace(1e-4, 2e-2, num_timesteps, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1, device=device), self.alpha_bars[:-1]))
        self.posterior_variance = betas * (1.0 - alpha_bars_previous) / (1.0 - self.alpha_bars)

    def prior_mean(self, condition: torch.Tensor) -> torch.Tensor:
        return self.prior_intercept + condition * self.prior_slope

    def q_sample_with_condition(
        self,
        state_zero: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        mean = self.prior_mean(condition)
        alpha_bar = self.alpha_bars[timestep].unsqueeze(1)
        return mean + alpha_bar.sqrt() * (state_zero - mean) + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        state = self.prior_mean(condition) + torch.randn(
            condition.shape[0], self.prior_intercept.shape[1], device=self.device
        )

        for timestep_value in range(self.num_timesteps - 1, -1, -1):
            timestep = torch.full(
                (condition.shape[0],), timestep_value, device=self.device, dtype=torch.long
            )
            noise_prediction = model(state, timestep, condition)
            alpha = self.alphas[timestep_value]
            alpha_bar = self.alpha_bars[timestep_value]
            mean = self.prior_mean(condition)
            residual = state - mean
            residual_mean = (residual - self.betas[timestep_value] * noise_prediction / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()

            if timestep_value > 0:
                residual_mean = residual_mean + self.posterior_variance[timestep_value].sqrt() * torch.randn_like(state)
            state = mean + residual_mean

        return state
