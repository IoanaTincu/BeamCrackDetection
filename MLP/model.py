from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Generic fully-connected neural network (MLP) for regression.

    Flexible:
    - input_dim: variable number of inputs
    - output_dim: variable number of outputs (you will use 1 initially)
    - hidden_layers: list/tuple of hidden layer widths

    Initial experiments (your requirement):
    - use exactly one hidden layer with a fixed width (e.g., hidden_layers=(64,))
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int = 1,
            hidden_layers: tuple[int, ...] = (64,),
            activation: str = "relu",
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if any(w <= 0 for w in hidden_layers):
            raise ValueError("All hidden layer widths must be positive integers.")

        act_cls = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "sigmoid": nn.Sigmoid,
        }.get(activation.lower(), nn.ReLU)

        layers: list[nn.Module] = []
        prev = input_dim

        for width in hidden_layers:
            layers.append(nn.Linear(prev, width))
            layers.append(act_cls())
            prev = width

        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
