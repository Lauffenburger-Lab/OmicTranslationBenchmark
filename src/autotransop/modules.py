from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def activation_from_name(name: str) -> torch.nn.Module:
    table = {
        "ELU": torch.nn.ELU(),
        "ReLU": torch.nn.ReLU(),
        "LeakyReLU": torch.nn.LeakyReLU(0.01),
        "GELU": torch.nn.GELU(),
        "Tanh": torch.nn.Tanh(),
        "Sigmoid": torch.nn.Sigmoid(),
        "Identity": torch.nn.Identity(),
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def init_linear(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                child.bias.data.zero_()
        if isinstance(child, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(child.weight)


def l2_regularization(module: torch.nn.Module, weight: float) -> torch.Tensor:
    param = next(module.parameters(), None)
    if param is None:
        return torch.tensor(0.0)
    if weight <= 0:
        return param.new_tensor(0.0)
    out = param.new_tensor(0.0)
    for value in module.parameters():
        out = out + value.pow(2).sum()
    return weight * out


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int],
        output_dim: int,
        dropout: float = 0.1,
        input_dropout: float = 0.0,
        batch_norm_momentum: Optional[float] = 0.6,
        activation: str = "ELU",
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_dropout = torch.nn.Dropout(input_dropout) if input_dropout > 0 else torch.nn.Identity()
        layers = []
        prev = input_dim
        act = activation_from_name(activation)
        for width in hidden:
            layers.append(torch.nn.Linear(prev, int(width), bias=bias, dtype=dtype))
            if batch_norm_momentum is not None:
                layers.append(torch.nn.BatchNorm1d(int(width), momentum=batch_norm_momentum, dtype=dtype))
            layers.append(act.__class__() if isinstance(act, (torch.nn.ReLU, torch.nn.ELU, torch.nn.GELU)) else act)
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            prev = int(width)
        layers.append(torch.nn.Linear(prev, output_dim, bias=bias, dtype=dtype))
        self.net = torch.nn.Sequential(*layers)
        init_linear(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_dropout(x))


class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int],
        latent_dim: int,
        dropout: float = 0.1,
        input_dropout: float = 0.0,
        batch_norm_momentum: Optional[float] = 0.6,
        activation: str = "ELU",
        bias: bool = True,
        normalize_output: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.normalize_output = normalize_output
        self.network = FeedForward(
            input_dim,
            hidden,
            latent_dim,
            dropout=dropout,
            input_dropout=input_dropout,
            batch_norm_momentum=batch_norm_momentum,
            activation=activation,
            bias=bias,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask.to(dtype=x.dtype)
        z = self.network(x)
        if self.normalize_output:
            z = F.normalize(z, dim=-1)
        return z


class Decoder(torch.nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden: Sequence[int],
        output_dim: int,
        dropout: float = 0.1,
        input_dropout: float = 0.0,
        batch_norm_momentum: Optional[float] = 0.6,
        activation: str = "ELU",
        bias: bool = True,
        distribution: str = "deterministic",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if distribution not in {"deterministic", "gaussian", "negative_binomial"}:
            raise ValueError("distribution must be deterministic, gaussian, or negative_binomial")
        self.distribution = distribution
        output_size = output_dim if distribution == "deterministic" else output_dim * 2
        self.output_dim = output_dim
        self.network = FeedForward(
            latent_dim,
            hidden,
            output_size,
            dropout=dropout,
            input_dropout=input_dropout,
            batch_norm_momentum=batch_norm_momentum,
            activation=activation,
            bias=bias,
            dtype=dtype,
        )

    def forward(self, z: torch.Tensor):
        out = self.network(z)
        if self.distribution == "deterministic":
            return out
        mu, var = out[:, : self.output_dim], out[:, self.output_dim :]
        return F.softplus(mu).add(1e-3), F.softplus(var).add(1e-3)


class ClassifierHead(FeedForward):
    pass


class RegressionHead(FeedForward):
    pass


class LocalDiscriminator(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: Optional[int] = None, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.block = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, input_dim, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, output_dim, dtype=dtype),
            torch.nn.ReLU(),
        )
        self.shortcut = torch.nn.Linear(input_dim, output_dim, dtype=dtype)
        init_linear(self)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.block(z) + self.shortcut(z)

    def score_matrix(self, z: torch.Tensor) -> torch.Tensor:
        z_un = self.forward(z)
        return torch.matmul(z_un, z_un.t())


class PriorDiscriminator(torch.nn.Module):
    def __init__(self, input_dim: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, input_dim, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, 1, dtype=dtype),
            torch.nn.Sigmoid(),
        )
        init_linear(self)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DomainEffect(torch.nn.Module):
    def __init__(
        self,
        num_domains: int,
        latent_dim: int,
        mode: str = "none",
        hidden: Sequence[int] = (),
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if mode not in {"none", "vector", "mlp"}:
            raise ValueError("domain effect mode must be none, vector, or mlp")
        self.mode = mode
        self.num_domains = num_domains
        self.latent_dim = latent_dim
        self.dropout = torch.nn.Dropout(dropout)
        if mode == "vector":
            self.embedding = torch.nn.Embedding(num_domains, latent_dim, dtype=dtype)
            init_linear(self)
        elif mode == "mlp":
            self.effect_net = FeedForward(
                num_domains,
                hidden,
                latent_dim,
                dropout=dropout,
                input_dropout=0.0,
                batch_norm_momentum=None,
                activation="ELU",
                bias=True,
                dtype=dtype,
            )

    def forward(self, z: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return z
        if self.mode == "vector":
            return z + self.dropout(self.embedding(domain_ids))
        one_hot = F.one_hot(domain_ids, num_classes=self.num_domains).to(dtype=z.dtype, device=z.device)
        return z + self.dropout(self.effect_net(one_hot))
