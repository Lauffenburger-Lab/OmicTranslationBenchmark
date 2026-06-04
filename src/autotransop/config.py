from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


@dataclass
class MLPConfig:
    hidden: Sequence[int] = field(default_factory=lambda: [64])
    dropout: float = 0.1
    input_dropout: float = 0.0
    batch_norm_momentum: Optional[float] = 0.6
    activation: str = "ELU"
    bias: bool = True


@dataclass
class DomainConfig:
    name: str
    input_dim: int
    encoder: MLPConfig = field(default_factory=MLPConfig)
    decoder: MLPConfig = field(default_factory=MLPConfig)
    reconstruction: str = "mse"
    decoder_distribution: str = "deterministic"
    reconstruction_weight: float = 1.0
    encoder_l2: float = 0.0
    decoder_l2: float = 0.0
    mask_inputs: bool = False


@dataclass
class DomainEffectConfig:
    mode: str = "none"
    hidden: Sequence[int] = field(default_factory=tuple)
    dropout: float = 0.1
    l2: float = 0.0


@dataclass
class HeadConfig:
    name: str
    target_key: str
    kind: str = "classifier"
    num_outputs: int = 2
    latent_space: str = "global"
    hidden: Sequence[int] = field(default_factory=lambda: [64, 32])
    domains: Optional[Sequence[str]] = None
    weight: float = 1.0
    l2: float = 0.0
    input_dropout: float = 0.0
    dropout: float = 0.1
    batch_norm_momentum: Optional[float] = 0.6


@dataclass
class AdversaryConfig:
    name: str
    target_key: str
    num_classes: int
    latent_space: str = "global"
    hidden: Sequence[int] = field(default_factory=lambda: [64, 32])
    domains: Optional[Sequence[str]] = None
    weight: float = 1.0
    steps: int = 1
    l2: float = 0.0
    input_dropout: float = 0.0
    dropout: float = 0.1
    batch_norm_momentum: Optional[float] = 0.6


@dataclass
class PairRegularizerConfig:
    name: str
    label_key: str
    metrics: Sequence[str] = field(default_factory=lambda: ("euclidean",))
    latent_space: str = "global"
    weight: float = 1.0
    metric_weights: Dict[str, float] = field(default_factory=dict)
    domains: Optional[Sequence[str]] = None
    defined_key: Optional[str] = None
    same_label: bool = True
    cross_domain_only: bool = False


@dataclass
class MutualInformationConfig:
    enabled: bool = False
    label_key: Optional[str] = None
    latent_space: str = "global"
    weight: float = 1.0
    domains: Optional[Sequence[str]] = None
    defined_key: Optional[str] = None
    cross_domain_only: bool = False
    discriminator_hidden_dim: Optional[int] = None


@dataclass
class PriorConfig:
    enabled: bool = False
    distribution: str = "normal"
    latent_space: str = "global"
    weight: float = 1.0
    discriminator_lr: Optional[float] = None
    beta: float = 1.0


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 128
    batch_sizes: Dict[str, int] = field(default_factory=dict)
    lr: float = 1e-3
    adversary_lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler_step: Optional[int] = None
    scheduler_gamma: float = 0.8
    grad_clip: Optional[float] = None
    device: Optional[str] = None
    dtype: str = "float32"
    shuffle: bool = True
    seed: Optional[int] = None
    log_every: int = 1


@dataclass
class AutoTransOPConfig:
    domains: Sequence[DomainConfig]
    latent_dim: int = 32
    version: str = "v1"
    domain_effect: DomainEffectConfig = field(default_factory=DomainEffectConfig)
    heads: Sequence[HeadConfig] = field(default_factory=tuple)
    adversaries: Sequence[AdversaryConfig] = field(default_factory=tuple)
    pair_regularizers: Sequence[PairRegularizerConfig] = field(default_factory=tuple)
    mutual_information: MutualInformationConfig = field(default_factory=MutualInformationConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    normalize_latent: bool = False

    def validate(self) -> None:
        if self.version not in {"v1", "v2", "v3"}:
            raise ValueError("version must be one of 'v1', 'v2', or 'v3'")
        names = [d.name for d in self.domains]
        if len(names) != len(set(names)):
            raise ValueError("domain names must be unique")
        if len(names) < 2:
            raise ValueError("AutoTransOP requires at least two domains")
        if self.prior.distribution not in {"normal", "uniform"}:
            raise ValueError("prior distribution must be 'normal' or 'uniform'")
        if self.domain_effect.mode not in {"none", "vector", "mlp"}:
            raise ValueError("domain_effect.mode must be none, vector, or mlp")
        for domain in self.domains:
            if domain.reconstruction == "nb" and domain.decoder_distribution != "negative_binomial":
                raise ValueError("nb reconstruction requires decoder_distribution='negative_binomial'")
            if domain.reconstruction == "gaussian_nll" and domain.decoder_distribution != "gaussian":
                raise ValueError("gaussian_nll reconstruction requires decoder_distribution='gaussian'")
            if domain.reconstruction not in {"mse", "masked_mse", "gaussian_nll", "nb"}:
                raise ValueError(f"Unsupported reconstruction for {domain.name}: {domain.reconstruction}")
        for head in self.heads:
            if head.kind not in {"classifier", "regressor"}:
                raise ValueError("head kind must be classifier or regressor")
            if head.latent_space not in {"global", "basal", "composed"}:
                raise ValueError("head latent_space must be global, basal, or composed")
        for adversary in self.adversaries:
            if adversary.latent_space not in {"global", "basal", "composed"}:
                raise ValueError("adversary latent_space must be global, basal, or composed")
        for pair in self.pair_regularizers:
            if pair.latent_space not in {"global", "basal", "composed"}:
                raise ValueError("pair regularizer latent_space must be global, basal, or composed")
            unsupported = set(pair.metrics) - {"euclidean", "cosine"}
            if unsupported:
                raise ValueError(f"Unsupported pair metrics: {sorted(unsupported)}")


DEFAULT_L1000_LANDMARKS = AutoTransOPConfig(
    latent_dim=292,
    version="v1",
    domains=(
        DomainConfig("domain_1", 978, encoder=MLPConfig([640, 384], 0.1, 0.5), decoder=MLPConfig([384, 640], 0.2)),
        DomainConfig("domain_2", 978, encoder=MLPConfig([640, 384], 0.1, 0.5), decoder=MLPConfig([384, 640], 0.2)),
    ),
)


DEFAULT_SEROLOGY = AutoTransOPConfig(
    latent_dim=32,
    version="v2",
    domain_effect=DomainEffectConfig(mode="vector", dropout=0.5, l2=1e-6),
    domains=(
        DomainConfig("human", 128, encoder=MLPConfig([128, 64], 0.3), decoder=MLPConfig([64, 128], 0.2)),
        DomainConfig("nhp", 64, encoder=MLPConfig([64], 0.3), decoder=MLPConfig([64], 0.2)),
    ),
    pair_regularizers=(
        PairRegularizerConfig("protection_alignment", "protection", metrics=("euclidean", "cosine"), weight=4.0),
    ),
    mutual_information=MutualInformationConfig(enabled=True, label_key="protection", weight=10.0),
)
