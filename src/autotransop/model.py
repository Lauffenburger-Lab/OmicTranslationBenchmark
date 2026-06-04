from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch

from .config import (
    AdversaryConfig,
    AutoTransOPConfig,
    DomainConfig,
    HeadConfig,
)
from .data import DomainBatch
from .modules import (
    ClassifierHead,
    Decoder,
    DomainEffect,
    Encoder,
    LocalDiscriminator,
    PriorDiscriminator,
    RegressionHead,
)


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float": torch.float32,
        "double": torch.float64,
        "float64": torch.float64,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


class AutoTransOP(torch.nn.Module):
    def __init__(self, config: AutoTransOPConfig, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.domain_names = [domain.name for domain in config.domains]
        self.domain_to_id = {name: i for i, name in enumerate(self.domain_names)}
        self.domain_configs: Dict[str, DomainConfig] = {domain.name: domain for domain in config.domains}

        self.encoders = torch.nn.ModuleDict()
        self.decoders = torch.nn.ModuleDict()
        for domain in config.domains:
            self.encoders[domain.name] = Encoder(
                domain.input_dim,
                domain.encoder.hidden,
                config.latent_dim,
                dropout=domain.encoder.dropout,
                input_dropout=domain.encoder.input_dropout,
                batch_norm_momentum=domain.encoder.batch_norm_momentum,
                activation=domain.encoder.activation,
                bias=domain.encoder.bias,
                normalize_output=config.normalize_latent,
                dtype=dtype,
            )
            self.decoders[domain.name] = Decoder(
                config.latent_dim,
                domain.decoder.hidden,
                domain.input_dim,
                dropout=domain.decoder.dropout,
                input_dropout=domain.decoder.input_dropout,
                batch_norm_momentum=domain.decoder.batch_norm_momentum,
                activation=domain.decoder.activation,
                bias=domain.decoder.bias,
                distribution=domain.decoder_distribution,
                dtype=dtype,
            )

        effect_mode = config.domain_effect.mode if config.version == "v2" else "none"
        self.domain_effect = DomainEffect(
            len(self.domain_names),
            config.latent_dim,
            mode=effect_mode,
            hidden=config.domain_effect.hidden,
            dropout=config.domain_effect.dropout,
            dtype=dtype,
        )

        self.heads = torch.nn.ModuleDict()
        self.head_configs: Dict[str, HeadConfig] = {}
        for head in config.heads:
            self.head_configs[head.name] = head
            cls = ClassifierHead if head.kind == "classifier" else RegressionHead
            self.heads[head.name] = cls(
                config.latent_dim,
                head.hidden,
                head.num_outputs,
                dropout=head.dropout,
                input_dropout=head.input_dropout,
                batch_norm_momentum=head.batch_norm_momentum,
                activation="ReLU",
                dtype=dtype,
            )

        self.adversaries = torch.nn.ModuleDict()
        self.adversary_configs: Dict[str, AdversaryConfig] = {}
        for adversary in config.adversaries:
            self.adversary_configs[adversary.name] = adversary
            self.adversaries[adversary.name] = ClassifierHead(
                config.latent_dim,
                adversary.hidden,
                adversary.num_classes,
                dropout=adversary.dropout,
                input_dropout=adversary.input_dropout,
                batch_norm_momentum=adversary.batch_norm_momentum,
                activation="ReLU",
                dtype=dtype,
            )

        self.local_discriminator = None
        if config.mutual_information.enabled:
            dim = config.mutual_information.discriminator_hidden_dim or config.latent_dim
            self.local_discriminator = LocalDiscriminator(config.latent_dim, dim, dtype=dtype)

        self.prior_discriminator = None
        if config.prior.enabled:
            self.prior_discriminator = PriorDiscriminator(config.latent_dim, dtype=dtype)

    def _domain_ids(self, domain: str, n: int, device: torch.device) -> torch.Tensor:
        return torch.full((n,), self.domain_to_id[domain], dtype=torch.long, device=device)

    def encode(self, domain: str, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        domain_config = self.domain_configs[domain]
        return self.encoders[domain](x, mask if domain_config.mask_inputs else None)

    def compose(self, domain: str, z: torch.Tensor) -> torch.Tensor:
        domain_ids = self._domain_ids(domain, z.shape[0], z.device)
        return self.domain_effect(z, domain_ids)

    def decode(self, domain: str, z: torch.Tensor):
        return self.decoders[domain](z)

    def forward_domain(self, batch: DomainBatch) -> dict:
        z_global = self.encode(batch.domain, batch.x, batch.mask)
        z_composed = self.compose(batch.domain, z_global)
        decode_z = z_composed if self.config.version == "v2" else z_global
        reconstruction = self.decode(batch.domain, decode_z)
        return {
            "domain": batch.domain,
            "global": z_global,
            "composed": z_composed,
            "decode_z": decode_z,
            "reconstruction": reconstruction,
        }

    def forward_batches(self, batches: Sequence[DomainBatch]) -> Dict[str, dict]:
        return {batch.domain: self.forward_domain(batch) for batch in batches}

    @torch.no_grad()
    def translate(
        self,
        x: torch.Tensor,
        source_domain: str,
        target_domain: str,
        mask: Optional[torch.Tensor] = None,
    ):
        self.eval()
        z = self.encode(source_domain, x, mask)
        if self.config.version == "v2":
            z = self.compose(target_domain, z)
        return self.decode(target_domain, z)

    def main_parameters(self) -> Iterable[torch.nn.Parameter]:
        modules = [self.encoders, self.decoders, self.domain_effect, self.heads]
        if self.local_discriminator is not None:
            modules.append(self.local_discriminator)
        for module in modules:
            yield from module.parameters()

    def adversary_parameters(self, name: str) -> Iterable[torch.nn.Parameter]:
        yield from self.adversaries[name].parameters()

    def prior_parameters(self) -> Iterable[torch.nn.Parameter]:
        if self.prior_discriminator is not None:
            yield from self.prior_discriminator.parameters()

    def latent_for_space(self, output: Mapping[str, torch.Tensor], latent_space: str) -> torch.Tensor:
        if latent_space not in {"global", "basal", "composed"}:
            raise ValueError("latent_space must be global, basal, or composed")
        return output["global"] if latent_space in {"global", "basal"} else output["composed"]

    def latent_space_for_decode(self) -> str:
        return "composed" if self.config.version == "v2" else "global"
