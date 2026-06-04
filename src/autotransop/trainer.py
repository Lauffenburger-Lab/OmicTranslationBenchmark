from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import TrainingConfig
from .data import DomainBatch, DomainTensorData, balanced_domain_batches
from .losses import (
    bce_discriminator_loss,
    bce_generator_prior_loss,
    label_pair_mask,
    metric_pair_loss,
    mutual_information_loss,
    prior_sample_like,
    reconstruction_loss,
)
from .model import AutoTransOP, dtype_from_name
from .modules import l2_regularization


@dataclass
class TrainingHistory:
    values: Dict[str, list] = field(default_factory=lambda: defaultdict(list))

    def append_epoch(self, epoch_values: Mapping[str, Sequence[float]]) -> None:
        for key, vals in epoch_values.items():
            clean = [v for v in vals if v is not None and not np.isnan(v)]
            self.values[key].append(float(np.mean(clean)) if clean else float("nan"))

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(dict(self.values))


class AutoTransOPTrainer:
    def __init__(self, model: AutoTransOP, training: Optional[TrainingConfig] = None) -> None:
        self.model = model
        self.training = training or TrainingConfig()
        device_name = self.training.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)
        self.dtype = dtype_from_name(self.training.dtype)
        self.model.to(device=self.device, dtype=self.dtype)

    def fit(self, datasets: Mapping[str, DomainTensorData]) -> TrainingHistory:
        if self.training.seed is not None:
            torch.manual_seed(self.training.seed)
            np.random.seed(self.training.seed)

        main_optimizer = torch.optim.Adam(
            list(self.model.main_parameters()),
            lr=self.training.lr,
            weight_decay=self.training.weight_decay,
        )
        adversary_optimizers = {
            name: torch.optim.Adam(list(self.model.adversary_parameters(name)), lr=self.training.adversary_lr)
            for name in self.model.adversaries
        }
        prior_optimizer = None
        if self.model.prior_discriminator is not None:
            prior_lr = self.model.config.prior.discriminator_lr or self.training.adversary_lr
            prior_optimizer = torch.optim.Adam(list(self.model.prior_parameters()), lr=prior_lr)

        scheduler = None
        if self.training.scheduler_step is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                main_optimizer,
                step_size=self.training.scheduler_step,
                gamma=self.training.scheduler_gamma,
            )

        history = TrainingHistory()
        for epoch in range(self.training.epochs):
            epoch_values = self.train_epoch(
                datasets,
                main_optimizer,
                adversary_optimizers,
                prior_optimizer=prior_optimizer,
                epoch_seed=None if self.training.seed is None else self.training.seed + epoch,
            )
            history.append_epoch(epoch_values)
            if scheduler is not None:
                scheduler.step()
        return history

    def train_epoch(
        self,
        datasets: Mapping[str, DomainTensorData],
        main_optimizer: torch.optim.Optimizer,
        adversary_optimizers: Mapping[str, torch.optim.Optimizer],
        prior_optimizer: Optional[torch.optim.Optimizer] = None,
        epoch_seed: Optional[int] = None,
    ) -> Dict[str, list]:
        self.model.train()
        epoch_values: Dict[str, list] = defaultdict(list)
        iterator = balanced_domain_batches(
            datasets,
            self.training.batch_sizes,
            self.training.batch_size,
            self.device,
            self.dtype,
            shuffle=self.training.shuffle,
            seed=epoch_seed,
        )
        for batches in iterator:
            self._adversary_steps(batches, adversary_optimizers, epoch_values)
            if prior_optimizer is not None:
                self._prior_discriminator_step(batches, prior_optimizer, epoch_values)
            main_optimizer.zero_grad()
            outputs = self.model.forward_batches(batches)
            total_loss = self._main_loss(batches, outputs, epoch_values)
            total_loss.backward()
            if self.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(list(self.model.main_parameters()), self.training.grad_clip)
            main_optimizer.step()
            epoch_values["loss"].append(float(total_loss.detach().cpu()))
        return epoch_values

    def _adversary_steps(
        self,
        batches: Sequence[DomainBatch],
        optimizers: Mapping[str, torch.optim.Optimizer],
        epoch_values: Dict[str, list],
    ) -> None:
        if not self.model.adversaries:
            return
        with torch.no_grad():
            outputs = self.model.forward_batches(batches)
        for name, adversary in self.model.adversaries.items():
            config = self.model.adversary_configs[name]
            for _ in range(max(1, int(config.steps))):
                z, y, _domain_ids, _defined = self._collect_latents_and_labels(
                    batches,
                    outputs,
                    config.latent_space,
                    config.target_key,
                    config.domains,
                    None,
                )
                if z is None:
                    continue
                optimizers[name].zero_grad()
                logits = adversary(z.detach())
                loss = F.cross_entropy(logits, y.long().view(-1))
                loss = loss + l2_regularization(adversary, config.l2)
                loss.backward()
                optimizers[name].step()
                epoch_values[f"adversary_{name}_loss"].append(float(loss.detach().cpu()))

    def _prior_discriminator_step(
        self,
        batches: Sequence[DomainBatch],
        optimizer: torch.optim.Optimizer,
        epoch_values: Dict[str, list],
    ) -> None:
        prior = self.model.config.prior
        if self.model.prior_discriminator is None:
            return
        with torch.no_grad():
            outputs = self.model.forward_batches(batches)
            z, _domains = self._collect_latents(batches, outputs, prior.latent_space)
        optimizer.zero_grad()
        z_prior = prior_sample_like(z, prior.distribution)
        pred_real = self.model.prior_discriminator(z_prior)
        pred_fake = self.model.prior_discriminator(z.detach())
        loss = prior.beta * bce_discriminator_loss(pred_real, pred_fake)
        loss.backward()
        optimizer.step()
        epoch_values["prior_discriminator_loss"].append(float(loss.detach().cpu()))

    def _main_loss(self, batches: Sequence[DomainBatch], outputs: Mapping[str, dict], epoch_values: Dict[str, list]) -> torch.Tensor:
        ref = next(iter(outputs.values()))["global"]
        total = ref.new_tensor(0.0)
        total = total + self._reconstruction_terms(batches, outputs, epoch_values)
        total = total + self._pair_regularizer_terms(batches, outputs, epoch_values)
        total = total + self._mutual_information_term(batches, outputs, epoch_values)
        total = total + self._head_terms(batches, outputs, epoch_values)
        total = total + self._adversary_penalty_terms(batches, outputs, epoch_values)
        total = total + self._prior_generator_term(batches, outputs, epoch_values)
        return total

    def _reconstruction_terms(self, batches: Sequence[DomainBatch], outputs: Mapping[str, dict], epoch_values: Dict[str, list]) -> torch.Tensor:
        total = next(iter(outputs.values()))["global"].new_tensor(0.0)
        for batch in batches:
            domain_config = self.model.domain_configs[batch.domain]
            loss = reconstruction_loss(
                outputs[batch.domain]["reconstruction"],
                batch.x,
                domain_config.reconstruction,
                mask=batch.mask,
            )
            loss = domain_config.reconstruction_weight * loss
            loss = loss + l2_regularization(self.model.encoders[batch.domain], domain_config.encoder_l2)
            loss = loss + l2_regularization(self.model.decoders[batch.domain], domain_config.decoder_l2)
            total = total + loss
            epoch_values[f"reconstruction_{batch.domain}"].append(float(loss.detach().cpu()))
        if self.model.config.version == "v2":
            total = total + l2_regularization(self.model.domain_effect, self.model.config.domain_effect.l2)
        return total

    def _pair_regularizer_terms(self, batches: Sequence[DomainBatch], outputs: Mapping[str, dict], epoch_values: Dict[str, list]) -> torch.Tensor:
        total = next(iter(outputs.values()))["global"].new_tensor(0.0)
        for config in self.model.config.pair_regularizers:
            z, y, domain_ids, defined = self._collect_latents_and_labels(
                batches,
                outputs,
                config.latent_space,
                config.label_key,
                config.domains,
                config.defined_key,
            )
            if z is None:
                continue
            mask = label_pair_mask(
                y,
                domain_ids=domain_ids,
                defined=defined,
                same_label=config.same_label,
                cross_domain_only=config.cross_domain_only,
            )
            if mask.sum() <= 0:
                continue
            term = config.weight * metric_pair_loss(z, mask, config.metrics, config.metric_weights)
            total = total + term
            epoch_values[f"pair_{config.name}"].append(float(term.detach().cpu()))
        return total

    def _mutual_information_term(self, batches: Sequence[DomainBatch], outputs: Mapping[str, dict], epoch_values: Dict[str, list]) -> torch.Tensor:
        config = self.model.config.mutual_information
        if not config.enabled:
            return next(iter(outputs.values()))["global"].new_tensor(0.0)
        if self.model.local_discriminator is None or config.label_key is None:
            return next(iter(outputs.values()))["global"].new_tensor(0.0)
        z, y, domain_ids, defined = self._collect_latents_and_labels(
            batches,
            outputs,
            config.latent_space,
            config.label_key,
            config.domains,
            config.defined_key,
        )
        if z is None:
            return next(iter(outputs.values()))["global"].new_tensor(0.0)
        pos_mask = label_pair_mask(y, domain_ids, defined, same_label=True, cross_domain_only=config.cross_domain_only)
        neg_mask = label_pair_mask(y, domain_ids, defined, same_label=False, cross_domain_only=config.cross_domain_only)
        scores = self.model.local_discriminator.score_matrix(z)
        term = config.weight * mutual_information_loss(scores, pos_mask, neg_mask)
        epoch_values["mutual_information"].append(float(term.detach().cpu()))
        return term

    def _head_terms(self, batches: Sequence[DomainBatch], outputs: Mapping[str, dict], epoch_values: Dict[str, list]) -> torch.Tensor:
        total = next(iter(outputs.values()))["global"].new_tensor(0.0)
        for name, head in self.model.heads.items():
            config = self.model.head_configs[name]
            z, y, _domain_ids, _defined = self._collect_latents_and_labels(
                batches,
                outputs,
                config.latent_space,
                config.target_key,
                config.domains,
                None,
            )
            if z is None:
                continue
            pred = head(z)
            if config.kind == "classifier":
                loss = F.cross_entropy(pred, y.long().view(-1))
            elif config.kind == "regressor":
                target = y.to(dtype=pred.dtype)
                loss = F.mse_loss(pred.squeeze(), target.squeeze())
            else:
                raise ValueError(f"Unsupported head kind: {config.kind}")
            loss = config.weight * loss + l2_regularization(head, config.l2)
            total = total + loss
            epoch_values[f"head_{name}"].append(float(loss.detach().cpu()))
        return total

    def _adversary_penalty_terms(self, batches: Sequence[DomainBatch], outputs: Mapping[str, dict], epoch_values: Dict[str, list]) -> torch.Tensor:
        total = next(iter(outputs.values()))["global"].new_tensor(0.0)
        for name, adversary in self.model.adversaries.items():
            config = self.model.adversary_configs[name]
            z, y, _domain_ids, _defined = self._collect_latents_and_labels(
                batches,
                outputs,
                config.latent_space,
                config.target_key,
                config.domains,
                None,
            )
            if z is None:
                continue
            self._set_requires_grad(adversary, False)
            penalty = F.cross_entropy(adversary(z), y.long().view(-1))
            self._set_requires_grad(adversary, True)
            term = -config.weight * penalty
            total = total + term
            epoch_values[f"adversary_{name}_penalty"].append(float(term.detach().cpu()))
        return total

    def _prior_generator_term(self, batches: Sequence[DomainBatch], outputs: Mapping[str, dict], epoch_values: Dict[str, list]) -> torch.Tensor:
        prior = self.model.config.prior
        if not prior.enabled or self.model.prior_discriminator is None:
            return next(iter(outputs.values()))["global"].new_tensor(0.0)
        z, _domain_ids = self._collect_latents(batches, outputs, prior.latent_space)
        self._set_requires_grad(self.model.prior_discriminator, False)
        term = prior.weight * bce_generator_prior_loss(self.model.prior_discriminator(z))
        self._set_requires_grad(self.model.prior_discriminator, True)
        epoch_values["prior_generator"].append(float(term.detach().cpu()))
        return term

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(value)

    def _collect_latents(
        self,
        batches: Sequence[DomainBatch],
        outputs: Mapping[str, dict],
        latent_space: str,
        domains: Optional[Sequence[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        allowed = None if domains is None else set(domains)
        zs = []
        domain_ids = []
        for batch in batches:
            if allowed is not None and batch.domain not in allowed:
                continue
            z = self.model.latent_for_space(outputs[batch.domain], latent_space)
            zs.append(z)
            domain_ids.append(torch.full((z.shape[0],), self.model.domain_to_id[batch.domain], device=z.device, dtype=torch.long))
        if not zs:
            ref = next(iter(outputs.values()))["global"]
            return ref.new_zeros((0, ref.shape[1])), torch.empty(0, device=ref.device, dtype=torch.long)
        return torch.cat(zs, dim=0), torch.cat(domain_ids, dim=0)

    def _collect_latents_and_labels(
        self,
        batches: Sequence[DomainBatch],
        outputs: Mapping[str, dict],
        latent_space: str,
        label_key: str,
        domains: Optional[Sequence[str]] = None,
        defined_key: Optional[str] = None,
    ):
        allowed = None if domains is None else set(domains)
        zs = []
        labels = []
        domain_ids = []
        defined_values = []
        for batch in batches:
            if allowed is not None and batch.domain not in allowed:
                continue
            if label_key not in batch.labels:
                continue
            z = self.model.latent_for_space(outputs[batch.domain], latent_space)
            y = batch.labels[label_key].to(z.device)
            if y.ndim > 1 and y.shape[-1] == 1:
                y = y.view(-1)
            zs.append(z)
            labels.append(y)
            domain_ids.append(torch.full((z.shape[0],), self.model.domain_to_id[batch.domain], device=z.device, dtype=torch.long))
            if defined_key is not None:
                if defined_key not in batch.labels:
                    defined_values.append(torch.ones(z.shape[0], device=z.device, dtype=torch.bool))
                else:
                    defined_values.append(batch.labels[defined_key].to(z.device).bool())
        if not zs:
            return None, None, None, None
        defined = torch.cat(defined_values, dim=0) if defined_values else None
        return torch.cat(zs, dim=0), torch.cat(labels, dim=0), torch.cat(domain_ids, dim=0), defined
