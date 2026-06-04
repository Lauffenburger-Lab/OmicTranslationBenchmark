from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def zero_like_reference(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(0.0)


def reconstruction_loss(prediction, target: torch.Tensor, mode: str = "mse", mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mode in {"mse", "masked_mse"}:
        pred = prediction[0] if isinstance(prediction, tuple) else prediction
        err = (pred - target).pow(2)
        if mask is not None or mode == "masked_mse":
            if mask is None:
                mask = torch.ones_like(target)
            mask = mask.to(dtype=target.dtype)
            return (err * mask).sum(dim=1).div(mask.sum(dim=1).clamp_min(1.0)).mean()
        return err.sum(dim=1).mean()
    if mode == "gaussian_nll":
        mu, var = prediction
        var = var.clamp_min(1e-6)
        loss = 0.5 * (torch.log(var) + (target - mu).pow(2) / var)
        if mask is not None:
            mask = mask.to(dtype=target.dtype)
            return (loss * mask).sum(dim=1).div(mask.sum(dim=1).clamp_min(1.0)).mean()
        return loss.sum(dim=1).mean()
    if mode == "nb":
        mu, theta = prediction
        return negative_binomial_nll(mu, theta, target, mask=mask)
    raise ValueError(f"Unsupported reconstruction mode: {mode}")


def negative_binomial_nll(
    mu: torch.Tensor,
    theta: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))
    log_theta_mu = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu)
        + target * (torch.log(mu + eps) - log_theta_mu)
        + torch.lgamma(target + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target + 1)
    )
    loss = -torch.where(torch.isnan(res), torch.zeros_like(res) + torch.inf, res)
    if mask is not None:
        mask = mask.to(dtype=target.dtype)
        return (loss * mask).sum(dim=1).div(mask.sum(dim=1).clamp_min(1.0)).mean()
    return loss.mean()


def label_pair_mask(
    labels: torch.Tensor,
    domain_ids: Optional[torch.Tensor] = None,
    defined: Optional[torch.Tensor] = None,
    same_label: bool = True,
    cross_domain_only: bool = False,
    exclude_self: bool = True,
) -> torch.Tensor:
    labels = labels.view(-1)
    mask = labels.view(-1, 1).eq(labels.view(1, -1))
    if not same_label:
        mask = ~mask
    if defined is not None:
        defined = defined.view(-1).bool()
        mask = mask & (defined.view(-1, 1) & defined.view(1, -1))
    if cross_domain_only:
        if domain_ids is None:
            raise ValueError("domain_ids are required for cross_domain_only masks")
        domain_ids = domain_ids.view(-1)
        mask = mask & domain_ids.view(-1, 1).ne(domain_ids.view(1, -1))
    if exclude_self:
        eye = torch.eye(mask.shape[0], dtype=torch.bool, device=mask.device)
        mask = mask & ~eye
    return mask.float()


def euclidean_pair_loss(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (torch.cdist(z, z) * mask).sum() / denom


def cosine_pair_loss(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    zn = F.normalize(z, dim=-1)
    cosine = torch.matmul(zn, zn.t())
    return ((1.0 - cosine) * mask).sum() / denom


def mutual_information_loss(scores: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor) -> torch.Tensor:
    log_2 = math.log(2.0)
    pos_mask = pos_mask.float()
    neg_mask = neg_mask.float()
    if pos_mask.sum() <= 0 or neg_mask.sum() <= 0:
        return scores.new_tensor(0.0)
    ep = (log_2 - F.softplus(-scores)) * pos_mask
    eq = (F.softplus(-scores) + scores - log_2) * neg_mask
    ep = ep.sum() / pos_mask.sum().clamp_min(1.0)
    eq = eq.sum() / neg_mask.sum().clamp_min(1.0)
    return eq - ep


def prior_sample_like(z: torch.Tensor, distribution: str = "normal") -> torch.Tensor:
    if distribution == "normal":
        return torch.randn_like(z)
    if distribution == "uniform":
        return torch.rand_like(z)
    raise ValueError("distribution must be normal or uniform")


def bce_discriminator_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
    real = torch.ones_like(pred_real)
    fake = torch.zeros_like(pred_fake)
    return F.binary_cross_entropy(pred_real, real) + F.binary_cross_entropy(pred_fake, fake)


def bce_generator_prior_loss(pred_fake: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(pred_fake, torch.ones_like(pred_fake))


def metric_pair_loss(z: torch.Tensor, mask: torch.Tensor, metrics: Sequence[str], metric_weights: Optional[dict] = None) -> torch.Tensor:
    metric_weights = metric_weights or {}
    out = z.new_tensor(0.0)
    for metric in metrics:
        if metric == "euclidean":
            out = out + metric_weights.get(metric, 1.0) * euclidean_pair_loss(z, mask)
        elif metric == "cosine":
            out = out + metric_weights.get(metric, 1.0) * cosine_pair_loss(z, mask)
        else:
            raise ValueError(f"Unsupported pair metric: {metric}")
    return out
