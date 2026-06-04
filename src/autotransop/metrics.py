from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.stats import spearmanr


def pearson_r(y_true: torch.Tensor, y_pred: torch.Tensor, dim: int = 0, eps: float = 1e-12) -> torch.Tensor:
    x = y_true
    y = y_pred
    xm = x - torch.nanmean(x, dim=dim, keepdim=True)
    ym = y - torch.nanmean(y, dim=dim, keepdim=True)
    num = torch.nansum(xm * ym, dim=dim)
    den = torch.sqrt(torch.nansum(xm * xm, dim=dim) * torch.nansum(ym * ym, dim=dim)).clamp_min(eps)
    return num / den


def masked_pearson_r(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: torch.Tensor,
    min_obs: int = 3,
    eps: float = 1e-12,
) -> torch.Tensor:
    m = mask.to(dtype=y_true.dtype)
    n = m.sum(dim=0)
    mx = (y_true * m).sum(dim=0) / n.clamp_min(1.0)
    my = (y_pred * m).sum(dim=0) / n.clamp_min(1.0)
    xm = (y_true - mx) * m
    ym = (y_pred - my) * m
    num = (xm * ym).sum(dim=0)
    den_x = (xm * xm).sum(dim=0)
    den_y = (ym * ym).sum(dim=0)
    r = num / torch.sqrt(den_x * den_y).clamp_min(eps)
    return r.masked_fill((n < min_obs) | (den_x < eps) | (den_y < eps), float("nan"))


def r_square(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    ss_res = torch.sum((y_true - y_pred).pow(2))
    ss_tot = torch.sum((y_true - torch.mean(y_true)).pow(2))
    return 1.0 - ss_res / (ss_tot + eps)


def sign_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    true_sign = torch.sign(torch.where(torch.abs(y_true) < eps, torch.zeros_like(y_true), y_true))
    pred_sign = torch.sign(torch.where(torch.abs(y_pred) < eps, torch.zeros_like(y_pred), y_pred))
    return true_sign.eq(pred_sign).float().mean(dim=1)


def per_sample_spearman(y_true, y_pred) -> np.ndarray:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    out = []
    for i in range(true.shape[0]):
        rho, _ = spearmanr(true[i, :], pred[i, :])
        out.append(rho)
    return np.asarray(out)


def to_numpy_prediction(prediction):
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    if torch.is_tensor(prediction):
        return prediction.detach().cpu().numpy()
    return np.asarray(prediction)
