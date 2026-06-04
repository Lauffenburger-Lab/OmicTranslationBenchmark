from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from .metrics import masked_pearson_r, pearson_r, to_numpy_prediction


def reconstruct_array(model, x, domain: str, mask=None, batch_size: int = 256, device: Optional[str] = None) -> np.ndarray:
    device_obj = torch.device(device or next(model.parameters()).device)
    arr = torch.as_tensor(x, dtype=next(model.parameters()).dtype)
    mask_tensor = None if mask is None else torch.as_tensor(mask, dtype=arr.dtype)
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, arr.shape[0], batch_size):
            block = arr[start : start + batch_size].to(device_obj)
            block_mask = None if mask_tensor is None else mask_tensor[start : start + batch_size].to(device_obj)
            z = model.encode(domain, block, block_mask)
            if model.config.version == "v2":
                z = model.compose(domain, z)
            preds.append(to_numpy_prediction(model.decode(domain, z)))
    return np.concatenate(preds, axis=0)


def translate_array(
    model,
    x,
    source_domain: str,
    target_domain: str,
    mask=None,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> np.ndarray:
    device_obj = torch.device(device or next(model.parameters()).device)
    arr = torch.as_tensor(x, dtype=next(model.parameters()).dtype)
    mask_tensor = None if mask is None else torch.as_tensor(mask, dtype=arr.dtype)
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, arr.shape[0], batch_size):
            block = arr[start : start + batch_size].to(device_obj)
            block_mask = None if mask_tensor is None else mask_tensor[start : start + batch_size].to(device_obj)
            preds.append(to_numpy_prediction(model.translate(block, source_domain, target_domain, block_mask)))
    return np.concatenate(preds, axis=0)


def per_feature_performance(
    y_true,
    y_pred,
    feature_names: Optional[Sequence[str]] = None,
    mask=None,
    set_name: str = "validation",
    species: Optional[str] = None,
    fold: Optional[int] = None,
) -> pd.DataFrame:
    true = torch.as_tensor(y_true, dtype=torch.float32)
    pred = torch.as_tensor(y_pred, dtype=torch.float32)
    if mask is None:
        r = pearson_r(true, pred, dim=0).detach().cpu().numpy()
    else:
        r = masked_pearson_r(true, pred, torch.as_tensor(mask, dtype=torch.float32)).detach().cpu().numpy()
    names = list(feature_names) if feature_names is not None else [str(i) for i in range(true.shape[1])]
    out = pd.DataFrame({"feature_id": names, "r": r, "set": set_name})
    if species is not None:
        out["species"] = species
    if fold is not None:
        out["fold"] = fold
    return out


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    out[valid] = tmp
    return out


def compare_per_feature_to_reference(
    performance: pd.DataFrame,
    target_name: str = "shuffled",
    reference_name: str = "validation",
    feature_col: str = "feature_id",
    r_col: str = "r",
    set_col: str = "set",
) -> pd.DataFrame:
    rows = []
    target = performance[performance[set_col] == target_name]
    reference = performance[performance[set_col] == reference_name]
    for feature in performance[feature_col].dropna().unique():
        r_target = target[target[feature_col] == feature][r_col].dropna().values
        r_reference = reference[reference[feature_col] == feature][r_col].dropna().values
        if len(r_reference) < 1 or len(r_target) < 1:
            continue
        _, p_value = mannwhitneyu(r_reference, r_target, alternative="greater")
        rows.append(
            {
                feature_col: feature,
                "avg_r": float(np.nanmean(r_reference)),
                "reference_mean": float(np.nanmean(r_reference)),
                "target_mean": float(np.nanmean(r_target)),
                "p_value": float(p_value),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        out["p_adjusted"] = []
        return out
    out["p_adjusted"] = benjamini_hochberg(out["p_value"].values)
    return out.sort_values(["p_adjusted", "avg_r"], ascending=[True, False]).reset_index(drop=True)


def plot_per_feature_performance_scatter(
    comparison: pd.DataFrame,
    output_path: Optional[str] = None,
    feature_col: str = "feature_id",
    alpha: float = 0.4,
    point_size: float = 25,
    title: str = "Per-feature performance vs. reference",
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    y = -np.log10(np.clip(comparison["p_adjusted"].values, np.finfo(float).tiny, 1.0))
    ax.scatter(comparison["avg_r"], y, alpha=alpha, s=point_size)
    ax.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Average validation Pearson r")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title(title)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, ax


def distribution_moment_frame(translated, target, feature_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    translated = pd.DataFrame(translated, columns=feature_names)
    target = pd.DataFrame(target, columns=feature_names)
    return pd.DataFrame(
        {
            "translated_mean": translated.mean(axis=0),
            "target_mean": target.mean(axis=0),
            "translated_var": translated.var(axis=0, ddof=1),
            "target_var": target.var(axis=0, ddof=1),
        }
    )
