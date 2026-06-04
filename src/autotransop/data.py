from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch


@dataclass
class DomainTensorData:
    x: torch.Tensor
    labels: Mapping[str, torch.Tensor] = field(default_factory=dict)
    mask: Optional[torch.Tensor] = None

    @classmethod
    def from_arrays(
        cls,
        x,
        labels: Optional[Mapping[str, object]] = None,
        mask: Optional[object] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "DomainTensorData":
        x_tensor = torch.as_tensor(x, dtype=dtype)
        label_tensors = {k: torch.as_tensor(v) for k, v in (labels or {}).items()}
        mask_tensor = None if mask is None else torch.as_tensor(mask, dtype=dtype)
        return cls(x_tensor, label_tensors, mask_tensor)

    def __len__(self) -> int:
        return int(self.x.shape[0])


@dataclass
class DomainBatch:
    domain: str
    x: torch.Tensor
    labels: Dict[str, torch.Tensor] = field(default_factory=dict)
    mask: Optional[torch.Tensor] = None


def make_index_batches(n: int, batch_size: int, shuffle: bool = True, rng: Optional[np.random.Generator] = None) -> list:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = rng or np.random.default_rng()
    order = np.arange(n)
    if shuffle:
        rng.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def _pad_batches(
    batches: list,
    n: int,
    batch_size: int,
    target_len: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> list:
    while len(batches) < target_len:
        batches.extend(make_index_batches(n, batch_size, shuffle=shuffle, rng=rng))
    return batches[:target_len]


def balanced_domain_batches(
    datasets: Mapping[str, DomainTensorData],
    batch_sizes: Mapping[str, int],
    default_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Iterator[Sequence[DomainBatch]]:
    rng = np.random.default_rng(seed)
    index_batches = {}
    for name, data in datasets.items():
        bs = int(batch_sizes.get(name, default_batch_size))
        index_batches[name] = make_index_batches(len(data), bs, shuffle=shuffle, rng=rng)
    max_len = max(len(v) for v in index_batches.values())
    for name, data in datasets.items():
        bs = int(batch_sizes.get(name, default_batch_size))
        index_batches[name] = _pad_batches(index_batches[name], len(data), bs, max_len, shuffle, rng)

    for step in range(max_len):
        out = []
        for name, data in datasets.items():
            idx = index_batches[name][step]
            labels = {key: value[idx].to(device) for key, value in data.labels.items()}
            mask = None if data.mask is None else data.mask[idx].to(device=device, dtype=dtype)
            out.append(DomainBatch(name, data.x[idx].to(device=device, dtype=dtype), labels, mask))
        yield out
