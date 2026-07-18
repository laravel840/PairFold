"""Optimal segmentation of a long sequence into length-[MIN,MAX] fragments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

from .config import MAX_LEN, MIN_LEN


@dataclass
class Segment:
    start: int
    end: int  # exclusive
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start


def optimal_segmentation(
    seq_len: int,
    score_fn: Callable[[int, int], float],
    min_len: int = MIN_LEN,
    max_len: int = MAX_LEN,
) -> Tuple[List[Segment], float]:
    """
    Dynamic programming: maximize sum of fragment scores.
    score_fn(i, j) = quality of fragment seq[i:j] (j-i in [min_len, max_len]).
    """
    NEG = -1e18
    dp = np.full(seq_len + 1, NEG, dtype=np.float64)
    prev = np.full(seq_len + 1, -1, dtype=np.int32)
    dp[0] = 0.0

    for i in range(seq_len):
        if dp[i] <= NEG / 2:
            continue
        for L in range(min_len, max_len + 1):
            j = i + L
            if j > seq_len:
                break
            s = score_fn(i, j)
            cand = dp[i] + s
            if cand > dp[j]:
                dp[j] = cand
                prev[j] = i

    if dp[seq_len] <= NEG / 2:
        # fallback: greedy chunks of max_len
        segs = []
        i = 0
        while i < seq_len:
            j = min(i + max_len, seq_len)
            if j - i < min_len and i > 0:
                # merge remainder into previous by shifting
                break
            if j - i < min_len:
                j = seq_len
            segs.append(Segment(i, j, 0.0))
            i = j
        return segs, 0.0

    segs: List[Segment] = []
    j = seq_len
    while j > 0:
        i = int(prev[j])
        segs.append(Segment(i, j, float(score_fn(i, j))))
        j = i
    segs.reverse()
    return segs, float(dp[seq_len])


def enumerate_near_optimal(
    seq_len: int,
    score_fn: Callable[[int, int], float],
    top_k: int = 5,
) -> List[Tuple[List[Segment], float]]:
    """Return best segmentation only (top_k reserved for future beam search)."""
    segs, total = optimal_segmentation(seq_len, score_fn)
    return [(segs, total)]
