"""Hard limits to stop PairFold from freezing the OS with huge allocations."""

from __future__ import annotations

from typing import Optional

# Refuse any dense NxN float64 above ~0.25 GB (≈ ~5800 residues)
MAX_MATRIX_BYTES = int(0.25 * (1024**3))
# Abort prediction if process working set exceeds this
MAX_RSS_BYTES = int(3.5 * (1024**3))


class MemoryGuardError(RuntimeError):
    """Raised when an operation would thrash the machine."""


def rss_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return int(counters.WorkingSetSize)
    except Exception:
        pass
    return None


def guard_matrix(n: int, itemsize: int = 8, label: str = "matrix") -> None:
    """Refuse dense N×N allocations that would freeze Windows."""
    n = int(n)
    est = n * n * int(itemsize)
    if est > MAX_MATRIX_BYTES:
        raise MemoryGuardError(
            f"Safety stop: refused {label} {n}×{n} "
            f"(~{est / (1024**3):.1f} GB). "
            f"Long-chain clash scoring is disabled to protect the OS."
        )


def guard_rss(context: str = "predict") -> None:
    """Abort if the Python process is already using too much RAM."""
    b = rss_bytes()
    if b is None:
        return
    if b > MAX_RSS_BYTES:
        raise MemoryGuardError(
            f"Safety stop: memory use {b / (1024**3):.1f} GB exceeded "
            f"{MAX_RSS_BYTES / (1024**3):.1f} GB limit during {context}. "
            "Prediction aborted."
        )


def release_caches() -> None:
    """Best-effort cleanup after an abort."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import gc

        gc.collect()
    except Exception:
        pass
