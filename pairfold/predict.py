"""Inference: calibrated confidence + overlap consensus + DP segmentation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np

from .assemble import build_backbone
from .config import (
    AA_LIST,
    CALIB_DIR,
    CKPT_DIR,
    CONSENSUS_WEIGHT,
    CONTACT_USE_MAX_LEN,
    DP_FULL_MAX_LEN,
    FRAG_DIR,
    MAX_LEN,
    MAX_QUERY_LEN,
    MIN_LEN,
    PAD_IDX,
    SS_BOUNDARY_OPT_MAX_LEN,
    SS_PIPELINE_MAX_LEN,
    STRUCTURE_EXPORT_MAX_LEN,
    LEVER_ASSEMBLY_MAX_LEN,
    TERTIARY_MAX_LEN,
    UNK_IDX,
    VIEW_3D_MAX_LEN,
)
from .mem_guard import MemoryGuardError, guard_rss, release_caches
from .model.fragment_net import FragmentTorsionNet, sincos_to_angles
from .segment import Segment, optimal_segmentation

AA_SET = set(AA_LIST)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def aa_to_idx(ch: str) -> int:
    i = AA_LIST.find(ch)
    return i if i >= 0 else UNK_IDX


def circular_mean_deg(angles: List[float], weights: List[float]) -> float:
    if not angles:
        return 0.0
    rad = [a * math.pi / 180.0 for a in angles]
    wsum = sum(weights) + 1e-8
    s = sum(w * math.sin(a) for a, w in zip(rad, weights)) / wsum
    c = sum(w * math.cos(a) for a, w in zip(rad, weights)) / wsum
    return math.atan2(s, c) * 180.0 / math.pi


def angular_spread_deg(angles: List[float]) -> float:
    """Approximate circular std in degrees (0 = perfect agreement)."""
    if len(angles) <= 1:
        return 0.0
    mean = circular_mean_deg(angles, [1.0] * len(angles))
    rad_m = mean * math.pi / 180.0
    diffs = []
    for a in angles:
        d = (a * math.pi / 180.0 - rad_m + math.pi) % (2 * math.pi) - math.pi
        diffs.append(abs(d) * 180.0 / math.pi)
    return float(sum(diffs) / len(diffs))


class ConfidenceCalibrator:
    """
    Inference-time confidence calibration.

    Prefer sharpening temperature (T<1) from structure_blocks.TempCalibration
    so high-consensus predictions are not stuck near 50%. Falls back to legacy
    Platt / temperature fields in confidence_calibration.json.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        path = Path(path or CALIB_DIR / "confidence_calibration.json")
        self.enabled = path.exists()
        self.method = "none"
        self.T = 1.0
        self.a = 1.0
        self.b = 0.0
        self.sharpening_T: Optional[float] = None
        self.disorder_gamma = 0.45
        self.use_platt_before_temp = False
        self.conf_floor = 0.05
        if self.enabled:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.method = data.get("method", "platt")
            self.T = float(data.get("temperature_T", 1.0))
            self.a = float(data.get("platt_a", 1.0))
            self.b = float(data.get("platt_b", 0.0))
            if "sharpening_T" in data:
                self.sharpening_T = float(data["sharpening_T"])
                self.method = "temp_sharpen"
            self.disorder_gamma = float(data.get("disorder_gamma", 0.45))
            self.use_platt_before_temp = bool(data.get("use_platt_before_temp", False))
            self.conf_floor = float(data.get("conf_floor", 0.05))

    def __call__(self, conf: float, aa: str = "") -> float:
        if not self.enabled:
            return float(conf)
        eps = 1e-6
        c = min(max(conf, eps), 1 - eps)
        z = math.log(c / (1 - c))

        if self.method == "temp_sharpen" and self.sharpening_T is not None:
            if self.use_platt_before_temp:
                z = self.a * z + self.b
            p = 1.0 / (1.0 + math.exp(-z / max(self.sharpening_T, 1e-3)))
            # light per-residue disorder nudge for Gly/Pro
            if aa in ("G", "P"):
                p *= 1.0 - 0.5 * self.disorder_gamma
            elif aa in ("N", "D", "S"):
                p *= 1.0 - 0.25 * self.disorder_gamma
            return float(min(max(p, self.conf_floor), 1.0))

        if self.method == "temperature":
            p = 1.0 / (1.0 + math.exp(-z / self.T))
        else:
            p = 1.0 / (1.0 + math.exp(-(self.a * z + self.b)))
        return float(min(max(p, 0.0), 1.0))

    def calibrate_sequence(self, confs: List[float], sequence: str) -> List[float]:
        """Vectorized path via structure_blocks when sharpening is enabled."""
        if self.method == "temp_sharpen":
            from .structure_blocks import TempCalibration, calibrate_confidence
            import numpy as np

            calib = TempCalibration(
                T=float(self.sharpening_T or 0.65),
                platt_a=self.a,
                platt_b=self.b,
                use_platt=self.use_platt_before_temp,
                disorder_gamma=self.disorder_gamma,
                conf_floor=self.conf_floor,
            )
            return calibrate_confidence(np.asarray(confs, dtype=np.float64), sequence, calib).tolist()
        return [self(c, sequence[i] if i < len(sequence) else "") for i, c in enumerate(confs)]


class FragmentPredictor:
    def __init__(self, ckpt_path: Optional[Path] = None) -> None:
        path = Path(ckpt_path or CKPT_DIR / "best.pt")
        if not path.exists():
            path = CKPT_DIR / "last.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint in {CKPT_DIR}. Train with: python train.py"
            )
        self.dev = _device()
        ckpt = torch.load(path, map_location=self.dev)
        cfg = ckpt["config"]
        self.model = FragmentTorsionNet(
            vocab_size=cfg["vocab_size"],
            max_len=cfg["max_len"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"],
            dropout=0.0,
        ).to(self.dev)
        # Allow loading older checkpoints that had `uncert` instead of `log_sigma`
        try:
            self.model.load_state_dict(ckpt["model"])
        except RuntimeError:
            # Old checkpoint incompatible — re-raise with clear message
            raise RuntimeError(
                "Checkpoint architecture mismatch. Retrain with: python train.py"
            ) from None
        self.model.eval()
        self.max_len = cfg["max_len"]
        self.ckpt_path = str(path)
        self.calibrator = ConfidenceCalibrator()

        prior_path = FRAG_DIR / "seq_prior.json"
        self.seq_prior = {}
        if prior_path.exists():
            self.seq_prior = json.loads(prior_path.read_text(encoding="utf-8"))
        self.max_prior = max(self.seq_prior.values()) if self.seq_prior else 1
        self._frag_cache: Dict[str, Dict] = {}
        self._contact_predictor = None  # lazy ContactPredictor

    def _get_contact_predictor(self):
        if self._contact_predictor is None:
            try:
                from .contact_predict import ContactPredictor

                self._contact_predictor = ContactPredictor(device=self.dev)
            except Exception:
                self._contact_predictor = False
        return self._contact_predictor if self._contact_predictor is not False else None

    def predict_contacts(self, sequence: str) -> Dict:
        """Run ContactPairNet and return top-k anchors (empty if no ckpt)."""
        cp = self._get_contact_predictor()
        if cp is None or not cp.enabled:
            return {
                "enabled": False,
                "anchors": [],
                "contacts": [],
                "n_candidates": 0,
                "mean_score": 0.0,
                "ckpt": "",
            }
        return cp.top_anchors(sequence)

    def _prior_boost(self, seq: str) -> float:
        c = self.seq_prior.get(seq, 0)
        if c <= 0:
            return 0.0
        # log-scaled prior in [0, 0.15]
        return 0.15 * math.log(1 + c) / math.log(1 + self.max_prior)

    @torch.no_grad()
    def predict_fragment(self, seq: str) -> Dict:
        if not (MIN_LEN <= len(seq) <= self.max_len):
            raise ValueError(f"fragment length must be {MIN_LEN}-{self.max_len}")
        cached = self._frag_cache.get(seq)
        if cached is not None:
            return cached
        L = len(seq)
        tokens = [aa_to_idx(c) for c in seq] + [PAD_IDX] * (self.max_len - L)
        mask = [True] * L + [False] * (self.max_len - L)
        t = torch.tensor([tokens], dtype=torch.long, device=self.dev)
        m = torch.tensor([mask], dtype=torch.bool, device=self.dev)
        out = self.model(t, m)
        if len(out) == 3:
            sc, _log_sigma, conf = out
        else:
            sc, conf = out
        phi, psi = sincos_to_angles(sc)
        phi = phi[0, :L].cpu().tolist()
        psi = psi[0, :L].cpu().tolist()
        raw_conf = conf[0, :L].cpu().tolist()
        boost = self._prior_boost(seq)
        boosted = [min(1.0, c + boost) for c in raw_conf]
        if self.calibrator.method == "temp_sharpen":
            cal = self.calibrator.calibrate_sequence(boosted, seq)
        else:
            cal = [self.calibrator(c) for c in boosted]
        result = {
            "seq": seq,
            "phis_rad": phi,
            "psis_rad": psi,
            "phis_deg": [x * 180.0 / math.pi for x in phi],
            "psis_deg": [x * 180.0 / math.pi for x in psi],
            "confidence_raw": raw_conf,
            "confidence": cal,
            "mean_confidence": float(sum(cal) / max(len(cal), 1)),
            "prior_boost": boost,
        }
        self._frag_cache[seq] = result
        return result

    def _overlap_refine(
        self,
        seq: str,
        progress: Optional[Callable[[float, str], None]] = None,
        progress_lo: float = 0.0,
        progress_hi: float = 1.0,
    ) -> Dict:
        """Sliding windows of all lengths; consensus angles + agreement confidence."""
        n = len(seq)
        phi_lists: List[List[float]] = [[] for _ in range(n)]
        psi_lists: List[List[float]] = [[] for _ in range(n)]
        conf_lists: List[List[float]] = [[] for _ in range(n)]

        windows = []
        for L in range(MIN_LEN, min(MAX_LEN, n) + 1):
            for i in range(0, n - L + 1):
                windows.append((i, L))
        total_w = max(len(windows), 1)

        for wi, (i, L) in enumerate(windows):
            frag = self.predict_fragment(seq[i : i + L])
            for k in range(L):
                phi_lists[i + k].append(frag["phis_deg"][k])
                psi_lists[i + k].append(frag["psis_deg"][k])
                conf_lists[i + k].append(frag["confidence"][k])
            if progress and (wi % 8 == 0 or wi + 1 == total_w):
                t = progress_lo + (progress_hi - progress_lo) * ((wi + 1) / total_w)
                progress(t, f"Overlap refine {wi + 1}/{total_w}")

        phis, psis, confs = [], [], []
        for i in range(n):
            w = conf_lists[i] or [1.0]
            phis.append(circular_mean_deg(phi_lists[i], w))
            psis.append(circular_mean_deg(psi_lists[i], w))
            model_c = sum(conf_lists[i]) / max(len(conf_lists[i]), 1)
            spread = 0.5 * (
                angular_spread_deg(phi_lists[i]) + angular_spread_deg(psi_lists[i])
            )
            agree = max(0.0, 1.0 - spread / 60.0)
            confs.append(
                (1.0 - CONSENSUS_WEIGHT) * model_c + CONSENSUS_WEIGHT * agree
            )
        return {
            "phis": phis,
            "psis": psis,
            "confidence": confs,
            "mean_confidence": float(sum(confs) / max(len(confs), 1)),
        }

    def predict_sequence(
        self,
        sequence: str,
        progress: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        seq = "".join(ch for ch in sequence.upper() if ch.isalpha())
        if not seq or any(c not in AA_SET for c in seq):
            bad = {c for c in seq if c not in AA_SET}
            raise ValueError(f"Invalid sequence characters: {bad}")
        if len(seq) > MAX_QUERY_LEN:
            raise ValueError(f"Sequence longer than {MAX_QUERY_LEN}")
        if len(seq) < MIN_LEN:
            raise ValueError(f"Sequence shorter than {MIN_LEN}")

        t0 = time.perf_counter()

        def report(pct: float, message: str) -> None:
            if not progress:
                return
            pct = float(min(max(pct, 0.0), 0.99))
            elapsed = time.perf_counter() - t0
            eta = (elapsed / pct - elapsed) if pct > 0.02 else None
            progress(
                {
                    "pct": round(pct * 100, 1),
                    "message": message,
                    "elapsed_s": round(elapsed, 1),
                    "eta_s": round(eta, 1) if eta is not None else None,
                    "n": len(seq),
                }
            )

        report(0.01, f"Starting ({len(seq)} aa)")

        # Contacts only help tertiary/lever (≤CONTACT_USE_MAX_LEN). Above that,
        # skip — old path allocated N×N maps and froze the machine at ~25k aa.
        anchors: Optional[List[Tuple[int, int, float]]] = None
        contact_info: Dict = {
            "enabled": False,
            "anchors": [],
            "contacts": [],
            "mean_score": 0.0,
            "ckpt": "",
        }
        contact_note = ""
        if len(seq) <= CONTACT_USE_MAX_LEN:
            report(0.02, "Predicting long-range contacts")
            contact_info = self.predict_contacts(seq)
            if contact_info.get("enabled") and contact_info.get("anchors"):
                anchors = [tuple(a) for a in contact_info["anchors"]]  # type: ignore[misc]
            if anchors:
                contact_note = (
                    f" Contacts={len(anchors)} (mean score {contact_info.get('mean_score', 0):.2f})."
                )
            elif contact_info.get("enabled"):
                contact_note = " Contacts=0 (below threshold)."
        else:
            contact_note = (
                f" Contacts skipped (length > {CONTACT_USE_MAX_LEN}; "
                "anchors only used for short tertiary refine)."
            )

        calib_note = (
            f"calibrated={self.calibrator.enabled}/{self.calibrator.method}"
            if self.calibrator.enabled
            else "calibrated=false"
        )

        def maybe_structure(phis, psis):
            # Full atom export is heavy; UI rebuilds 3D from φ/ψ up to VIEW_3D_MAX_LEN.
            if len(seq) > STRUCTURE_EXPORT_MAX_LEN:
                # Do not duplicate sequence/angles here — top-level payload has them.
                return {
                    "sequence": "",
                    "phis": [],
                    "psis": [],
                    "residues": [],
                    "atoms": [],
                    "bonds": [],
                    "skipped_3d": True,
                    "reason": (
                        f"Server atom export omitted for length > {STRUCTURE_EXPORT_MAX_LEN}; "
                        "open 3D window to rebuild from torsions."
                    ),
                }
            return build_backbone(seq, phis, psis)

        def maybe_tertiary(phis, psis, confs, lo=0.88, hi=0.96):
            """Refine/rank tertiary for ≤TERTIARY_MAX_LEN; return angles + meta."""
            if len(seq) > TERTIARY_MAX_LEN:
                return list(phis), list(psis), confs, None, ""
            report(lo, "Tertiary structure refine / rank")
            try:
                from .tertiary import run_tertiary_pipeline

                upgraded = run_tertiary_pipeline(
                    seq,
                    phis,
                    psis,
                    progress=lambda t, m: report(lo + (hi - lo) * t, m),
                    anchors=anchors,
                )
                tmeta = upgraded["tertiary"]
                note = (
                    f" Tertiary score {tmeta['score']:.2f}"
                    f" (clash {tmeta['clash_energy']:.2f},"
                    f" Rg {tmeta['rg']:.1f}/{tmeta['rg_expected']:.1f}Å"
                    f"{', improved' if tmeta['improved'] else ''})."
                )
                return upgraded["phis"], upgraded["psis"], confs, tmeta, note
            except Exception as e:
                return list(phis), list(psis), confs, None, f" Tertiary skipped ({type(e).__name__})."

        if len(seq) <= MAX_LEN:
            report(0.05, "Short peptide — refining windows")
            if len(seq) >= MIN_LEN + 1:
                refined = self._overlap_refine(
                    seq, progress=lambda t, m: report(0.05 + 0.75 * t, m)
                )
                phis, psis, confs, tmeta, tnote = maybe_tertiary(
                    refined["phis"], refined["psis"], refined["confidence"], lo=0.82, hi=0.95
                )
                report(0.96, "Building 3D backbone")
                structure = maybe_structure(phis, psis)
                report(1.0, "Done")
                if progress:
                    progress(
                        {
                            "pct": 100.0,
                            "message": "Done",
                            "elapsed_s": round(time.perf_counter() - t0, 1),
                            "eta_s": 0.0,
                            "n": len(seq),
                        }
                    )
                out = {
                    "sequence": seq,
                    "mode": "overlap_consensus",
                    "segmentation": [
                        {
                            "start": 0,
                            "end": len(seq),
                            "seq": seq,
                            "confidence": refined["mean_confidence"],
                        }
                    ],
                    "phis": phis,
                    "psis": psis,
                    "confidence": confs,
                    "structure": structure,
                    "model": self.ckpt_path,
                    "device": str(self.dev),
                    "note": (
                        f"PDB-trained torsion model with overlap consensus ({calib_note})."
                        f"{contact_note}{tnote} Not AlphaFold."
                    ),
                    "contacts": {
                        "enabled": bool(contact_info.get("enabled")),
                        "n_anchors": len(anchors or []),
                        "anchors": [
                            {"i": a[0], "j": a[1], "dist": a[2]} for a in (anchors or [])
                        ],
                        "top": contact_info.get("contacts") or [],
                        "mean_score": contact_info.get("mean_score", 0.0),
                        "ckpt": contact_info.get("ckpt", ""),
                    },
                }
                if tmeta:
                    out["tertiary"] = tmeta
                return out

            report(0.3, "Direct fragment predict")
            frag = self.predict_fragment(seq)
            phis, psis, confs, tmeta, tnote = maybe_tertiary(
                frag["phis_deg"], frag["psis_deg"], frag["confidence"], lo=0.55, hi=0.92
            )
            report(0.95, "Building 3D backbone")
            structure = maybe_structure(phis, psis)
            if progress:
                progress(
                    {
                        "pct": 100.0,
                        "message": "Done",
                        "elapsed_s": round(time.perf_counter() - t0, 1),
                        "eta_s": 0.0,
                        "n": len(seq),
                    }
                )
            out = {
                "sequence": seq,
                "mode": "direct",
                "segmentation": [
                    {
                        "start": 0,
                        "end": len(seq),
                        "seq": seq,
                        "confidence": frag["mean_confidence"],
                    }
                ],
                "phis": phis,
                "psis": psis,
                "confidence": confs,
                "structure": structure,
                "model": self.ckpt_path,
                "device": str(self.dev),
                "note": (
                    f"PDB-trained short-fragment model ({calib_note})."
                    f"{contact_note}{tnote} Not AlphaFold."
                ),
                "contacts": {
                    "enabled": bool(contact_info.get("enabled")),
                    "n_anchors": len(anchors or []),
                    "anchors": [
                        {"i": a[0], "j": a[1], "dist": a[2]} for a in (anchors or [])
                    ],
                    "top": contact_info.get("contacts") or [],
                    "mean_score": contact_info.get("mean_score", 0.0),
                    "ckpt": contact_info.get("ckpt", ""),
                },
            }
            if tmeta:
                out["tertiary"] = tmeta
            return out

        # Long sequence: DP segmentation (short/medium) or greedy tiles (very long)
        n = len(seq)
        use_fast_tiles = n > DP_FULL_MAX_LEN

        if use_fast_tiles:
            report(0.05, f"Long chain — tiling into {MAX_LEN}-mers (skip full DP)")
            segs: List[Segment] = []
            i = 0
            while i < n:
                remaining = n - i
                if remaining <= MAX_LEN:
                    if remaining < MIN_LEN and segs:
                        # Merge tiny remainder into previous tile
                        prev = segs.pop()
                        segs.append(Segment(prev.start, n, 0.0))
                    else:
                        segs.append(Segment(i, n, 0.0))
                    break
                segs.append(Segment(i, i + MAX_LEN, 0.0))
                i += MAX_LEN
            total = 0.0
            report(0.20, f"Tiled into {len(segs)} fragments — refining")
        else:
            report(0.02, "Scoring fragment windows for segmentation")
            cache: Dict[tuple, float] = {}
            frag_cache: Dict[tuple, Dict] = {}
            # Approximate number of DP score lookups
            n_score = sum(max(0, n - L + 1) for L in range(MIN_LEN, MAX_LEN + 1))
            score_done = [0]

            def score_fn(i: int, j: int) -> float:
                key = (i, j)
                if key not in cache:
                    frag_cache[key] = self.predict_fragment(seq[i:j])
                    cache[key] = frag_cache[key]["mean_confidence"] * (j - i)
                score_done[0] += 1
                if progress and score_done[0] % 20 == 0:
                    report(
                        0.02 + 0.40 * min(1.0, score_done[0] / max(n_score, 1)),
                        f"Segmentation scoring {score_done[0]}/{n_score}",
                    )
                return cache[key]

            segs, total = optimal_segmentation(n, score_fn, MIN_LEN, MAX_LEN)
            report(0.45, f"Segmented into {len(segs)} fragments — refining")

        phis = [0.0] * n
        psis = [0.0] * n
        confs = [0.0] * n
        seg_info = []

        for si, seg in enumerate(segs):
            sub = seq[seg.start : seg.end]
            lo = 0.45 + 0.30 * (si / max(len(segs), 1))
            hi = 0.45 + 0.30 * ((si + 1) / max(len(segs), 1))
            if use_fast_tiles and MIN_LEN <= len(sub) <= MAX_LEN:
                # One network pass per tile — much cheaper than full overlap refine
                if progress and (si % 25 == 0 or si + 1 == len(segs)):
                    report(lo, f"Tile refine {si + 1}/{len(segs)}")
                frag = self.predict_fragment(sub)
                refined = {
                    "phis": frag["phis_deg"],
                    "psis": frag["psis_deg"],
                    "confidence": frag["confidence"],
                    "mean_confidence": frag["mean_confidence"],
                }
            else:
                refined = self._overlap_refine(
                    sub,
                    progress=lambda t, m, lo=lo, hi=hi: report(lo + (hi - lo) * t, m),
                )
            seg_info.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "seq": sub,
                    "confidence": refined["mean_confidence"],
                }
            )
            for k in range(len(sub)):
                idx = seg.start + k
                phis[idx] = refined["phis"][k]
                psis[idx] = refined["psis"][k]
                confs[idx] = refined["confidence"][k]

        # Clash-aware look-ahead assembly + lever correction (bounded length)
        asm_note = ""
        if len(seq) <= LEVER_ASSEMBLY_MAX_LEN:
            report(0.76, "Clash-aware assembly / lever correction")
            try:
                from .clash_assembly import (
                    AngleHypothesis,
                    assemble_greedy_backtrack,
                    make_pentamer_slots,
                )

                cons_ph = list(phis)
                cons_ps = list(psis)

                def hyp_fn(subseq: str, s: int, e: int):
                    L = e - s
                    hyps = [
                        AngleHypothesis(
                            confidence=0.92,
                            phis_deg=np.asarray(cons_ph[s:e], dtype=float),
                            psis_deg=np.asarray(cons_ps[s:e], dtype=float),
                            source="consensus",
                        )
                    ]
                    frag = self.predict_fragment(subseq)
                    hyps.append(
                        AngleHypothesis(
                            confidence=float(frag["mean_confidence"]),
                            phis_deg=np.asarray(frag["phis_deg"], dtype=float),
                            psis_deg=np.asarray(frag["psis_deg"], dtype=float),
                            source="network",
                        )
                    )
                    # Compact alternatives for backtracking diversity
                    hyps.append(
                        AngleHypothesis(
                            confidence=0.35,
                            phis_deg=np.full(L, -57.0),
                            psis_deg=np.full(L, -47.0),
                            source="helix",
                        )
                    )
                    hyps.append(
                        AngleHypothesis(
                            confidence=0.30,
                            phis_deg=np.full(L, -120.0),
                            psis_deg=np.full(L, 115.0),
                            source="sheet",
                        )
                    )
                    return hyps

                slots = make_pentamer_slots(seq, hyp_fn, frag_len=5)
                asm = assemble_greedy_backtrack(
                    seq,
                    slots,
                    lookahead=4,
                    relax_on_clash=True,
                    max_nodes=20_000,
                    anchors=anchors,
                )
                phis = [float(x) for x in asm.phis_deg]
                psis = [float(x) for x in asm.psis_deg]
                asm_note = (
                    f" Lever-assembly {asm.method} clashes={asm.n_clashes}"
                    f" ({asm.note})."
                )
            except Exception as e:
                asm_note = f" Lever-assembly skipped ({type(e).__name__})."

        # Boundary clash search is O(n²) per trial — keep cheap for short only.
        # Full SS pipeline also builds an N×N Cα matrix via clash_energy — skip
        # entirely above SS_PIPELINE_MAX_LEN (log-proven freeze at 25k / ~4.7 GB).
        do_boundary = len(seq) <= SS_BOUNDARY_OPT_MAX_LEN
        ss_note = ""
        if len(seq) > SS_PIPELINE_MAX_LEN:
            report(0.80, f"SS freeze skipped (>{SS_PIPELINE_MAX_LEN} aa — protects RAM)")
            ss_note = (
                f" SS skipped (length > {SS_PIPELINE_MAX_LEN}; "
                "full Cα distance matrix would freeze the OS)."
            )
        else:
            report(
                0.80,
                "Secondary-structure freeze"
                + (" / boundary optimize" if do_boundary else " (skip boundary opt)"),
            )
            try:
                from .structure_blocks import apply_ss_pipeline

                guard_rss("before_ss_pipeline")
                upgraded = apply_ss_pipeline(
                    seq,
                    phis,
                    psis,
                    conf=confs,
                    optimize_boundaries=do_boundary,
                    max_optimize_len=SS_BOUNDARY_OPT_MAX_LEN,
                )
                phis = list(map(float, upgraded["phis_deg"]))
                psis = list(map(float, upgraded["psis_deg"]))
                confs = list(map(float, upgraded["confidence"]))
                opt_tag = "boundary-opt" if do_boundary else "freeze-only"
                ss_note = (
                    f" SS-{opt_tag} {len(upgraded['blocks'])} block(s);"
                    f" clash_E={float(upgraded['clash_energy']):.3f}."
                )
            except MemoryGuardError:
                release_caches()
                raise
            except Exception as e:
                ss_note = f" SS-pipeline skipped ({type(e).__name__})."

        guard_rss("after_ss_stage")

        # Final O(N) lever polish after SS freeze so corrections are not overwritten
        if len(seq) <= LEVER_ASSEMBLY_MAX_LEN:
            try:
                from .clash_assembly import correct_lever_effect

                polished = correct_lever_effect(
                    seq, phis, psis, lookahead=4, relax_steps=14, anchors=anchors
                )
                phis = [float(x) for x in polished["phis_deg"]]
                psis = [float(x) for x in polished["psis_deg"]]
                if not asm_note:
                    asm_note = " Lever-polish"
                asm_note += (
                    f" post-SS repairs={polished['repairs']}"
                    f" e2e={polished['anchor_info'].get('end_to_end', 0):.1f}A."
                )
            except Exception:
                pass

        tmeta = None
        tnote = ""
        if len(seq) <= TERTIARY_MAX_LEN:
            phis, psis, confs, tmeta, tnote = maybe_tertiary(phis, psis, confs, lo=0.82, hi=0.94)

        report(0.96, "Assembling backbone" if len(seq) <= STRUCTURE_EXPORT_MAX_LEN else "Packing angles")
        structure = maybe_structure(phis, psis)
        # Compact long-chain payloads so JSON serialize cannot thrash the OS
        if len(seq) > DP_FULL_MAX_LEN:
            phis = [round(float(x), 1) for x in phis]
            psis = [round(float(x), 1) for x in psis]
            confs = []  # per-residue conf omitted; segment confidences remain
            for s in seg_info:
                if "confidence" in s:
                    s["confidence"] = round(float(s["confidence"]), 3)
        if progress:
            progress(
                {
                    "pct": 100.0,
                    "message": "Done",
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                    "eta_s": 0.0,
                    "n": len(seq),
                }
            )
        mode = "segment_assemble"
        if use_fast_tiles:
            mode = "tile_assemble"
        if tmeta:
            mode += "_tertiary"
        out = {
            "sequence": seq,
            "mode": mode,
            "segmentation": seg_info,
            "segmentation_score": total,
            "phis": phis,
            "psis": psis,
            "confidence": confs,
            "structure": structure,
            "model": self.ckpt_path,
            "device": str(self.dev),
            "note": (
                f"{'Greedy tiles' if use_fast_tiles else 'Segmented'} + "
                f"{'direct tile predict' if use_fast_tiles else 'overlap consensus'} + "
                f"{calib_note}.{contact_note}{asm_note}{ss_note}{tnote} "
                "Local assembly — not AlphaFold."
            ),
            "contacts": {
                "enabled": bool(contact_info.get("enabled")),
                "n_anchors": len(anchors or []),
                "anchors": [
                    {"i": a[0], "j": a[1], "dist": a[2]} for a in (anchors or [])
                ],
                "top": contact_info.get("contacts") or [],
                "mean_score": contact_info.get("mean_score", 0.0),
                "ckpt": contact_info.get("ckpt", ""),
            },
        }
        if tmeta:
            out["tertiary"] = tmeta
        return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("sequence")
    args = ap.parse_args()
    pred = FragmentPredictor()
    out = pred.predict_sequence(args.sequence)
    print(json.dumps({k: out[k] for k in out if k != "structure"}, indent=2))
    print("atoms", len(out["structure"]["atoms"]))


if __name__ == "__main__":
    main()
