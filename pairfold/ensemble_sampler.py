"""
Structural ensemble sampling from 5-mer dihedral distributions.

Models per-residue (φ, ψ) as mixtures of von Mises distributions on the
Ramachandran torus, samples K conformations with Metropolis–Hastings or
Gibbs sampling, then RMSD-clusters the ensemble and returns the top distinct
physically viable states.

Usage:
  python ensemble_sampler.py ACDEFGHIKLMN --k 64 --top 5 --method mh
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import vonmises
from sklearn.cluster import AgglomerativeClustering

from .assemble import build_backbone
from .clash_assembly import clash_energy, dihedrals_to_ca, has_steric_clash

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
TWO_PI = 2.0 * np.pi

# Classic Ramachandran modes (degrees) — used as mixture components / fallbacks
RAMA_MODES = (
    # name, phi_mu, psi_mu, kappa_phi, kappa_psi, prior_weight
    ("alpha", -57.0, -47.0, 12.0, 12.0, 0.45),
    ("beta", -120.0, 113.0, 8.0, 8.0, 0.35),
    ("ppii", -75.0, 145.0, 6.0, 6.0, 0.12),
    ("left", 57.0, 47.0, 5.0, 5.0, 0.08),
)


# ---------------------------------------------------------------------------
# Circular helpers
# ---------------------------------------------------------------------------


def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap radians to (−π, π]."""
    return (x + np.pi) % TWO_PI - np.pi


def deg_to_rad(a: np.ndarray) -> np.ndarray:
    return wrap_pi(np.asarray(a, dtype=np.float64) * DEG2RAD)


def rad_to_deg(a: np.ndarray) -> np.ndarray:
    return wrap_pi(np.asarray(a, dtype=np.float64)) * RAD2DEG


def circular_mean_rad(angles: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    if weights is None:
        weights = np.ones_like(angles)
    s = np.sum(weights * np.sin(angles))
    c = np.sum(weights * np.cos(angles))
    return float(math.atan2(s, c))


def circular_kappa(angles: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Approximate concentration from resultant length R̄ (Mardia)."""
    if weights is None:
        weights = np.ones_like(angles)
    w = weights / (np.sum(weights) + 1e-12)
    R = math.hypot(np.sum(w * np.sin(angles)), np.sum(w * np.cos(angles)))
    R = min(max(R, 1e-6), 0.999999)
    # Aoki / Banerjee approximation
    kappa = R * (2.0 - R * R) / (1.0 - R * R)
    return float(min(max(kappa, 0.5), 80.0))


# ---------------------------------------------------------------------------
# 1. Von Mises mixture on the Ramachandran torus
# ---------------------------------------------------------------------------


@dataclass
class VonMisesComponent:
    """Independent von Mises on φ and ψ (good lightweight Rama model)."""

    mu_phi: float  # radians
    mu_psi: float
    kappa_phi: float
    kappa_psi: float
    weight: float

    def log_prob(self, phi: float, psi: float) -> float:
        # scipy vonmises uses κ, loc=μ
        lp = vonmises.logpdf(phi, self.kappa_phi, loc=self.mu_phi)
        lp += vonmises.logpdf(psi, self.kappa_psi, loc=self.mu_psi)
        return float(lp)

    def sample(self, rng: np.random.Generator) -> Tuple[float, float]:
        phi = float(vonmises.rvs(self.kappa_phi, loc=self.mu_phi, random_state=rng))
        psi = float(vonmises.rvs(self.kappa_psi, loc=self.mu_psi, random_state=rng))
        return wrap_pi(np.array(phi)).item(), wrap_pi(np.array(psi)).item()


@dataclass
class DihedralMixture:
    """Mixture of von Mises components for one residue."""

    components: List[VonMisesComponent]

    def __post_init__(self) -> None:
        w = np.array([c.weight for c in self.components], dtype=np.float64)
        w = np.clip(w, 1e-8, None)
        w /= w.sum()
        for c, wi in zip(self.components, w):
            c.weight = float(wi)

    @property
    def weights(self) -> np.ndarray:
        return np.array([c.weight for c in self.components], dtype=np.float64)

    def log_prob(self, phi: float, psi: float) -> float:
        # logsumexp over components
        logs = np.array(
            [math.log(c.weight) + c.log_prob(phi, psi) for c in self.components],
            dtype=np.float64,
        )
        m = logs.max()
        return float(m + math.log(np.exp(logs - m).sum()))

    def sample(self, rng: np.random.Generator) -> Tuple[float, float]:
        k = int(rng.choice(len(self.components), p=self.weights))
        return self.components[k].sample(rng)

    @classmethod
    def from_rama_defaults(cls, scale_kappa: float = 1.0) -> "DihedralMixture":
        comps = []
        for _name, phi, psi, kp, ks, w in RAMA_MODES:
            comps.append(
                VonMisesComponent(
                    mu_phi=phi * DEG2RAD,
                    mu_psi=psi * DEG2RAD,
                    kappa_phi=kp * scale_kappa,
                    kappa_psi=ks * scale_kappa,
                    weight=w,
                )
            )
        return cls(comps)

    @classmethod
    def from_observations(
        cls,
        phis_deg: Sequence[float],
        psis_deg: Sequence[float],
        weights: Optional[Sequence[float]] = None,
        n_components: int = 3,
        min_obs: int = 4,
    ) -> "DihedralMixture":
        """
        Fit a small mixture by seeding at Ramachandran modes and soft-assigning
        observations (one EM-like pass). Falls back to defaults if too few obs.
        """
        phis = deg_to_rad(np.asarray(phis_deg, dtype=np.float64))
        psis = deg_to_rad(np.asarray(psis_deg, dtype=np.float64))
        if weights is None:
            w = np.ones(len(phis), dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            w = np.clip(w, 1e-8, None)

        if len(phis) < min_obs:
            base = cls.from_rama_defaults()
            # shrink toward observed mean if any
            if len(phis) > 0:
                mu_p = circular_mean_rad(phis, w)
                mu_s = circular_mean_rad(psis, w)
                kp = circular_kappa(phis, w)
                ks = circular_kappa(psis, w)
                base.components.append(
                    VonMisesComponent(mu_p, mu_s, kp, ks, weight=0.5)
                )
                return DihedralMixture(base.components)
            return base

        # Seed components at nearest default modes present in data density
        seeds = cls.from_rama_defaults().components[:n_components]
        # One soft-assignment + M-step
        comps: List[VonMisesComponent] = []
        for seed in seeds:
            # responsibility ∝ prior * lik
            log_r = np.array(
                [seed.log_prob(float(p), float(s)) for p, s in zip(phis, psis)]
            )
            r = np.exp(log_r - log_r.max())
            r *= w
            r_sum = r.sum() + 1e-12
            mu_p = circular_mean_rad(phis, r)
            mu_s = circular_mean_rad(psis, r)
            kp = max(circular_kappa(phis, r), 2.0)
            ks = max(circular_kappa(psis, r), 2.0)
            comps.append(
                VonMisesComponent(mu_p, mu_s, kp, ks, weight=float(r_sum))
            )
        return cls(comps)


@dataclass
class FragmentAngleHit:
    """One 5-mer (or k-mer) observation from the database."""

    sequence: str
    phis_deg: np.ndarray
    psis_deg: np.ndarray
    confidence: float = 1.0


class FragmentEnsembleDB:
    """Multi-hit fragment store: sequence → list of angle observations."""

    def __init__(self) -> None:
        self._hits: Dict[str, List[FragmentAngleHit]] = {}

    def add(self, hit: FragmentAngleHit) -> None:
        self._hits.setdefault(hit.sequence, []).append(hit)

    def add_many(self, hits: Sequence[FragmentAngleHit]) -> None:
        for h in hits:
            self.add(h)

    def get(self, seq: str) -> List[FragmentAngleHit]:
        return self._hits.get(seq, [])

    def __len__(self) -> int:
        return sum(len(v) for v in self._hits.values())


def build_residue_mixtures(
    sequence: str,
    db: FragmentEnsembleDB,
    window: int = 5,
    stride: int = 2,
) -> List[DihedralMixture]:
    """
    Pool overlapping k-mer hits onto each residue and fit a von Mises mixture.
    O(N) windows × hits_per_key (typically small).
    """
    n = len(sequence)
    phi_bags: List[List[float]] = [[] for _ in range(n)]
    psi_bags: List[List[float]] = [[] for _ in range(n)]
    w_bags: List[List[float]] = [[] for _ in range(n)]

    if n < window:
        starts = [0]
        win = n
    else:
        starts = list(range(0, n - window + 1, stride))
        if starts[-1] != n - window:
            starts.append(n - window)
        win = window

    for s in starts:
        key = sequence[s : s + win]
        hits = db.get(key)
        if not hits:
            # try any length-matched fallback: skip (defaults later)
            continue
        for hit in hits:
            L = min(win, len(hit.phis_deg))
            for k in range(L):
                phi_bags[s + k].append(float(hit.phis_deg[k]))
                psi_bags[s + k].append(float(hit.psis_deg[k]))
                w_bags[s + k].append(float(hit.confidence))

    mixtures: List[DihedralMixture] = []
    for i in range(n):
        if phi_bags[i]:
            mixtures.append(
                DihedralMixture.from_observations(phi_bags[i], psi_bags[i], w_bags[i])
            )
        else:
            mixtures.append(DihedralMixture.from_rama_defaults())
    return mixtures


# ---------------------------------------------------------------------------
# 2. MCMC samplers
# ---------------------------------------------------------------------------


@dataclass
class SampledConformation:
    phis_deg: np.ndarray
    psis_deg: np.ndarray
    ca: np.ndarray
    log_prior: float
    clash_E: float
    score: float  # higher = better (prior − clash)
    structure: dict = field(repr=False, default_factory=dict)


def _residue_log_prior(mix: DihedralMixture, phi_deg: float, psi_deg: float) -> float:
    return mix.log_prob(float(phi_deg) * DEG2RAD, float(psi_deg) * DEG2RAD)


def _chain_log_prior_from_deg(
    phis_deg: np.ndarray, psis_deg: np.ndarray, mixes: Sequence[DihedralMixture]
) -> float:
    return float(
        sum(
            _residue_log_prior(m, float(p), float(s))
            for m, p, s in zip(mixes, phis_deg, psis_deg)
        )
    )


def _score_angles(
    phis_deg: np.ndarray,
    psis_deg: np.ndarray,
    mixes: Sequence[DihedralMixture],
    clash_weight: float = 1.0,
    build_struct: bool = False,
    sequence: str = "",
    log_prior: Optional[float] = None,
) -> SampledConformation:
    lp = (
        float(log_prior)
        if log_prior is not None
        else _chain_log_prior_from_deg(phis_deg, psis_deg, mixes)
    )
    ca = dihedrals_to_ca(phis_deg, psis_deg)
    e = clash_energy(ca, soft=True)
    score = lp - clash_weight * e
    structure: dict = {}
    if build_struct:
        seq = sequence or ("X" * len(phis_deg))
        structure = build_backbone(seq, phis_deg.tolist(), psis_deg.tolist())
    return SampledConformation(
        phis_deg=np.asarray(phis_deg, dtype=np.float64),
        psis_deg=np.asarray(psis_deg, dtype=np.float64),
        ca=ca,
        log_prior=lp,
        clash_E=e,
        score=score,
        structure=structure,
    )


def sample_independent(
    mixes: Sequence[DihedralMixture],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw each residue independently from its mixture (fast proposal / init)."""
    n = len(mixes)
    phis = np.empty(n, dtype=np.float64)
    psis = np.empty(n, dtype=np.float64)
    for i, m in enumerate(mixes):
        phis[i], psis[i] = m.sample(rng)
    return rad_to_deg(phis), rad_to_deg(psis)


def metropolis_hastings(
    mixes: Sequence[DihedralMixture],
    n_samples: int,
    burn_in: int = 100,
    thin: int = 5,
    proposal_kappa: float = 4.0,
    clash_weight: float = 0.5,
    temperature: float = 1.0,
    seed: int = 0,
) -> List[SampledConformation]:
    """
    Metropolis–Hastings on the full (φ, ψ) chain.

    Proposal: independently resample a random residue from a von Mises
    centered at the current angle (κ = proposal_kappa), accept with
    min(1, exp(Δscore / T)) where score = log prior − λ · clash_energy.
    """
    rng = np.random.default_rng(seed)
    n = len(mixes)
    phis_deg, psis_deg = sample_independent(mixes, rng)
    # per-residue log prior cache
    res_lp = np.array(
        [_residue_log_prior(mixes[i], phis_deg[i], psis_deg[i]) for i in range(n)],
        dtype=np.float64,
    )
    current = _score_angles(
        phis_deg, psis_deg, mixes, clash_weight, log_prior=float(res_lp.sum())
    )

    out: List[SampledConformation] = []
    total_steps = burn_in + n_samples * thin

    for step in range(total_steps):
        i = int(rng.integers(0, n))
        prop_phi = float(
            vonmises.rvs(proposal_kappa, loc=phis_deg[i] * DEG2RAD, random_state=rng)
        )
        prop_psi = float(
            vonmises.rvs(proposal_kappa, loc=psis_deg[i] * DEG2RAD, random_state=rng)
        )
        new_phi = float(wrap_pi(np.array(prop_phi)) * RAD2DEG)
        new_psi = float(wrap_pi(np.array(prop_psi)) * RAD2DEG)

        new_phis = phis_deg.copy()
        new_psis = psis_deg.copy()
        new_phis[i] = new_phi
        new_psis[i] = new_psi

        new_res_lp = _residue_log_prior(mixes[i], new_phi, new_psi)
        new_lp = float(current.log_prior - res_lp[i] + new_res_lp)
        proposal = _score_angles(
            new_phis, new_psis, mixes, clash_weight, log_prior=new_lp
        )

        dE = (proposal.score - current.score) / max(temperature, 1e-8)
        if dE >= 0.0 or rng.random() < math.exp(min(dE, 50.0)):
            current = proposal
            phis_deg, psis_deg = new_phis, new_psis
            res_lp[i] = new_res_lp

        if step >= burn_in and (step - burn_in) % thin == 0:
            out.append(current)

    return out[:n_samples]


def gibbs_sample(
    mixes: Sequence[DihedralMixture],
    n_samples: int,
    burn_in: int = 20,
    thin: int = 2,
    clash_weight: float = 0.5,
    n_proposals: int = 3,
    seed: int = 0,
) -> List[SampledConformation]:
    """
    Gibbs-style sweep: for each residue, draw several mixture samples and
    keep the one that maximizes local score (prior + clash of full chain).
    """
    rng = np.random.default_rng(seed)
    n = len(mixes)
    phis_deg, psis_deg = sample_independent(mixes, rng)
    res_lp = np.array(
        [_residue_log_prior(mixes[i], phis_deg[i], psis_deg[i]) for i in range(n)],
        dtype=np.float64,
    )
    current = _score_angles(
        phis_deg, psis_deg, mixes, clash_weight, log_prior=float(res_lp.sum())
    )

    out: List[SampledConformation] = []
    total = burn_in + n_samples * thin

    for step in range(total):
        order = rng.permutation(n)
        for i in order:
            trials: List[SampledConformation] = []
            trial_store: List[Tuple[np.ndarray, np.ndarray, float]] = []
            # keep current as a candidate
            trials.append(current)
            trial_store.append((phis_deg, psis_deg, float(res_lp[i])))
            for _ in range(n_proposals):
                p, s = mixes[i].sample(rng)
                trial_phis = phis_deg.copy()
                trial_psis = psis_deg.copy()
                trial_phis[i] = p * RAD2DEG
                trial_psis[i] = s * RAD2DEG
                trial_res = _residue_log_prior(mixes[i], trial_phis[i], trial_psis[i])
                trial_lp = float(current.log_prior - res_lp[i] + trial_res)
                trial = _score_angles(
                    trial_phis, trial_psis, mixes, clash_weight, log_prior=trial_lp
                )
                trials.append(trial)
                trial_store.append((trial_phis, trial_psis, trial_res))
            # Softmax over proposals (temperature) → diversity vs pure argmax
            temp = 0.5
            scores = np.array([t.score for t in trials], dtype=np.float64)
            scores -= scores.max()
            probs = np.exp(scores / temp)
            probs /= probs.sum()
            pick = int(rng.choice(len(trials), p=probs))
            current = trials[pick]
            phis_deg, psis_deg = trial_store[pick][0], trial_store[pick][1]
            res_lp[i] = trial_store[pick][2]

        if step >= burn_in and (step - burn_in) % thin == 0:
            out.append(current)

    return out[:n_samples]


# ---------------------------------------------------------------------------
# 3–4. RMSD clustering → top distinct states
# ---------------------------------------------------------------------------


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Kabsch RMSD after optimal rotation (centroids aligned)."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T
    Prot = Pc @ R
    return float(np.sqrt(((Prot - Qc) ** 2).sum() / P.shape[0]))


def pairwise_rmsd_matrix(cas: Sequence[np.ndarray]) -> np.ndarray:
    m = len(cas)
    D = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            d = kabsch_rmsd(cas[i], cas[j])
            D[i, j] = D[j, i] = d
    return D


@dataclass
class EnsembleState:
    rank: int
    cluster_size: int
    rmsd_to_centroid: float
    conformation: SampledConformation
    members: List[int]


def select_top_states(
    samples: Sequence[SampledConformation],
    top_k: int = 5,
    rmsd_thresh: float = 2.0,
    sequence: str = "",
) -> List[EnsembleState]:
    """
    Cluster conformations by Cα RMSD and return up to `top_k` distinct states.

    Cluster representatives = medoid (min mean RMSD to members).
    Clusters ranked by medoid physical score (log-prior − clash).
    """
    if not samples:
        return []

    cas = [s.ca for s in samples]
    D = pairwise_rmsd_matrix(cas)

    # Agglomerative clustering with distance threshold (Å)
    n = len(samples)
    if n == 1:
        labels = np.array([0])
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=rmsd_thresh,
        )
        labels = clustering.fit_predict(D)

    states: List[EnsembleState] = []
    for lab in sorted(set(labels.tolist())):
        members = np.where(labels == lab)[0]
        # medoid
        sub = D[np.ix_(members, members)]
        mean_d = sub.mean(axis=1)
        medoid_local = int(members[int(np.argmin(mean_d))])
        conf = samples[medoid_local]
        if sequence:
            conf.structure = build_backbone(
                sequence, conf.phis_deg.tolist(), conf.psis_deg.tolist()
            )
        states.append(
            EnsembleState(
                rank=0,
                cluster_size=int(members.size),
                rmsd_to_centroid=0.0,
                conformation=conf,
                members=members.tolist(),
            )
        )

    # rank by score, prefer clash-free, then larger clusters as tie-break
    states.sort(
        key=lambda s: (
            s.conformation.score,
            -s.conformation.clash_E,
            s.cluster_size,
        ),
        reverse=True,
    )
    for i, st in enumerate(states[:top_k]):
        st.rank = i + 1
    return states[:top_k]


def generate_ensemble(
    sequence: str,
    db: FragmentEnsembleDB,
    k_samples: int = 64,
    top_k: int = 5,
    method: str = "mh",
    window: int = 5,
    seed: int = 0,
    rmsd_thresh: float = 2.0,
) -> Dict:
    """Full pipeline: fit mixtures → sample → cluster → top states."""
    seq = "".join(c for c in sequence.upper() if c.isalpha())
    mixes = build_residue_mixtures(seq, db, window=window)
    if method.lower() in ("mh", "metropolis", "metropolis-hastings"):
        samples = metropolis_hastings(mixes, n_samples=k_samples, seed=seed)
    elif method.lower() in ("gibbs",):
        samples = gibbs_sample(mixes, n_samples=k_samples, seed=seed)
    else:
        raise ValueError(f"unknown method {method}")

    # de-duplicate near-identical samples before clustering (optional speed)
    states = select_top_states(samples, top_k=top_k, rmsd_thresh=rmsd_thresh, sequence=seq)

    return {
        "sequence": seq,
        "method": method,
        "n_samples": len(samples),
        "n_states": len(states),
        "states": [
            {
                "rank": st.rank,
                "cluster_size": st.cluster_size,
                "score": st.conformation.score,
                "log_prior": st.conformation.log_prior,
                "clash_energy": st.conformation.clash_E,
                "has_clash": has_steric_clash(st.conformation.ca),
                "phis_deg": st.conformation.phis_deg.tolist(),
                "psis_deg": st.conformation.psis_deg.tolist(),
                "ca": st.conformation.ca.tolist(),
                "structure": st.conformation.structure,
            }
            for st in states
        ],
    }


# ---------------------------------------------------------------------------
# Demo database + CLI
# ---------------------------------------------------------------------------


def _synthetic_db(seed: int = 0) -> FragmentEnsembleDB:
    """Build a toy multi-hit 5-mer DB around helix / sheet modes."""
    rng = np.random.default_rng(seed)
    db = FragmentEnsembleDB()
    keys = [
        "AAAAA",
        "AAAAL",
        "AAALA",
        "AALAA",
        "ALAAA",
        "LAAAA",
        "VVVVV",
        "AAVAA",
        "AVAAA",
        "VAAAA",
        "AAAVA",
        "AAAAV",
    ]
    for key in keys:
        for _ in range(12):
            if "V" in key:
                phi = rng.normal(-120, 18, size=5)
                psi = rng.normal(120, 20, size=5)
                conf = float(rng.uniform(0.6, 0.95))
            else:
                phi = rng.normal(-57, 10, size=5)
                psi = rng.normal(-47, 10, size=5)
                conf = float(rng.uniform(0.7, 0.99))
            db.add(
                FragmentAngleHit(
                    sequence=key,
                    phis_deg=phi,
                    psis_deg=psi,
                    confidence=conf,
                )
            )
    return db


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample structural ensembles from 5-mer dihedral mixtures")
    ap.add_argument("sequence", nargs="?", default="AAAAALAAAAAVAA")
    ap.add_argument("--k", type=int, default=48, help="number of MCMC samples")
    ap.add_argument("--top", type=int, default=5, help="top distinct states to keep")
    ap.add_argument("--method", choices=("mh", "gibbs"), default="mh")
    ap.add_argument("--rmsd", type=float, default=2.0, help="RMSD cluster threshold (A)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    db = _synthetic_db(args.seed)
    result = generate_ensemble(
        args.sequence,
        db,
        k_samples=args.k,
        top_k=args.top,
        method=args.method,
        seed=args.seed,
        rmsd_thresh=args.rmsd,
    )

    # compact print (drop heavy structure coords in stdout summary)
    summary = {
        "sequence": result["sequence"],
        "method": result["method"],
        "n_samples": result["n_samples"],
        "n_states": result["n_states"],
        "states": [
            {
                "rank": s["rank"],
                "cluster_size": s["cluster_size"],
                "score": round(s["score"], 3),
                "log_prior": round(s["log_prior"], 3),
                "clash_energy": round(s["clash_energy"], 4),
                "has_clash": s["has_clash"],
                "phi_mean": round(float(np.mean(s["phis_deg"])), 1),
                "psi_mean": round(float(np.mean(s["psis_deg"])), 1),
            }
            for s in result["states"]
        ],
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote full ensemble -> {args.output}")


if __name__ == "__main__":
    main()
