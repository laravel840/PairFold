import { AA_BY_CODE, AA_CODES } from "./aminoAcids.js";

/**
 * Peptide geometry constants (approximate, Å / degrees).
 */
export const PEPTIDE = {
  caCaDistance: 3.8,
  omega: 180,
  minLength: 2,
  maxLength: 5,
};

/**
 * Pair-specific adjustments for well-known local motifs.
 */
const PAIR_ADJUST = {
  "*P": { psi1Delta: 10, phi2Delta: 0, label: "X–Pro kink" },
  "P*": { psi1Delta: 0, phi2Delta: 5, label: "Pro–X turn bias" },
  GG: { psi1Delta: -20, phi2Delta: 15, label: "flexible Gly–Gly" },
  NG: { psi1Delta: -40, phi2Delta: 20, label: "type I′ / II′ turn" },
  DG: { psi1Delta: -35, phi2Delta: 18, label: "type I′ / II′ turn" },
  PG: { psi1Delta: 5, phi2Delta: 25, label: "Pro–Gly turn" },
};

/** Short 3-residue motifs. */
const TRI_MOTIF = {
  PGP: "Pro–Gly–Pro turn",
  NPG: "Asn–Pro–Gly turn",
  DPG: "Asp–Pro–Gly turn",
  GPG: "Gly–Pro–Gly hinge",
  PPP: "polyproline stretch",
  GGG: "flexible polyglycine",
};

function getAdjust(code1, code2) {
  if (PAIR_ADJUST[code1 + code2]) return PAIR_ADJUST[code1 + code2];
  if (code2 === "P" && PAIR_ADJUST["*P"]) return PAIR_ADJUST["*P"];
  if (code1 === "P" && PAIR_ADJUST["P*"]) return PAIR_ADJUST["P*"];
  return { psi1Delta: 0, phi2Delta: 0, label: "coil preference" };
}

function bendFrom(psiPrev, phiNext) {
  return normalizeAngle(90 + psiPrev * 0.35 + phiNext * 0.25);
}

function requireAA(code, context) {
  const aa = AA_BY_CODE[code];
  if (!aa) throw new Error(`Unknown amino acid in ${context}`);
  return aa;
}

function resolveMotif(codes, bondLabels) {
  for (let i = 0; i <= codes.length - 3; i++) {
    const key = codes.slice(i, i + 3).join("");
    if (TRI_MOTIF[key]) return TRI_MOTIF[key];
  }
  const labels = bondLabels.filter((l) => l !== "coil preference");
  return labels.length ? labels.join(" · ") : "coil preference";
}

/**
 * On-demand peptide angles for length 2–5.
 * Does not materialize the full combinatorial space.
 *
 * @param {string[]} codes one-letter codes, length 2–5
 */
export function getPeptideAngles(codes) {
  if (!Array.isArray(codes) || codes.length < PEPTIDE.minLength || codes.length > PEPTIDE.maxLength) {
    throw new Error(
      `Peptide length must be ${PEPTIDE.minLength}–${PEPTIDE.maxLength} (got ${codes?.length ?? 0})`,
    );
  }

  const seq = codes.join("");
  const residues = codes.map((c) => requireAA(c, seq));
  const n = codes.length;

  const phis = residues.map((r) => r.phi);
  const psis = residues.map((r) => r.psi);
  const bondLabels = [];

  for (let i = 0; i < n - 1; i++) {
    const adj = getAdjust(codes[i], codes[i + 1]);
    bondLabels.push(adj.label);
    psis[i] += adj.psi1Delta;
    phis[i + 1] += adj.phi2Delta;
  }

  const bends = [];
  for (let i = 0; i < n - 1; i++) {
    bends.push(bendFrom(psis[i], phis[i + 1]));
  }

  const peptide = {
    length: n,
    codes: [...codes],
    names: residues.map((r) => r.name),
    colors: residues.map((r) => r.color),
    abbrs: residues.map((r) => r.abbr),
    phis,
    psis,
    bends,
    bend: bends[0],
    omega: PEPTIDE.omega,
    motif: resolveMotif(codes, bondLabels),
    caCa: PEPTIDE.caCaDistance,
  };

  // Convenience aliases used by older UI paths
  for (let i = 0; i < n; i++) {
    peptide[`code${i + 1}`] = codes[i];
    peptide[`name${i + 1}`] = residues[i].name;
    peptide[`color${i + 1}`] = residues[i].color;
    peptide[`abbr${i + 1}`] = residues[i].abbr;
    peptide[`phi${i + 1}`] = phis[i];
    peptide[`psi${i + 1}`] = psis[i];
  }
  if (bends[1] != null) peptide.bend2 = bends[1];

  return peptide;
}

export function getDipeptideAngles(code1, code2) {
  return getPeptideAngles([code1, code2]);
}

export function getTripeptideAngles(code1, code2, code3) {
  return getPeptideAngles([code1, code2, code3]);
}

export function normalizeAngle(deg) {
  let a = deg % 360;
  if (a > 180) a -= 360;
  if (a <= -180) a += 360;
  return a;
}

/**
 * Resolve free text to an ordered list of one-letter codes (any length 1–5), or null.
 * Accepts "AGPV", "A-G-P-V", "Ala-Gly-Pro-Val".
 */
export function codesFromQuery(raw) {
  if (!raw || !String(raw).trim()) return null;
  const text = String(raw).trim();

  const parts = text.split(/[\s,–—\-_/]+/).filter(Boolean);
  if (parts.length >= 1 && parts.length <= PEPTIDE.maxLength) {
    const looksNamed = parts.some((p) => p.length > 1);
    if (looksNamed || parts.length >= PEPTIDE.minLength) {
      const fromAbbr = parts.map((p) => {
        const u = p.toUpperCase();
        if (u.length === 1 && AA_BY_CODE[u]) return u;
        const hit = AA_CODES.find((c) => {
          const aa = AA_BY_CODE[c];
          return (
            aa.abbr.toUpperCase() === u ||
            aa.name.toUpperCase() === u ||
            aa.name.toUpperCase().startsWith(u)
          );
        });
        return hit || null;
      });
      if (fromAbbr.every(Boolean)) return fromAbbr;
    }
  }

  const letters = text.toUpperCase().replace(/[^A-Z]/g, "");
  if (
    letters.length >= 1 &&
    letters.length <= PEPTIDE.maxLength &&
    [...letters].every((c) => AA_BY_CODE[c])
  ) {
    return [...letters];
  }

  return null;
}

/** Exact sequence of length 2–5, or null. */
export function parseSequenceQuery(raw) {
  const codes = codesFromQuery(raw);
  if (
    codes &&
    codes.length >= PEPTIDE.minLength &&
    codes.length <= PEPTIDE.maxLength
  ) {
    return codes;
  }
  return null;
}

/**
 * Build search suggestions without materializing all combinations.
 * Exact match (if length ≥ 2) plus next-residue expansions (≤ 20).
 */
export function suggestPeptides(raw) {
  const codes = codesFromQuery(raw);
  if (!codes || !codes.length) return [];

  const suggestions = [];

  if (codes.length >= PEPTIDE.minLength && codes.length <= PEPTIDE.maxLength) {
    suggestions.push(getPeptideAngles(codes));
  }

  if (codes.length < PEPTIDE.maxLength) {
    for (const next of AA_CODES) {
      suggestions.push(getPeptideAngles([...codes, next]));
      if (suggestions.length >= 21) break;
    }
  }

  return suggestions;
}

/** @deprecated Do not materialize combinatorial space; use getPeptideAngles / suggestPeptides. */
export function allPeptides() {
  console.warn(
    "allPeptides() is deprecated — use getPeptideAngles / suggestPeptides (on-demand).",
  );
  return [];
}
