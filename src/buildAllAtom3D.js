/**
 * Client-side Stage-2/3: backbone N/CA/C/O + sidechain heavy atoms from φ/ψ.
 * Used by the 3D viewer so all-atom shows even when the API omits structure.atoms.
 */
import { buildBackbone3D } from "./buildBackbone3D.js";
import { AA_BY_CODE } from "./data/aminoAcids.js";

const DEG = Math.PI / 180;

function v(x, y, z) {
  return { x, y, z };
}
function add(a, b) {
  return v(a.x + b.x, a.y + b.y, a.z + b.z);
}
function sub(a, b) {
  return v(a.x - b.x, a.y - b.y, a.z - b.z);
}
function scale(a, s) {
  return v(a.x * s, a.y * s, a.z * s);
}
function cross(a, b) {
  return v(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
function len(a) {
  return Math.hypot(a.x, a.y, a.z);
}
function norm(a) {
  const L = len(a) || 1;
  return scale(a, 1 / L);
}

function placeAtom(A, B, C, bondLength, bondAngleDeg, dihedralDeg) {
  const bc = norm(sub(C, B));
  let n = cross(sub(B, A), bc);
  if (len(n) < 1e-8) {
    n = Math.abs(bc.x) < 0.9 ? v(1, 0, 0) : v(0, 1, 0);
    n = norm(cross(bc, n));
  } else {
    n = norm(n);
  }
  const m = cross(n, bc);
  const theta = bondAngleDeg * DEG;
  const phi = dihedralDeg * DEG;
  const st = Math.sin(theta);
  const dir = add(
    add(scale(bc, -Math.cos(theta)), scale(m, st * Math.cos(phi))),
    scale(n, st * Math.sin(phi)),
  );
  return add(C, scale(norm(dir), bondLength));
}

function placeCb(N, CA, C) {
  return placeAtom(C, N, CA, 1.522, 110.1, -122.5);
}

const ROT = {
  A: [[]],
  G: [[]],
  C: [[-60], [180], [60]],
  S: [[-60], [60], [180]],
  T: [[-60], [60], [180]],
  V: [[-60], [180], [60]],
  I: [[-60, 170], [60, 170]],
  L: [[-60, 180], [180, 180]],
  D: [[-70, 0], [180, 0]],
  N: [[-70, 0], [180, 0]],
  E: [[-70, 180, 0], [180, 180, 0]],
  Q: [[-70, 180, 0], [180, 180, 0]],
  M: [[-70, 180, 60], [180, 180, 60]],
  K: [[-70, 180, 180, 180]],
  R: [[-70, 180, 180, 180]],
  P: [[-20, 30]],
  H: [[-70, -70], [180, -70]],
  F: [[-70, 90], [180, 90]],
  Y: [[-70, 90], [180, 90]],
  W: [[-70, 90], [180, 90]],
};

function chi(chis, i, d = 180) {
  return chis[i] != null ? chis[i] : d;
}

function buildSC(aa, N, CA, C, chis) {
  if (aa === "G") return {};
  const cb = placeCb(N, CA, C);
  const out = { CB: cb };
  const c1 = chi(chis, 0, -60);
  if (aa === "A") return out;
  if (aa === "C") {
    out.SG = placeAtom(N, CA, cb, 1.808, 110.8, c1);
    return out;
  }
  if (aa === "S") {
    out.OG = placeAtom(N, CA, cb, 1.417, 111.0, c1);
    return out;
  }
  if (aa === "T") {
    out.OG1 = placeAtom(N, CA, cb, 1.433, 109.6, c1);
    out.CG2 = placeAtom(N, CA, cb, 1.529, 111.6, c1 + 120);
    return out;
  }
  if (aa === "V") {
    out.CG1 = placeAtom(N, CA, cb, 1.527, 110.9, c1);
    out.CG2 = placeAtom(N, CA, cb, 1.527, 110.9, c1 + 120);
    return out;
  }
  if (aa === "I") {
    const c2 = chi(chis, 1, 170);
    out.CG1 = placeAtom(N, CA, cb, 1.53, 111, c1);
    out.CG2 = placeAtom(N, CA, cb, 1.53, 111, c1 - 120);
    out.CD1 = placeAtom(CA, cb, out.CG1, 1.516, 113.8, c2);
    return out;
  }
  if (aa === "L") {
    const c2 = chi(chis, 1, 180);
    out.CG = placeAtom(N, CA, cb, 1.53, 116.3, c1);
    out.CD1 = placeAtom(CA, cb, out.CG, 1.524, 111.2, c2);
    out.CD2 = placeAtom(CA, cb, out.CG, 1.524, 111.2, c2 + 120);
    return out;
  }
  if (aa === "D" || aa === "N") {
    const c2 = chi(chis, 1, 0);
    out.CG = placeAtom(N, CA, cb, 1.516, 112.8, c1);
    if (aa === "D") {
      out.OD1 = placeAtom(CA, cb, out.CG, 1.249, 118.6, c2);
      out.OD2 = placeAtom(CA, cb, out.CG, 1.249, 118.6, c2 + 180);
    } else {
      out.OD1 = placeAtom(CA, cb, out.CG, 1.231, 120.8, c2);
      out.ND2 = placeAtom(CA, cb, out.CG, 1.328, 116.6, c2 + 180);
    }
    return out;
  }
  if (aa === "E" || aa === "Q") {
    const c2 = chi(chis, 1, 180);
    const c3 = chi(chis, 2, 0);
    out.CG = placeAtom(N, CA, cb, 1.522, 114.2, c1);
    out.CD = placeAtom(CA, cb, out.CG, 1.524, 114.2, c2);
    if (aa === "E") {
      out.OE1 = placeAtom(cb, out.CG, out.CD, 1.249, 118.6, c3);
      out.OE2 = placeAtom(cb, out.CG, out.CD, 1.249, 118.6, c3 + 180);
    } else {
      out.OE1 = placeAtom(cb, out.CG, out.CD, 1.231, 120.8, c3);
      out.NE2 = placeAtom(cb, out.CG, out.CD, 1.328, 116.6, c3 + 180);
    }
    return out;
  }
  if (aa === "M") {
    const c2 = chi(chis, 1, 180);
    const c3 = chi(chis, 2, 60);
    out.CG = placeAtom(N, CA, cb, 1.522, 114, c1);
    out.SD = placeAtom(CA, cb, out.CG, 1.803, 112.8, c2);
    out.CE = placeAtom(cb, out.CG, out.SD, 1.791, 100.9, c3);
    return out;
  }
  if (aa === "K") {
    const c2 = chi(chis, 1, 180);
    const c3 = chi(chis, 2, 180);
    const c4 = chi(chis, 3, 180);
    out.CG = placeAtom(N, CA, cb, 1.53, 114.2, c1);
    out.CD = placeAtom(CA, cb, out.CG, 1.521, 111.9, c2);
    out.CE = placeAtom(cb, out.CG, out.CD, 1.521, 111.9, c3);
    out.NZ = placeAtom(out.CG, out.CD, out.CE, 1.489, 111.9, c4);
    return out;
  }
  if (aa === "R") {
    const c2 = chi(chis, 1, 180);
    const c3 = chi(chis, 2, 180);
    const c4 = chi(chis, 3, 180);
    out.CG = placeAtom(N, CA, cb, 1.52, 114.2, c1);
    out.CD = placeAtom(CA, cb, out.CG, 1.52, 111.9, c2);
    out.NE = placeAtom(cb, out.CG, out.CD, 1.461, 112, c3);
    out.CZ = placeAtom(out.CG, out.CD, out.NE, 1.33, 124.2, c4);
    out.NH1 = placeAtom(out.CD, out.NE, out.CZ, 1.326, 120.3, 0);
    out.NH2 = placeAtom(out.CD, out.NE, out.CZ, 1.326, 120.3, 180);
    return out;
  }
  if (aa === "P") {
    const c2 = chi(chis, 1, 30);
    out.CG = placeAtom(N, CA, cb, 1.492, 104, c1);
    out.CD = placeAtom(CA, cb, out.CG, 1.503, 105, c2);
    return out;
  }
  if (aa === "H") {
    const c2 = chi(chis, 1, -70);
    out.CG = placeAtom(N, CA, cb, 1.497, 113.8, c1);
    out.ND1 = placeAtom(CA, cb, out.CG, 1.378, 122.8, c2);
    out.CD2 = placeAtom(CA, cb, out.CG, 1.356, 130.8, c2 + 180);
    out.CE1 = placeAtom(cb, out.CG, out.ND1, 1.347, 108.9, 180);
    out.NE2 = placeAtom(cb, out.CG, out.CD2, 1.374, 109.8, 180);
    return out;
  }
  if (aa === "F" || aa === "Y") {
    const c2 = chi(chis, 1, 90);
    out.CG = placeAtom(N, CA, cb, 1.502, 113.8, c1);
    out.CD1 = placeAtom(CA, cb, out.CG, 1.389, 120.7, c2);
    out.CD2 = placeAtom(CA, cb, out.CG, 1.389, 120.7, c2 + 180);
    out.CE1 = placeAtom(cb, out.CG, out.CD1, 1.382, 120.7, 180);
    out.CE2 = placeAtom(cb, out.CG, out.CD2, 1.382, 120.7, 180);
    out.CZ = placeAtom(out.CG, out.CD1, out.CE1, 1.382, 120, 0);
    if (aa === "Y") out.OH = placeAtom(out.CD1, out.CE1, out.CZ, 1.376, 119.9, 180);
    return out;
  }
  if (aa === "W") {
    const c2 = chi(chis, 1, 90);
    out.CG = placeAtom(N, CA, cb, 1.498, 113.6, c1);
    out.CD1 = placeAtom(CA, cb, out.CG, 1.365, 126.9, c2);
    out.CD2 = placeAtom(CA, cb, out.CG, 1.433, 126.9, c2 + 180);
    out.NE1 = placeAtom(cb, out.CG, out.CD1, 1.374, 110.3, 180);
    out.CE2 = placeAtom(out.CG, out.CD1, out.NE1, 1.413, 106.5, 0);
    out.CE3 = placeAtom(cb, out.CG, out.CD2, 1.4, 133.8, 180);
    out.CZ2 = placeAtom(out.CD1, out.NE1, out.CE2, 1.394, 122, 180);
    out.CZ3 = placeAtom(out.CG, out.CD2, out.CE3, 1.394, 118, 180);
    out.CH2 = placeAtom(out.NE1, out.CE2, out.CZ2, 1.368, 120, 0);
    return out;
  }
  return out;
}

const PARENT = {
  CB: "CA",
  SG: "CB",
  OG: "CB",
  OG1: "CB",
  CG: "CB",
  CG1: "CB",
  CG2: "CB",
  CD: "CG",
  CD1: "CG",
  CD2: "CG",
  SD: "CG",
  CE: "CD",
  CE1: "CD1",
  CE2: "CD2",
  CE3: "CD2",
  NZ: "CE",
  NE: "CD",
  CZ: "CE1",
  CZ2: "CE2",
  CZ3: "CE3",
  OH: "CZ",
  OD1: "CG",
  OD2: "CG",
  ND2: "CG",
  OE1: "CD",
  OE2: "CD",
  NE2: "CD",
  NH1: "CZ",
  NH2: "CZ",
  NE1: "CD1",
  CH2: "CZ2",
  ND1: "CG",
};

/**
 * @param {object} input — { sequence, phis, psis, codes? }
 * @returns {{ residues, atoms, bonds, allAtom: true }}
 */
export function buildAllAtom3D(input) {
  const n = input.length || input.phis?.length || 0;
  if (!n || !input.phis?.length) throw new Error("No angles for all-atom 3D");
  const seq =
    input.sequence ||
    (input.codes ? input.codes.join("") : "") ||
    "X".repeat(n);

  const bb = buildBackbone3D({
    length: n,
    sequence: seq,
    codes: [...seq],
    abbrs: [...seq].map((c) => AA_BY_CODE[c]?.abbr || c),
    colors: [...seq].map((c) => AA_BY_CODE[c]?.color || "#8a95a1"),
    phis: input.phis,
    psis: input.psis,
    omega: input.omega ?? 180,
    caOnly: false,
  });

  const atoms = [];
  const bonds = [];
  const resIdx = [];

  for (let i = 0; i < n; i++) {
    const r = bb.residues[i];
    const map = {};
    const push = (name, xyz, sidechain = false) => {
      const idx = atoms.length;
      const el =
        name === "CA"
          ? "CA"
          : name.startsWith("N")
            ? "N"
            : name.startsWith("O")
              ? "O"
              : name.startsWith("S")
                ? "S"
                : "C";
      atoms.push({
        x: xyz.x,
        y: xyz.y,
        z: xyz.z,
        element: el,
        name,
        atomName: name,
        residue: i,
        code: r.code,
        sidechain,
        color: sidechain
          ? r.color
          : name === "CA"
            ? r.color
            : name === "N"
              ? "#4b6bfb"
              : name.startsWith("O")
                ? "#e4572e"
                : "#8a95a1",
      });
      map[name] = idx;
      return idx;
    };

    push("N", r.N);
    push("CA", r.CA);
    push("C", r.C);
    // carbonyl O
    let O;
    if (i + 1 < n) {
      O = placeAtom(bb.residues[i + 1].N, r.CA, r.C, 1.231, 120.8, 0);
    } else {
      O = placeAtom(r.N, r.CA, r.C, 1.231, 120.8, 180);
    }
    push("O", O);
    bonds.push([map.N, map.CA], [map.CA, map.C], [map.C, map.O]);
    if (i > 0) bonds.push([resIdx[i - 1].C, map.N]);

    const aa = seq[i] || "A";
    const rot = (ROT[aa] || [[]])[0];
    const sc = buildSC(aa, r.N, r.CA, r.C, rot);
    for (const [name, xyz] of Object.entries(sc)) {
      push(name, xyz, true);
      const p = PARENT[name];
      if (p && map[p] != null) bonds.push([map[p], map[name]]);
    }
    // ring closures
    for (const [a, b] of [
      ["CE1", "CZ"],
      ["CE2", "CZ"],
      ["NE2", "CE1"],
      ["NE1", "CE2"],
      ["CZ2", "CH2"],
      ["CZ3", "CH2"],
    ]) {
      if (map[a] != null && map[b] != null) bonds.push([map[a], map[b]]);
    }
    resIdx.push(map);
  }

  return {
    residues: bb.residues,
    atoms,
    bonds,
    allAtom: true,
  };
}

/** True if structure looks like backbone-only (no sidechains). */
export function isBackboneOnly(structure, nRes) {
  const atoms = structure?.atoms;
  if (!atoms?.length) return true;
  const hasSC = atoms.some(
    (a) => a.sidechain || ["CB", "CG", "SG", "OG"].includes(String(a.name || "").toUpperCase()),
  );
  if (hasSC) return false;
  // ~3 atoms/res = N/CA/C only
  return atoms.length <= nRes * 3 + 2;
}
