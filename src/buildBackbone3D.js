/**
 * Build peptide backbone coordinates (N, CA, C) from φ / ψ / ω angles.
 * Distances in Ångströms; angles in degrees.
 */

const DEG = Math.PI / 180;

const LEN = {
  N_CA: 1.458,
  CA_C: 1.525,
  C_N: 1.329,
};

const ANG = {
  N_CA_C: 111.2,
  CA_C_N: 116.2,
  C_N_CA: 121.7,
};

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
  return v(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x,
  );
}

function len(a) {
  return Math.hypot(a.x, a.y, a.z);
}

function norm(a) {
  const L = len(a) || 1;
  return scale(a, 1 / L);
}

/** Place atom D given A–B–C, bond length C–D, angle B–C–D, dihedral A–B–C–D. */
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

/**
 * @param {object} peptide from getDipeptideAngles / getTripeptideAngles
 */
function codeAt(peptide, i) {
  if (peptide.codes) return peptide.codes[i];
  if (peptide.sequence) return peptide.sequence[i] || "X";
  return "X";
}

export function buildBackbone3D(peptide) {
  const nRes = peptide.length;
  const omega = peptide.omega ?? 180;
  const caOnly = Boolean(peptide.caOnly) || nRes > 400;
  const residues = [];

  const N0 = v(0, 0, 0);
  const CA0 = v(LEN.N_CA, 0, 0);
  const C0 = placeAtom(v(0, 1, 0), N0, CA0, LEN.CA_C, ANG.N_CA_C, 0);

  const c0 = codeAt(peptide, 0);
  residues.push({
    index: 0,
    code: c0,
    abbr: peptide.abbrs?.[0] || c0,
    color: peptide.colors?.[0] || "#8a95a1",
    N: N0,
    CA: CA0,
    C: C0,
    phi: peptide.phis[0],
    psi: peptide.psis[0],
  });

  for (let i = 0; i < nRes - 1; i++) {
    const res = residues[i];

    // N(i+1): torsion N(i)–CA(i)–C(i)–N(i+1) = ψ(i)
    const Nnext = placeAtom(
      res.N,
      res.CA,
      res.C,
      LEN.C_N,
      ANG.CA_C_N,
      peptide.psis[i],
    );

    // CA(i+1): torsion CA(i)–C(i)–N(i+1)–CA(i+1) = ω
    const CAnext = placeAtom(
      res.CA,
      res.C,
      Nnext,
      LEN.N_CA,
      ANG.C_N_CA,
      omega,
    );

    // C(i+1): torsion C(i)–N(i+1)–CA(i+1)–C(i+1) = φ(i+1)
    const Cnext = placeAtom(
      res.C,
      Nnext,
      CAnext,
      LEN.CA_C,
      ANG.N_CA_C,
      peptide.phis[i + 1],
    );

    const ci = codeAt(peptide, i + 1);
    residues.push({
      index: i + 1,
      code: ci,
      abbr: peptide.abbrs?.[i + 1] || ci,
      color: peptide.colors?.[i + 1] || "#8a95a1",
      N: Nnext,
      CA: CAnext,
      C: Cnext,
      phi: peptide.phis[i + 1],
      psi: peptide.psis[i + 1],
    });
  }

  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (const r of residues) {
    cx += r.CA.x;
    cy += r.CA.y;
    cz += r.CA.z;
  }
  cx /= residues.length;
  cy /= residues.length;
  cz /= residues.length;

  for (const r of residues) {
    r.CA = v(r.CA.x - cx, r.CA.y - cy, r.CA.z - cz);
    if (caOnly) {
      // Trace mode only keeps Cα after the build walk
      r.N = undefined;
      r.C = undefined;
    } else {
      r.N = v(r.N.x - cx, r.N.y - cy, r.N.z - cz);
      r.C = v(r.C.x - cx, r.C.y - cy, r.C.z - cz);
    }
  }

  if (caOnly) {
    return { residues, atoms: [], bonds: [], caOnly: true };
  }

  const atoms = [];
  const bonds = [];

  for (const r of residues) {
    const iN = atoms.length;
    atoms.push({ ...r.N, element: "N", residue: r.index, color: "#4b6bfb" });
    const iCA = atoms.length;
    atoms.push({
      ...r.CA,
      element: "CA",
      residue: r.index,
      color: r.color,
      code: r.code,
      abbr: r.abbr,
    });
    const iC = atoms.length;
    atoms.push({ ...r.C, element: "C", residue: r.index, color: "#6b7280" });
    bonds.push([iN, iCA], [iCA, iC]);
  }

  for (let i = 0; i < residues.length - 1; i++) {
    bonds.push([i * 3 + 2, (i + 1) * 3]);
  }

  return { residues, atoms, bonds };
}
