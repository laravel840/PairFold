import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  CSS2DRenderer,
  CSS2DObject,
} from "three/addons/renderers/CSS2DRenderer.js";
import { buildBackbone3D } from "./buildBackbone3D.js";
import { buildAllAtom3D, isBackboneOnly } from "./buildAllAtom3D.js";
import { AA_BY_CODE } from "./data/aminoAcids.js";

/** @type {Map<HTMLElement, object>} */
const actives = new Map();

function disposeObject(obj) {
  if (obj.geometry) obj.geometry.dispose();
  if (obj.material) {
    if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
    else obj.material.dispose();
  }
}

function makeBond(a, b, color, radius = 0.11) {
  const start = new THREE.Vector3(a.x, a.y, a.z);
  const end = new THREE.Vector3(b.x, b.y, b.z);
  const dir = new THREE.Vector3().subVectors(end, start);
  const length = dir.length();
  if (length < 1e-6) return new THREE.Group();
  const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);

  const geom = new THREE.CylinderGeometry(radius, radius, length, 10);
  const mat = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.45,
    metalness: 0.05,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.copy(mid);
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir.clone().normalize(),
  );
  return mesh;
}

function makeAtom(atom, radius, residueMeta) {
  const geom = new THREE.SphereGeometry(radius, 28, 28);
  const mat = new THREE.MeshStandardMaterial({
    color: atom.color,
    roughness: 0.35,
    metalness: 0.08,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.set(atom.x, atom.y, atom.z);
  mesh.userData = {
    pickable: true,
    element: atom.element,
    atomName: atom.atomName || atom.element,
    residue: residueMeta,
    baseEmissive: 0x000000,
  };
  return mesh;
}

function makeLabel(text, position, className) {
  const el = document.createElement("div");
  el.className = className;
  el.textContent = text;
  const label = new CSS2DObject(el);
  label.position.copy(position);
  return label;
}

function colorForCode(code) {
  return AA_BY_CODE[code]?.color || "#8a95a1";
}

function abbrForCode(code) {
  return AA_BY_CODE[code]?.abbr || code;
}

function nameForCode(code) {
  return AA_BY_CODE[code]?.name || code;
}

function fmtAngle(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return `${Math.round(Number(v))}°`;
}

/** Turn API pdbResult (sequence + phis/psis) into buildBackbone3D input. */
function peptideFromAngles(input) {
  const n = input.length || input.phis?.length || 0;
  if (!n || !input.phis?.length || !input.psis?.length) {
    throw new Error("No backbone angles available for 3D");
  }
  const seq = input.sequence || "";
  // Avoid materializing 25k abbr/color arrays — trace mode only needs codes lazily
  const caOnly = n > 400 || input.caTrace;
  if (caOnly) {
    return {
      length: n,
      codes: null,
      sequence: seq,
      abbrs: null,
      colors: null,
      phis: input.phis,
      psis: input.psis,
      omega: input.omega ?? 180,
      caOnly: true,
    };
  }
  const codes = input.codes
    ? [...input.codes]
    : seq
      ? [...seq]
      : [];
  return {
    length: n,
    codes: codes.length === n ? codes : codes.slice(0, n),
    abbrs: input.abbrs || codes.map((c) => abbrForCode(c)),
    colors: input.colors || codes.map((c) => colorForCode(c)),
    phis: input.phis,
    psis: input.psis,
    omega: input.omega ?? 180,
    caOnly: false,
  };
}

/** Normalize peptide angles OR API structure into renderable residues/atoms/bonds. */
function normalizeGeometry(input) {
  const n =
    input?.length ||
    input?.phis?.length ||
    input?.sequence?.length ||
    input?.structure?.residues?.length ||
    0;

  // Prefer API all-atom when it actually has sidechains
  const structAtoms = input?.structure?.atoms;
  if (
    Array.isArray(structAtoms) &&
    structAtoms.length > 0 &&
    !isBackboneOnly(input.structure, n || 1)
  ) {
    return normalizeApiStructure(input.structure, input);
  }
  if (
    Array.isArray(input?.atoms) &&
    input.atoms.length > 0 &&
    input?.residues &&
    !isBackboneOnly(input, n || 1)
  ) {
    return normalizeApiStructure(input, null);
  }

  // Client-side Stage-2/3 for short chains (works even if API stripped atoms)
  if (n > 0 && n <= 256 && input?.phis?.length && !input?.caTrace) {
    try {
      return buildAllAtom3D(input);
    } catch (err) {
      console.warn("all-atom build failed, falling back to backbone", err);
    }
  }

  return buildBackbone3D(peptideFromAngles(input));
}

function atomName(a) {
  return String(a.name || a.element || "").toUpperCase();
}

function isCaAtom(a) {
  return atomName(a) === "CA";
}

/** CPK-ish colors; sidechain carbons tinted by residue color. */
function colorForAtom(a, residueColor) {
  const nm = atomName(a);
  const el = String(a.element || "").toUpperCase();
  if (nm === "CA") return residueColor || colorForCode(a.code);
  if (nm === "N" || el === "N") return "#4b6bfb";
  if (nm === "O" || nm === "OG" || nm === "OG1" || nm === "OH" || el === "O")
    return "#e4572e";
  if (nm.startsWith("S") || el === "S") return "#e2c044";
  if (el === "H" || nm === "H") return "#f5f5f5";
  if (a.sidechain) return residueColor || "#9aa5b1";
  return "#8a95a1"; // backbone C
}

function radiusForAtom(a) {
  const nm = atomName(a);
  if (nm === "CA") return 0.48;
  if (nm === "N") return 0.28;
  if (nm === "C" || nm === "O") return 0.26;
  if (a.sidechain) {
    if (nm.startsWith("S")) return 0.32;
    if (nm.startsWith("N") || nm.startsWith("O")) return 0.24;
    return 0.22;
  }
  return 0.24;
}

function normalizeApiStructure(structure, meta) {
  const rawRes = structure.residues || [];
  const residues = rawRes.map((r, i) => ({
    index: i,
    code: r.code,
    abbr: abbrForCode(r.code),
    name: nameForCode(r.code),
    color: colorForCode(r.code),
    N: Array.isArray(r.N) ? { x: r.N[0], y: r.N[1], z: r.N[2] } : r.N,
    CA: Array.isArray(r.CA) ? { x: r.CA[0], y: r.CA[1], z: r.CA[2] } : r.CA,
    C: Array.isArray(r.C) ? { x: r.C[0], y: r.C[1], z: r.C[2] } : r.C,
    phi: r.phi ?? meta?.phis?.[i] ?? 0,
    psi: r.psi ?? meta?.psis?.[i] ?? 0,
  }));

  // If API omitted residue frames, reconstruct CA from atoms
  if (!residues.length && structure.atoms?.length) {
    const byRes = new Map();
    for (const a of structure.atoms) {
      const i = a.residue ?? 0;
      if (!byRes.has(i)) byRes.set(i, { index: i, code: a.code || "X" });
      const row = byRes.get(i);
      const nm = atomName(a);
      if (nm === "N" || nm === "CA" || nm === "C") {
        row[nm] = { x: a.x, y: a.y, z: a.z };
      }
      if (a.code) row.code = a.code;
    }
    const maxI = Math.max(...byRes.keys());
    for (let i = 0; i <= maxI; i++) {
      const r = byRes.get(i) || { index: i, code: "X" };
      residues.push({
        index: i,
        code: r.code,
        abbr: abbrForCode(r.code),
        name: nameForCode(r.code),
        color: colorForCode(r.code),
        N: r.N,
        CA: r.CA,
        C: r.C,
        phi: meta?.phis?.[i] ?? 0,
        psi: meta?.psis?.[i] ?? 0,
      });
    }
  }

  const atoms = structure.atoms.map((a) => {
    const code = a.code || residues[a.residue]?.code;
    const resColor = residues[a.residue]?.color || colorForCode(code);
    return {
      x: a.x,
      y: a.y,
      z: a.z,
      residue: a.residue,
      element: isCaAtom(a) ? "CA" : a.element || atomName(a)[0] || "C",
      atomName: atomName(a),
      sidechain: Boolean(a.sidechain),
      color: colorForAtom({ ...a, code }, resColor),
      code,
      abbr: abbrForCode(code),
      name: nameForCode(code),
    };
  });

  return {
    residues,
    atoms,
    bonds: structure.bonds || [],
    allAtom: true,
  };
}

function buildSceneContent(root, input) {
  const geom = normalizeGeometry(input);
  const { residues, atoms, bonds } = geom;
  for (const r of residues) {
    if (!r.name) r.name = nameForCode(r.code);
  }
  const length = residues.length;
  const pickables = [];

  // Long chains: Cα trace only (full N/CA/C meshes would freeze the browser)
  if (length > 400) {
    return buildCaTraceScene(root, residues, pickables);
  }

  // Thinner bonds for dense all-atom models
  const allAtom = Boolean(geom.allAtom) || atoms.length > length * 3 + 2;
  const bondRadius = allAtom ? 0.07 : 0.11;

  for (const [i, j] of bonds) {
    const a = atoms[i];
    const b = atoms[j];
    if (!a || !b) continue;
    const color =
      a.element === "CA"
        ? a.color
        : b.element === "CA"
          ? b.color
          : a.sidechain || b.sidechain
            ? a.color || b.color
            : "#8a95a1";
    root.add(makeBond(a, b, color, bondRadius));
  }

  for (const atom of atoms) {
    const res = residues[atom.residue ?? 0];
    const meta = {
      index: (atom.residue ?? 0) + 1,
      code: res?.code || atom.code,
      abbr: res?.abbr || atom.abbr || abbrForCode(atom.code),
      name: res?.name || atom.name || nameForCode(atom.code),
      phi: res?.phi,
      psi: res?.psi,
      color: res?.color || atom.color,
      atomName: atom.atomName,
    };
    const mesh = makeAtom(atom, radiusForAtom(atom), meta);
    root.add(mesh);
    pickables.push(mesh);
  }

  return { residues, atoms, length, pickables, allAtom };
}

function buildCaTraceScene(root, residues, pickables) {
  const n = residues.length;
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const ca = residues[i].CA;
    positions[i * 3] = ca.x;
    positions[i * 3 + 1] = ca.y;
    positions[i * 3 + 2] = ca.z;
  }

  const lineGeom = new THREE.BufferGeometry();
  lineGeom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const lineMat = new THREE.LineBasicMaterial({
    color: 0xc4894a,
    transparent: true,
    opacity: 0.9,
  });
  root.add(new THREE.Line(lineGeom, lineMat));

  // Instanced Cα markers (sparse for very long chains)
  const stride = n > 20000 ? 10 : n > 8000 ? 5 : n > 2000 ? 2 : 1;
  const count = Math.ceil(n / stride);
  const sphereGeom = new THREE.SphereGeometry(n > 5000 ? 0.55 : 0.4, 8, 8);
  const sphereMat = new THREE.MeshStandardMaterial({
    roughness: 0.4,
    metalness: 0.05,
  });
  const mesh = new THREE.InstancedMesh(sphereGeom, sphereMat, count);
  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  const instanceMeta = [];

  let k = 0;
  for (let i = 0; i < n; i += stride) {
    const r = residues[i];
    dummy.position.set(r.CA.x, r.CA.y, r.CA.z);
    dummy.updateMatrix();
    mesh.setMatrixAt(k, dummy.matrix);
    color.set(r.color || "#c4894a");
    mesh.setColorAt(k, color);
    instanceMeta[k] = {
      index: i + 1,
      code: r.code,
      abbr: r.abbr,
      name: r.name,
      phi: r.phi,
      psi: r.psi,
      color: r.color,
    };
    k += 1;
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.userData.instanceMeta = instanceMeta;
  mesh.userData.isCaInstances = true;
  root.add(mesh);
  pickables.push(mesh);

  return { residues, length: n, pickables };
}

function peptideBounds(residues, atoms) {
  const box = new THREE.Box3();
  if (atoms?.length) {
    for (const a of atoms) {
      box.expandByPoint(new THREE.Vector3(a.x, a.y, a.z));
    }
    if (!box.isEmpty()) return box;
  }
  for (const r of residues) {
    // Long-chain Cα-trace mode omits N/C — only CA is guaranteed
    if (r.N) box.expandByPoint(new THREE.Vector3(r.N.x, r.N.y, r.N.z));
    if (r.CA) box.expandByPoint(new THREE.Vector3(r.CA.x, r.CA.y, r.CA.z));
    if (r.C) box.expandByPoint(new THREE.Vector3(r.C.x, r.C.y, r.C.z));
  }
  return box;
}

function fitCamera(camera, controls, residues, atoms) {
  const box = peptideBounds(residues, atoms);
  const size = Math.max(box.getSize(new THREE.Vector3()).length(), 3);
  const center = box.getCenter(new THREE.Vector3());
  const n = residues.length;
  // Pull camera farther back for longer chains so the whole fold is visible
  const distScale =
    n >= 5000 ? 4.8 : n >= 1000 ? 4.0 : n >= 200 ? 3.4 : n >= 80 ? 2.7 : n >= 4 ? 2.05 : 1.85;
  const dist = size * distScale;

  camera.position.set(
    center.x + dist * 0.65,
    center.y + dist * 0.42,
    center.z + dist * 0.85,
  );
  controls.target.copy(center);
  controls.update();
}

function tooltipHTML(meta, element) {
  const atomLabel = meta.atomName || element;
  const atom = atomLabel ? ` · ${atomLabel}` : "";
  const name = meta.abbr || meta.name || meta.code || "";
  const phi = fmtAngle(meta.phi);
  const psi = fmtAngle(meta.psi);
  const variantRow = meta.variantLabel
    ? `<div class="viewer-tip__row viewer-tip__variant">Variant ${meta.variantLabel}${
        meta.variantFrom || meta.variantTo
          ? ` · WT ${meta.variantFrom || "?"} → ${meta.variantTo || "Δ"}`
          : ""
      }</div>`
    : "";
  return `
    <div class="viewer-tip__title" style="--c:${meta.color}">
      <span class="viewer-tip__swatch"></span>
      ${meta.index}${name ? ` · ${name}` : ""}
    </div>
    <div class="viewer-tip__row">${meta.code || ""}${atom}</div>
    ${variantRow}
    <div class="viewer-tip__row viewer-tip__angles">φ ${phi} · ψ ${psi}</div>
  `;
}

/** Mark UniProt / WT-vs-mutant sites as glowing Cα points + labels. */
function addVariantMarkers(root, residues, pickables, sites) {
  if (!sites?.length || !residues?.length) return 0;
  let placed = 0;
  for (const site of sites) {
    const idx = Number(site.index) - 1;
    if (!Number.isFinite(idx) || idx < 0 || idx >= residues.length) continue;
    const ca = residues[idx]?.CA;
    if (!ca) continue;

    const pos = new THREE.Vector3(ca.x, ca.y, ca.z);
    const glow = new THREE.Mesh(
      new THREE.SphereGeometry(1.25, 22, 22),
      new THREE.MeshBasicMaterial({
        color: 0xe07070,
        transparent: true,
        opacity: 0.28,
        depthWrite: false,
      }),
    );
    glow.position.copy(pos);
    root.add(glow);

    const core = new THREE.Mesh(
      new THREE.SphereGeometry(0.62, 20, 20),
      new THREE.MeshStandardMaterial({
        color: 0xe07070,
        emissive: 0xb91c1c,
        emissiveIntensity: 0.85,
        roughness: 0.35,
        metalness: 0.1,
      }),
    );
    core.position.copy(pos);
    const res = residues[idx];
    const labelText = site.label || `${site.from || ""}${site.index}${site.to || "Δ"}`;
    core.userData = {
      pickable: true,
      element: "CA",
      atomName: "CA",
      isVariantMark: true,
      residue: {
        index: site.index,
        code: res.code || site.to || "",
        abbr: res.abbr || abbrForCode(site.to || res.code),
        name: res.name || nameForCode(site.to || res.code),
        phi: res.phi,
        psi: res.psi,
        color: "#e07070",
        variantLabel: labelText,
        variantFrom: site.from || "",
        variantTo: site.to ?? "",
      },
    };
    root.add(core);
    pickables.push(core);

    const label = makeLabel(
      labelText,
      new THREE.Vector3(ca.x, ca.y + 1.55, ca.z),
      "label3d label3d--variant",
    );
    root.add(label);
    placed += 1;
  }
  return placed;
}

export function destroyPeptide3D(container) {
  if (container) {
    const active = actives.get(container);
    if (!active) return;
    cancelAnimationFrame(active.raf);
    window.removeEventListener("resize", active.onResize);
    if (active.onPointerMove) {
      active.pointerTarget.removeEventListener("pointermove", active.onPointerMove);
      active.pointerTarget.removeEventListener("pointerleave", active.onPointerLeave);
    }
    active.controls.dispose();
    active.renderer.dispose();
    active.container.replaceChildren();
    actives.delete(container);
    return;
  }
  for (const el of [...actives.keys()]) destroyPeptide3D(el);
}

export function mountPeptide3D(container, input) {
  if (!container) return;
  destroyPeptide3D(container);

  const width = container.clientWidth || 360;
  const height = Math.max(container.clientHeight || 0, 340);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x241e18);

  const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 5000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  container.appendChild(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(width, height);
  labelRenderer.domElement.className = "label3d-layer";
  container.appendChild(labelRenderer.domElement);

  const tip = document.createElement("div");
  tip.className = "viewer-tip";
  tip.hidden = true;
  container.appendChild(tip);

  const controls = new OrbitControls(camera, labelRenderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 2;

  const root = new THREE.Group();
  scene.add(root);
  const { residues, atoms, length, pickables, allAtom } = buildSceneContent(
    root,
    input,
  );
  if (!residues?.length) {
    container.innerHTML = `<p class="empty">No residues to display.</p>`;
    return;
  }
  const nVariant = addVariantMarkers(
    root,
    residues,
    pickables,
    input.variantSites || [],
  );
  const spanGuess = Math.max(length * 1.2, 10);
  controls.maxDistance = Math.max(500, spanGuess * 15);

  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(5, 8, 6);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.35);
  fill.position.set(-6, -2, -4);
  scene.add(fill);

  const box = peptideBounds(residues, atoms);
  const span = Math.max(box.getSize(new THREE.Vector3()).length(), 4);
  const gridSize = Math.ceil(span * 1.6 + 4);
  const grid = new THREE.GridHelper(gridSize, 14, 0x4a3c30, 0x322820);
  grid.position.y = box.min.y - 1.2;
  scene.add(grid);

  fitCamera(camera, controls, residues, atoms);

  // Badge so it's obvious all-atom / sidechains are shown
  if (allAtom && atoms?.length) {
    const badge = document.createElement("div");
    badge.className = "viewer-badge";
    badge.textContent = `All-atom · ${atoms.length} atoms`;
    container.appendChild(badge);
  }
  if (nVariant > 0) {
    const vBadge = document.createElement("div");
    vBadge.className = "viewer-badge viewer-badge--variant";
    const first = input.variantSites?.[0]?.label;
    vBadge.textContent =
      nVariant === 1 && first
        ? `Variant · ${first}`
        : `${nVariant} variant site${nVariant === 1 ? "" : "s"}`;
    container.appendChild(vBadge);
  }
  // Keep far plane beyond camera distance so long chains aren't clipped away
  const camDist = camera.position.distanceTo(controls.target);
  camera.far = Math.max(5000, camDist * 20 + span * 10);
  camera.near = Math.max(0.05, Math.min(1, camDist / 500));
  camera.updateProjectionMatrix();
  controls.maxDistance = Math.max(controls.maxDistance, camDist * 8, span * 20);

  const raycaster = new THREE.Raycaster();
  // Prefer Cα hits: slightly larger threshold helps thin atoms
  raycaster.params.Mesh = { threshold: 0 };
  const pointer = new THREE.Vector2();
  let hovered = null;

  function clearHover() {
    if (hovered?.type === "instance") {
      hovered = null;
    } else if (hovered?.material?.emissive) {
      hovered.material.emissive.setHex(0x000000);
      hovered.material.emissiveIntensity = 0;
      hovered = null;
    } else {
      hovered = null;
    }
    tip.hidden = true;
    labelRenderer.domElement.style.cursor = "";
  }

  function setHover(mesh, clientX, clientY, instanceId = null) {
    const key = instanceId != null ? `i:${instanceId}` : mesh.uuid || mesh;
    if (hovered?.key === key || (instanceId == null && hovered === mesh)) {
      positionTip(clientX, clientY);
      return;
    }
    clearHover();

    let meta;
    let element = mesh.userData.element;
    if (mesh.userData?.isCaInstances && instanceId != null) {
      meta = mesh.userData.instanceMeta?.[instanceId];
      element = "CA";
      hovered = { type: "instance", key, mesh, instanceId };
    } else {
      hovered = mesh;
      if (mesh.material?.emissive) {
        mesh.material.emissive.setHex(0x224466);
        mesh.material.emissiveIntensity = 0.55;
      }
      meta = mesh.userData.residue;
    }
    if (!meta) {
      clearHover();
      return;
    }
    tip.innerHTML = tooltipHTML(
      { ...meta, atomName: mesh.userData?.atomName || meta.atomName },
      element,
    );
    tip.hidden = false;
    labelRenderer.domElement.style.cursor = "pointer";
    positionTip(clientX, clientY);
  }

  function positionTip(clientX, clientY) {
    const rect = container.getBoundingClientRect();
    let x = clientX - rect.left + 12;
    let y = clientY - rect.top + 12;
    const tw = tip.offsetWidth || 160;
    const th = tip.offsetHeight || 48;
    if (x + tw > rect.width - 6) x = rect.width - tw - 6;
    if (y + th > rect.height - 6) y = clientY - rect.top - th - 10;
    if (x < 6) x = 6;
    if (y < 6) y = 6;
    tip.style.transform = `translate(${x}px, ${y}px)`;
  }

  function onPointerMove(event) {
    const rect = labelRenderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(pickables, false);
    if (!hits.length) {
      clearHover();
      return;
    }
    const hit =
      hits.find((h) => h.object.userData.isVariantMark) ||
      hits.find((h) => h.object.userData.element === "CA") ||
      hits.find((h) => h.object.userData.isCaInstances) ||
      hits[0];
    const id = hit.instanceId != null ? hit.instanceId : null;
    setHover(hit.object, event.clientX, event.clientY, id);
  }

  function onPointerLeave() {
    clearHover();
  }

  labelRenderer.domElement.addEventListener("pointermove", onPointerMove);
  labelRenderer.domElement.addEventListener("pointerleave", onPointerLeave);

  const onResize = () => {
    if (!actives.has(container)) return;
    const w = container.clientWidth || 360;
    const h = Math.max(container.clientHeight || 0, 340);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    labelRenderer.setSize(w, h);
  };

  window.addEventListener("resize", onResize);

  const state = {
    container,
    renderer,
    labelRenderer,
    controls,
    raf: 0,
    onResize,
    onPointerMove,
    onPointerLeave,
    pointerTarget: labelRenderer.domElement,
  };
  actives.set(container, state);

  const tick = () => {
    const cur = actives.get(container);
    if (!cur) return;
    cur.raf = requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  };
  tick();
}
