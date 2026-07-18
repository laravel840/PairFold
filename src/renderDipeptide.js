/**
 * SVG 2D sketches of peptide backbone geometry.
 * Angles are drawn macroscopically as arcs + degree labels on the diagram.
 */

function polar(cx, cy, r, deg) {
  const rad = (deg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function formatAngle(v) {
  const sign = v > 0 ? "+" : "";
  return `${sign}${Math.round(v)}°`;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function angleOf(from, to) {
  return (Math.atan2(to.y - from.y, to.x - from.x) * 180) / Math.PI;
}

function normalizeDeg(deg) {
  let a = deg % 360;
  if (a < 0) a += 360;
  return a;
}

function shortestArc(fromDeg, toDeg) {
  let delta = normalizeDeg(toDeg) - normalizeDeg(fromDeg);
  if (delta > 180) delta -= 360;
  if (delta < -180) delta += 360;
  return delta;
}

/**
 * Place residues so the chain kink visibly matches bend magnitude.
 */
function placeChain(count, bends, size) {
  const pad = size * 0.16;
  const r = size * (count === 3 ? 0.095 : 0.12);
  const step = (size - pad * 2 - r * 2) / Math.max(1, count - 1);

  const points = [];
  let x = pad + r;
  let y = size * (count === 3 ? 0.58 : 0.55);
  // Start slightly upward so large bends stay readable
  let heading = count === 3 ? -18 : -12;
  points.push({ x, y, heading });

  for (let i = 0; i < count - 1; i++) {
    // Macroscopic turn: map backbone bend into a clear on-canvas kink
    const turn = clamp((bends[i] || 0) * 0.42, -70, 70);
    heading += turn;
    const next = polar(x, y, step, heading);
    next.x = clamp(next.x, pad + r, size - pad - r);
    next.y = clamp(next.y, pad + r + 28, size - pad - r - 18);
    next.heading = heading;
    points.push(next);
    x = next.x;
    y = next.y;
  }

  return { points, r };
}

function backbonePath(points) {
  if (points.length < 2) return "";
  let d = `M ${points[0].x} ${points[0].y}`;
  for (let i = 1; i < points.length; i++) {
    d += ` L ${points[i].x} ${points[i].y}`;
  }
  return d;
}

function gradientStops(colors) {
  return colors
    .map((c, i) => {
      const pct =
        colors.length === 1
          ? 0
          : Math.round((i / (colors.length - 1)) * 100);
      return `<stop offset="${pct}%" stop-color="${c}" />`;
    })
    .join("");
}

/**
 * Draw a large angle wedge + arc + label at a joint.
 */
function renderAngleMark(joint, prev, next, label, value, arcRadius, color) {
  const a0 = angleOf(joint, prev);
  const a1 = angleOf(joint, next);
  const sweep = shortestArc(a0, a1);
  const end = a0 + sweep;
  const large = Math.abs(sweep) > 180 ? 1 : 0;
  const sweepFlag = sweep >= 0 ? 1 : 0;

  const pStart = polar(joint.x, joint.y, arcRadius, a0);
  const pEnd = polar(joint.x, joint.y, arcRadius, end);
  const mid = polar(joint.x, joint.y, arcRadius + 16, a0 + sweep / 2);

  // Extension rays past the residues so the angle reads clearly
  const rayLen = arcRadius + 8;
  const rayA = polar(joint.x, joint.y, rayLen, a0);
  const rayB = polar(joint.x, joint.y, rayLen, end);

  const arcPath = `M ${pStart.x} ${pStart.y} A ${arcRadius} ${arcRadius} 0 ${large} ${sweepFlag} ${pEnd.x} ${pEnd.y}`;
  const wedgePath = `M ${joint.x} ${joint.y} L ${pStart.x} ${pStart.y} A ${arcRadius} ${arcRadius} 0 ${large} ${sweepFlag} ${pEnd.x} ${pEnd.y} Z`;

  return `
    <g class="angle-mark">
      <path class="angle-wedge" d="${wedgePath}" fill="${color}" />
      <line class="angle-ray" x1="${joint.x}" y1="${joint.y}" x2="${rayA.x}" y2="${rayA.y}" stroke="${color}" />
      <line class="angle-ray" x1="${joint.x}" y1="${joint.y}" x2="${rayB.x}" y2="${rayB.y}" stroke="${color}" />
      <path class="angle-arc" d="${arcPath}" stroke="${color}" />
      <text class="angle-label" x="${mid.x}" y="${mid.y}">
        <tspan class="angle-label__name" x="${mid.x}" dy="-0.35em">${label}</tspan>
        <tspan class="angle-label__value" x="${mid.x}" dy="1.15em">${formatAngle(value)}</tspan>
      </text>
    </g>
  `;
}

/**
 * Macroscopic angle overlays for each peptide bond.
 * Uses the geometric turn on canvas and annotates with ψᵢ / φᵢ₊₁.
 */
function renderMacroAngles(peptide, points, r) {
  const marks = [];
  const colors = ["#1f6f8b", "#b45309"];

  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];

    // Joint sits at the first residue of each bond; for middle bonds use the shared residue
    const joint = a;
    // Incoming reference: previous segment, or a leftward baseline for the first bond
    const prevRef =
      i === 0
        ? { x: joint.x - 40, y: joint.y }
        : points[i - 1];
    const nextRef = b;

    const psi = peptide.psis[i];
    const phi = peptide.phis[i + 1];
    const color = colors[i % colors.length];
    const arcR = r + 34 + i * 6;

    // Primary macroscopic mark: bend at this bond (ψ of donor)
    marks.push(
      renderAngleMark(
        joint,
        prevRef,
        nextRef,
        `ψ${i + 1}`,
        psi,
        arcR,
        color,
      ),
    );

    // Secondary mark near the acceptor residue for φ
    const mid = {
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
    };
    const along = angleOf(a, b);
    const side = polar(mid.x, mid.y, 22, along + 90);
    marks.push(`
      <g class="angle-mark angle-mark--phi">
        <line
          class="angle-tick"
          x1="${mid.x}"
          y1="${mid.y}"
          x2="${side.x}"
          y2="${side.y}"
          stroke="${color}"
        />
        <text class="angle-label angle-label--phi" x="${side.x}" y="${side.y}">
          <tspan class="angle-label__name" x="${side.x}" dy="-0.2em">φ${i + 2}</tspan>
          <tspan class="angle-label__value" x="${side.x}" dy="1.1em">${formatAngle(phi)}</tspan>
        </text>
      </g>
    `);
  }

  return marks.join("");
}

export function renderPeptideSVG(peptide, options = {}) {
  const {
    size = 320,
    showLabels = true,
    showAngles = true,
  } = options;

  const codes = peptide.codes;
  const colors = peptide.colors;
  const abbrs = peptide.abbrs;
  const bends = peptide.bends || [peptide.bend];
  const id = codes.join("");
  const { points, r } = placeChain(codes.length, bends, size);
  const path = backbonePath(points);

  const labels = points
    .map((p, i) => {
      const letterClass = showLabels ? "aa-letter" : "aa-letter aa-letter--sm";
      const abbr = showLabels
        ? `<text class="aa-abbr" x="${p.x}" y="${p.y + r + 14}">${abbrs[i]}</text>`
        : "";
      return `
        <text class="${letterClass}" x="${p.x}" y="${p.y + 1}" fill="#0b1215">${codes[i]}</text>
        ${abbr}
      `;
    })
    .join("");

  const macroAngles = showAngles
    ? renderMacroAngles(peptide, points, r)
    : "";

  const motif = showAngles
    ? `<text class="angle-hud motif" x="${size / 2}" y="${size - 8}">${peptide.motif}</text>`
    : "";

  const circles = points
    .map(
      (p, i) => `
      <circle
        class="residue"
        cx="${p.x}"
        cy="${p.y}"
        r="${r}"
        fill="${colors[i]}"
      />
    `,
    )
    .join("");

  const kind = peptide.length === 3 ? "Tripeptide" : "Dipeptide";

  return `
    <svg
      class="dipeptide-svg"
      viewBox="0 0 ${size} ${size}"
      width="${size}"
      height="${size}"
      role="img"
      aria-label="${kind} ${abbrs.join("-")}"
      data-pair="${id}"
    >
      <defs>
        <linearGradient id="bond-${id}" x1="0%" y1="0%" x2="100%" y2="0%">
          ${gradientStops(colors)}
        </linearGradient>
      </defs>

      ${macroAngles}

      <path
        class="backbone"
        d="${path}"
        stroke="url(#bond-${id})"
        fill="none"
      />

      ${circles}
      ${labels}
      ${motif}
    </svg>
  `;
}

/** @deprecated use renderPeptideSVG */
export function renderDipeptideSVG(angles, options = {}) {
  return renderPeptideSVG(angles, options);
}
