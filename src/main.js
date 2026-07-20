import "./style.css";
import { AMINO_ACIDS, AA_BY_CODE } from "./data/aminoAcids.js";
import {
  getPeptideAngles,
  parseSequenceQuery,
  suggestPeptides,
  PEPTIDE,
} from "./data/angles.js";
import {
  applySequenceEdit,
  fetchUniProtEntry,
  filterVariants,
  searchUniProt,
} from "./uniprot.js";

let selected = getPeptideAngles(["A", "G", "P", "V"]);
let pdbResult = null;
let query = "";
let predictStatus = "";
let predicting = false;
let segFilter = "";
let segLimit = 40;
let predictAbort = null;
let stallTimer = 0;
let lastProgressAt = 0;

/** UniProtKB browser state */
let uniprotQuery = "";
let uniprotStatus = "";
let uniprotBusy = false;
let uniprotHits = [];
let uniprotEntry = null;
let variantTab = "natural"; // natural | mutant
let variantFilter = "";
let variantLimit = 40;
const VARIANT_PAGE = 40;
/** Sites differing from UniProt wild-type — marked in the 3D panel */
let variantSites = [];

const app = document.querySelector("#app");
/** Same-origin API (FastAPI UI or Vite proxy of /predict + /health). */
const API = "";
const MAX_PDB_LEN = 50000;
/** Full ball-and-stick only for short chains; above ~400 the viewer uses a Cα trace. */
const VIEW_3D_MAX = 50000;
const NAME_LIST_MAX = 64;
const SEG_PAGE_SIZE = 40;
const SEG_SEARCH_MIN_LEN = 100;
/** Abort predict if no progress/heartbeat for this long (ms). */
const STALL_ABORT_MS = 120000;

const KIND = {
  2: "dipeptide",
  3: "tripeptide",
  4: "tetrapeptide",
  5: "pentapeptide",
};

function sequenceLength(view) {
  if (!view) return 0;
  if (view.sequence) return view.sequence.length;
  if (view.codes) return view.codes.length;
  if (view.length) return view.length;
  if (view.phis) return view.phis.length;
  return 0;
}

function show3DFor(view) {
  return sequenceLength(view) > 0 && sequenceLength(view) <= VIEW_3D_MAX;
}

function stageHTML(view, ariaLabel, hint) {
  if (show3DFor(view)) {
    const n = sequenceLength(view);
    const inline = n <= 256;
    return `
      <div class="detail__stage detail__stage--launch${inline ? " detail__stage--inline3d" : ""}">
        ${
          inline
            ? `<div class="viewer3d" id="viewer3d-inline" role="img" aria-label="${ariaLabel}"></div>`
            : ""
        }
        <button type="button" class="btn-3d" id="btn-open-3d" aria-label="${ariaLabel}">
          ${inline ? "Open larger 3D window" : "Open 3D structure"}
        </button>
        <p class="viewer3d__hint">
          ${n} residues · ${hint}
          ${
            variantSites.length
              ? ` · <span class="viewer3d__variant-hint">${variantSites.length} variant site${
                  variantSites.length === 1 ? "" : "s"
                } marked</span>`
              : ""
          }
        </p>
      </div>
    `;
  }
  return `
    <div class="detail__stage detail__stage--no3d">
      <div class="viewer3d viewer3d--disabled" aria-hidden="true"></div>
    </div>
  `;
}

function cloneViewPayload(view) {
  if (!view) return null;
  // Compact payload for popup (huge atom lists blow past sessionStorage quota)
  try {
    const n = sequenceLength(view);
    const sequence = view.sequence || (view.codes ? view.codes.join("") : "");
    const payload = {
      sequence,
      phis: view.phis,
      psis: view.psis,
      omega: view.omega ?? 180,
      length: n || view.phis?.length || 0,
      mode: view.mode,
      tertiary: n <= 1000 ? view.tertiary || null : null,
      caTrace: n > 400,
      variantSites: variantSites.length ? variantSites.map((s) => ({ ...s })) : [],
    };
    if (n <= 400) {
      payload.codes = view.codes
        ? [...view.codes]
        : sequence
          ? [...sequence]
          : [];
      payload.abbrs = view.abbrs || undefined;
      payload.colors = view.colors || undefined;
    }
    // Pass Stage-2/3 all-atom structure for short chains (viewer prefers this)
    const st = view.structure;
    const nAtoms = st?.atoms?.length || 0;
    if (n <= 256 && nAtoms > 0 && !st.skipped_3d) {
      payload.structure = {
        atoms: st.atoms,
        bonds: st.bonds || [],
        residues: st.residues || [],
        enabled: st.enabled,
        stage2: st.stage2 || null,
        n_atoms: st.n_atoms || nAtoms,
        note: st.note || "",
      };
      payload.allAtom = true;
    }
    if (!payload.phis?.length || !payload.psis?.length) return null;
    return payload;
  } catch {
    return null;
  }
}

window.__pairfoldGet3DPayload = function pairfoldGet3DPayload() {
  return cloneViewPayload(activeView());
};

function open3DWindow() {
  const view = activeView();
  if (!show3DFor(view)) return;
  const payload = cloneViewPayload(view);
  if (!payload) return;
  try {
    sessionStorage.setItem("pairfold_3d_payload", JSON.stringify(payload));
  } catch {
    /* quota — opener bridge still works */
  }
  const w = 960;
  const h = 720;
  const left = Math.max(0, Math.round((window.screen.width - w) / 2));
  const top = Math.max(0, Math.round((window.screen.height - h) / 2));
  const popup = window.open(
    new URL("viewer.html", window.location.href).href,
    "pairfold3d",
    `popup=yes,width=${w},height=${h},left=${left},top=${top},resizable=yes,scrollbars=no`,
  );
  if (!popup) {
    predictStatus = "Pop-up blocked — allow pop-ups for PairFold to open 3D.";
    const status = document.getElementById("predict-status");
    if (status) status.textContent = predictStatus;
  }
}

function bind3DButton() {
  document.getElementById("btn-open-3d")?.addEventListener("click", () => {
    open3DWindow();
  });
}

function legendHTML() {
  return AMINO_ACIDS.map(
    (aa) => `
      <li class="legend-item" style="--c:${aa.color}" title="${aa.name}">
        <span class="legend-swatch"></span>
        <span class="legend-code">${aa.code}</span>
        <span class="legend-name">${aa.abbr}</span>
      </li>
    `,
  ).join("");
}

function searchMatches(raw) {
  return suggestPeptides(raw);
}

function activeView() {
  return pdbResult || selected;
}

function metricsHTML(a) {
  const phis = a.phis;
  const psis = a.psis;
  const n = phis.length;
  const rows = [];
  for (let i = 0; i < n; i++) {
    rows.push(`
      <div>
        <dt>φ${i + 1}</dt>
        <dd>${Math.round(phis[i])}°</dd>
      </div>
      <div>
        <dt>ψ${i + 1}</dt>
        <dd>${Math.round(psis[i])}°</dd>
      </div>
    `);
  }
  if (a.omega != null) {
    rows.push(`
      <div><dt>ω</dt><dd>${a.omega}°</dd></div>
      <div><dt>Cα–Cα</dt><dd>${(a.caCa ?? 3.8).toFixed(1)} Å</dd></div>
    `);
  }
  return rows.join("");
}

function titleFromCodes(codes, colors) {
  return codes
    .map((c, i) => {
      const abbr = AA_BY_CODE[c]?.abbr || c;
      const color = colors?.[i] || AA_BY_CODE[c]?.color || "#333";
      return `<span style="color:${color}">${abbr}</span>`;
    })
    .join('<span class="detail__sep">–</span>');
}

function formatSegAngles(start, end, phis, psis) {
  if (!phis?.length || !psis?.length) return "";
  const parts = [];
  for (let i = start; i < end && i < phis.length; i++) {
    parts.push(
      `φ${i + 1} ${Math.round(phis[i])}° ψ${i + 1} ${Math.round(psis[i])}°`,
    );
  }
  return parts.join(" · ");
}

function meanConfidence(segs) {
  if (!segs?.length) return 0;
  return segs.reduce((a, s) => a + (s.confidence || 0), 0) / segs.length;
}

function filterSegments(segs, raw) {
  const q = String(raw || "").trim();
  if (!q || !segs?.length) return segs || [];

  const confMatch = q.match(/^(?:conf\s*)?(>=|>|<=|<)?\s*(\d{1,3})\s*%?$/i);
  if (confMatch && !/[A-Za-z]{2,}/.test(q.replace(/conf/i, ""))) {
    const op = confMatch[1] || ">=";
    const thr = Number(confMatch[2]) / 100;
    return segs.filter((s) => {
      const c = s.confidence || 0;
      if (op === ">") return c > thr;
      if (op === ">=") return c >= thr;
      if (op === "<") return c < thr;
      if (op === "<=") return c <= thr;
      return c >= thr;
    });
  }

  const rangeMatch = q.match(/^(\d+)\s*[-–:]\s*(\d+)$/);
  if (rangeMatch) {
    const a = Number(rangeMatch[1]);
    const b = Number(rangeMatch[2]);
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    return segs.filter((s) => s.start + 1 <= hi && s.end >= lo);
  }

  const resMatch = q.match(/^(\d+)$/);
  if (resMatch) {
    const r = Number(resMatch[1]);
    return segs.filter((s) => s.start + 1 <= r && s.end >= r);
  }

  const motif = q.toUpperCase().replace(/[^A-Z]/g, "");
  if (!motif) return segs;
  return segs.filter((s) => String(s.seq || "").toUpperCase().includes(motif));
}

function segItemHTML(s, phis, psis) {
  const angles = formatSegAngles(s.start, s.end, phis, psis);
  return `
    <li>
      ${
        angles
          ? `<span class="seg-list__angles">${escapeHtml(angles)}</span>`
          : ""
      }
      <span class="seg-list__meta">
        <code>${s.start + 1}–${s.end}</code>
        <strong>${escapeHtml(s.seq || "")}</strong>
        <span>conf ${((s.confidence || 0) * 100).toFixed(0)}%</span>
      </span>
    </li>`;
}

function segmentationHTML(segs, phis, psis, seqLen) {
  if (!segs?.length) return "";
  const useSearch =
    seqLen > SEG_SEARCH_MIN_LEN || segs.length > SEG_PAGE_SIZE;
  const filtered = filterSegments(segs, segFilter);
  const shown = filtered.slice(0, segLimit);
  const remaining = Math.max(0, filtered.length - shown.length);
  const mean = meanConfidence(segs);

  return `
    <div class="seg-list" id="seg-panel">
      <div class="seg-list__head">
        <p class="seg-list__title">Segmentation</p>
        <p class="seg-list__summary">
          ${segs.length} segment(s) · mean conf ${(mean * 100).toFixed(0)}%
          ${seqLen > VIEW_3D_MAX ? ` · ${seqLen} aa` : ""}
        </p>
      </div>
      ${
        useSearch
          ? `
        <label class="seg-list__filter">
          <span class="seg-list__filter-label">Find segments</span>
          <input
            id="seg-filter"
            type="search"
            placeholder="e.g. 252 · 200-300 · VRE · conf>70"
            value="${escapeHtml(segFilter)}"
            autocomplete="off"
            spellcheck="false"
          />
        </label>`
          : ""
      }
      <p class="seg-list__count" id="seg-count">
        Showing ${shown.length} of ${filtered.length}
        ${segFilter.trim() ? " (filtered)" : ""}
      </p>
      <ul id="seg-list-items">
        ${
          shown.length
            ? shown.map((s) => segItemHTML(s, phis, psis)).join("")
            : `<li class="seg-list__empty">No segments match this filter.</li>`
        }
      </ul>
      ${
        remaining > 0
          ? `<button type="button" class="seg-list__more" id="seg-more">
              Show more (${Math.min(SEG_PAGE_SIZE, remaining)} of ${remaining} left)
            </button>`
          : ""
      }
    </div>
  `;
}

function bindSegPanel() {
  const filter = document.getElementById("seg-filter");
  filter?.addEventListener("input", (e) => {
    segFilter = e.target.value;
    segLimit = SEG_PAGE_SIZE;
    refreshSegPanel();
  });
  document.getElementById("seg-more")?.addEventListener("click", () => {
    segLimit += SEG_PAGE_SIZE;
    refreshSegPanel();
  });
}

function refreshSegPanel() {
  if (!pdbResult) return;
  const host = document.getElementById("seg-panel");
  if (!host) return;
  const keepFocus = document.activeElement?.id === "seg-filter";
  const caret = keepFocus ? document.getElementById("seg-filter")?.selectionStart : null;
  host.outerHTML = segmentationHTML(
    pdbResult.segmentation,
    pdbResult.phis,
    pdbResult.psis,
    pdbResult.sequence?.length || 0,
  );
  bindSegPanel();
  if (keepFocus) {
    const again = document.getElementById("seg-filter");
    if (again) {
      again.focus();
      if (caret != null) again.setSelectionRange(caret, caret);
    }
  }
}

function tertiaryHTML(t) {
  if (!t || !t.enabled) return "";
  const improved = t.improved ? "improved" : "baseline";
  return `
    <div class="tertiary-card">
      <p class="tertiary-card__title">Tertiary ranker · ${improved}</p>
      <dl class="tertiary-card__metrics">
        <div><dt>Score</dt><dd>${Number(t.score).toFixed(2)}</dd></div>
        <div><dt>Mode</dt><dd>${escapeHtml(t.mode || "refine")}</dd></div>
        <div><dt>Clash</dt><dd>${Number(t.clash_energy).toFixed(2)}</dd></div>
        <div><dt>Rg</dt><dd>${Number(t.rg).toFixed(1)} / ${Number(t.rg_expected).toFixed(1)} Å</dd></div>
        <div><dt>Hydro burial</dt><dd>${Number(t.hydrophobic_burial).toFixed(2)}</dd></div>
      </dl>
      <p class="tertiary-card__note">${escapeHtml(t.note || "")}</p>
    </div>
  `;
}

function detailHTML() {
  const pdb = pdbResult;
  if (pdb) {
    const codes = [...pdb.sequence];
    const n = codes.length;
    const hideNames = n > NAME_LIST_MAX;
    const title = hideNames ? `${n} residues` : titleFromCodes(codes);
    const namesLine = hideNames
      ? `${n} aa · ${escapeHtml(pdb.device || "")}`
      : `${escapeHtml(pdb.sequence)} · ${escapeHtml(pdb.device || "")}`;
    const metricsBlock = hideNames
      ? `<p class="detail__metrics-note">Per-residue angle table hidden for sequences &gt; ${NAME_LIST_MAX}. Use segment rows or the find box.</p>`
      : `
          <div class="metrics-scroll">
            <dl class="metrics metrics--flex">
              ${metricsHTML(pdb)}
            </dl>
          </div>`;
    const nAtoms = pdb.structure?.atoms?.length || 0;
    const allAtom = nAtoms > 0 && n <= 256 && !pdb.structure?.skipped_3d;
    const hint =
      n > 400
        ? "Cα trace in a new window (light mode for long chains)"
        : allAtom
          ? `all-atom (${nAtoms} atoms) · sidechains · opens in a new window`
          : "backbone 3D · opens in a new window";
    return `
      <section class="detail" aria-live="polite">
        ${stageHTML(pdb, "3D tertiary prediction", hint)}
        <div class="detail__info">
          <p class="detail__eyebrow">PDB-trained prediction · ${escapeHtml(pdb.mode || "")}</p>
          <h2 class="detail__title">${title}</h2>
          <p class="detail__names">${namesLine}</p>
          <p class="detail__motif">${escapeHtml(pdb.note || "PDB fragment model")}</p>
          ${tertiaryHTML(pdb.tertiary)}
          ${allAtom ? `<p class="detail__motif">Stage-2/3 all-atom: ${nAtoms} atoms${pdb.structure?.stage2?.n_sidechain_atoms != null ? ` · ${pdb.structure.stage2.n_sidechain_atoms} sidechain` : ""}.</p>` : ""}
          ${segmentationHTML(pdb.segmentation, pdb.phis, pdb.psis, n)}
          ${metricsBlock}
          <p class="detail__note">
            Local torsions from PDB fragments; 3D up to ${VIEW_3D_MAX} aa
            (&gt;400 aa = Cα trace only). All-atom sidechains for ≤256 aa.
            Tertiary ranker for ≤1000 aa. Not AlphaFold.
          </p>
        </div>
      </section>
    `;
  }

  const a = selected;
  const kind = KIND[a.length] || "peptide";
  const title = a.abbrs
    .map((abbr, i) => `<span style="color:${a.colors[i]}">${abbr}</span>`)
    .join('<span class="detail__sep">–</span>');

  return `
    <section class="detail" aria-live="polite">
      ${stageHTML(
        a,
        `3D ${kind} backbone`,
        "opens in a new window",
      )}
      <div class="detail__info">
        <p class="detail__eyebrow">Rule-based ${kind}</p>
        <h2 class="detail__title">${title}</h2>
        <p class="detail__names">${a.names.join(" → ")}</p>
        <p class="detail__motif">${a.motif}</p>
        <div class="metrics-scroll">
          <dl class="metrics metrics--flex">
            ${metricsHTML(a)}
          </dl>
        </div>
        <p class="detail__note">
          Quick local model from Ramachandran preferences.
          Use <strong>Predict (PDB)</strong> below for the trained fragment network.
        </p>
      </div>
    </section>
  `;
}

async function mountViewer() {
  bind3DButton();
  const host = document.getElementById("viewer3d-inline");
  if (!host) return;
  const view = activeView();
  if (!view?.phis?.length || sequenceLength(view) > 256) return;
  try {
    const { destroyPeptide3D, mountPeptide3D } = await import("./viewer3D.js");
    destroyPeptide3D(host);
    // Always rebuild all-atom client-side for the inline viewer
    const payload = {
      sequence: view.sequence || (view.codes ? view.codes.join("") : ""),
      phis: view.phis,
      psis: view.psis,
      omega: view.omega ?? 180,
      length: sequenceLength(view),
      // Force client all-atom path (ignore stripped API backbone-only structure)
      structure: null,
      caTrace: false,
      variantSites: variantSites.length ? variantSites.map((s) => ({ ...s })) : [],
    };
    mountPeptide3D(host, payload);
  } catch (err) {
    host.innerHTML = `<p class="empty">3D failed: ${String(err.message || err)}</p>`;
    console.error(err);
  }
}

function suggestionHTML(list) {
  if (!query.trim()) {
    return `<p class="search-hint">Short search: <kbd>AGPV</kbd>. PDB predict: up to <strong>${MAX_PDB_LEN}</strong> residues (3D Cα trace for long chains).</p>`;
  }
  const seq = cleanSeq(query);
  if (seq.length > PEPTIDE.maxLength) {
    return "";
  }
  if (!list.length) {
    if (pdbResult) return "";
    return `<p class="empty">No short-peptide match. For longer sequences use Predict (PDB).</p>`;
  }

  return `
    <ul class="search-results" role="listbox">
      ${list
        .map((p) => {
          const code = p.codes.join("");
          const active =
            !pdbResult && selected.codes?.join("") === code ? " is-active" : "";
          const chips = p.codes
            .map(
              (c, i) =>
                `<span class="pair-chip" style="--c:${p.colors[i]}">${c}</span>${
                  i < p.codes.length - 1
                    ? '<span class="pair-arrow">→</span>'
                    : ""
                }`,
            )
            .join("");
          return `
            <li>
              <button
                type="button"
                class="search-result${active}"
                role="option"
                data-pair="${code}"
              >
                <span class="search-result__codes">${chips}</span>
                <span class="search-result__names">${p.abbrs.join("–")}</span>
                <span class="search-result__motif">${p.motif}</span>
              </button>
            </li>
          `;
        })
        .join("")}
    </ul>
  `;
}

function setProgress(pct, message, etaS, elapsedS) {
  const wrap = document.getElementById("predict-progress");
  const fill = document.getElementById("predict-progress-fill");
  const meta = document.getElementById("predict-progress-meta");
  const btn = document.getElementById("btn-pdb");
  if (!wrap || !fill || !meta) return;
  wrap.hidden = false;
  const p = Math.max(0, Math.min(100, Number(pct) || 0));
  fill.style.width = `${p}%`;
  const eta =
    etaS == null || Number.isNaN(etaS)
      ? "…"
      : etaS <= 0.5
        ? "almost done"
        : `~${Math.ceil(etaS)}s left`;
  const elapsed =
    elapsedS == null || Number.isNaN(elapsedS) ? "" : ` · ${elapsedS.toFixed(0)}s elapsed`;
  meta.textContent = `${p.toFixed(0)}% · ${message || "Working"} · ${eta}${elapsed}`;
  if (btn) btn.disabled = predicting;
}

function hideProgress() {
  const wrap = document.getElementById("predict-progress");
  const fill = document.getElementById("predict-progress-fill");
  const btn = document.getElementById("btn-pdb");
  if (wrap) wrap.hidden = true;
  if (fill) fill.style.width = "0%";
  if (btn) btn.disabled = false;
  const cancel = document.getElementById("btn-cancel-predict");
  if (cancel) cancel.hidden = true;
}

function showSafetyAlert(message) {
  let el = document.getElementById("safety-alert");
  if (!el) {
    el = document.createElement("div");
    el.id = "safety-alert";
    el.className = "safety-alert";
    el.setAttribute("role", "alert");
    const host = document.querySelector(".explorer") || document.querySelector("#app");
    host?.prepend(el);
  }
  el.hidden = false;
  el.textContent = message;
}

function clearSafetyAlert() {
  const el = document.getElementById("safety-alert");
  if (el) {
    el.hidden = true;
    el.textContent = "";
  }
}

function clearStallWatch() {
  if (stallTimer) {
    window.clearInterval(stallTimer);
    stallTimer = 0;
  }
}

function abortPredict(reason) {
  clearStallWatch();
  try {
    predictAbort?.abort();
  } catch {
    /* ignore */
  }
  predictAbort = null;
  predicting = false;
  hideProgress();
  predictStatus = reason || "Prediction aborted.";
  const status = document.getElementById("predict-status");
  if (status) status.textContent = predictStatus;
  showSafetyAlert(predictStatus);
}

function armStallWatch() {
  clearStallWatch();
  lastProgressAt = Date.now();
  stallTimer = window.setInterval(() => {
    if (!predicting) {
      clearStallWatch();
      return;
    }
    if (Date.now() - lastProgressAt > STALL_ABORT_MS) {
      abortPredict(
        "Safety stop: no progress for 2 minutes. Prediction cancelled — try again or use a shorter sequence.",
      );
    }
  }, 2000);
}

async function predictPdb() {
  const seq = cleanSeq(query || selected.codes?.join("") || "");
  const status = document.getElementById("predict-status");
  if (predicting) return;
  if (seq.length < 2 || seq.length > MAX_PDB_LEN) {
    predictStatus = `Sequence must be 2–${MAX_PDB_LEN} standard amino acids.`;
    if (status) status.textContent = predictStatus;
    return;
  }
  if (!seqIsStandardAA(seq)) {
    predictStatus = "Only the 20 standard amino acids are supported.";
    if (status) status.textContent = predictStatus;
    return;
  }

  predicting = true;
  clearSafetyAlert();
  predictAbort = new AbortController();
  armStallWatch();
  predictStatus =
    seq.length > 2000
      ? `Running PDB fragment model on ${seq.length.toLocaleString()} aa (tiled long-chain path)…`
      : "Running PDB fragment model…";
  if (status) status.textContent = predictStatus;
  setProgress(1, `Starting (${seq.length} aa)`, null, 0);
  const cancelBtn = document.getElementById("btn-cancel-predict");
  if (cancelBtn) cancelBtn.hidden = false;

  try {
    const res = await fetch(`${API}/predict/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence: seq }),
      signal: predictAbort.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    if (!res.body) throw new Error("No response stream");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let result = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      lastProgressAt = Date.now();
      buffer += decoder.decode(value, { stream: true });
      // Keep only the incomplete trailing line — avoid holding multi-MB result twice
      const nl = buffer.lastIndexOf("\n");
      if (nl < 0) {
        // Huge unfinished result line → abort before browser thrash
        if (buffer.length > 8_000_000) {
          abortPredict(
            "Safety stop: result payload too large (>8 MB). Prediction cancelled.",
          );
          try {
            await reader.cancel();
          } catch {
            /* ignore */
          }
          return;
        }
        continue;
      }
      const complete = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
      for (const line of complete.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        let msg;
        try {
          msg = JSON.parse(trimmed);
        } catch {
          continue;
        }
        if (msg.type === "progress") {
          lastProgressAt = Date.now();
          setProgress(msg.pct, msg.message, msg.eta_s, msg.elapsed_s);
          predictStatus = msg.message || predictStatus;
          if (status) status.textContent = predictStatus;
        } else if (msg.type === "result") {
          result = msg.data;
          setProgress(100, "Done", 0, msg.data ? undefined : 0);
        } else if (msg.type === "error") {
          throw new Error(msg.detail || "Prediction failed");
        }
      }
    }

    if (!result) throw new Error("Stream ended without a result");
    // Keep Stage-2/3 all-atom for short chains; strip heavy payloads for long ones
    if (result.structure) {
      const n = result.sequence?.length || result.phis?.length || 0;
      const nAtoms = result.structure.atoms?.length || 0;
      if (result.structure.skipped_3d || n > 256 || nAtoms === 0) {
        result.structure = {
          skipped_3d: result.structure.skipped_3d || n > 256,
          reason:
            result.structure.reason ||
            (n > 256
              ? "All-atom 3D kept only for ≤256 aa; long chains use φ/ψ rebuild."
              : undefined),
          atoms: [],
          bonds: [],
          residues: [],
        };
      }
      // else: keep atoms/bonds/residues for all-atom viewer
    }
    pdbResult = result;
    segFilter = "";
    segLimit = SEG_PAGE_SIZE;
    predictStatus = `Done · ${pdbResult.mode} · ${pdbResult.segmentation?.length || 1} segment(s) · ${seq.length} aa`;
    clearSafetyAlert();
    refreshDetail();
    updateSuggestions();
  } catch (e) {
    if (e?.name === "AbortError") {
      if (!predictStatus.startsWith("Safety stop")) {
        predictStatus = "Prediction cancelled.";
      }
    } else {
      const msg = e?.message || String(e);
      predictStatus = msg.includes("Safety stop")
        ? msg
        : `API unavailable: ${msg}. Train/start pairfold.server first.`;
      if (msg.includes("Safety stop") || msg.includes("memory")) {
        showSafetyAlert(predictStatus);
      }
    }
  } finally {
    clearStallWatch();
    predictAbort = null;
    predicting = false;
    hideProgress();
    const s2 = document.getElementById("predict-status");
    if (s2) s2.textContent = predictStatus;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function trySelectFromQuery(raw) {
  const parsed = parseSequenceQuery(raw);
  if (Array.isArray(parsed) && parsed.length >= PEPTIDE.minLength) {
    selectPeptide(parsed);
    return true;
  }
  return false;
}

function selectPeptide(codes) {
  pdbResult = null;
  segFilter = "";
  segLimit = SEG_PAGE_SIZE;
  selected = getPeptideAngles(codes);
  refreshDetail();
  updateSuggestions();
}

function refreshDetail() {
  const detail = document.querySelector(".detail");
  if (detail) {
    detail.outerHTML = detailHTML();
    mountViewer();
    bindSegPanel();
  }
}

function updateSuggestions() {
  const host = document.getElementById("search-panel");
  if (!host) return;
  // Never run short-peptide suggestion logic on long PDB sequences (browser autofill can be huge).
  const seq = cleanSeq(query);
  if (seq.length > PEPTIDE.maxLength) {
    host.innerHTML = "";
    return;
  }
  host.innerHTML = suggestionHTML(searchMatches(query));
  bindResults();
}

function cleanSeq(raw) {
  return raw.toUpperCase().replace(/[^A-Z]/g, "");
}

function seqIsStandardAA(seq) {
  for (let i = 0; i < seq.length; i++) {
    if (!AA_BY_CODE[seq[i]]) return false;
  }
  return true;
}

/** Parse plain text or FASTA into a cleaned AA sequence. */
function parseSequenceFileText(text) {
  const lines = String(text || "").split(/\r?\n/);
  const seqLines = [];
  for (const line of lines) {
    const t = line.trim();
    if (!t || t.startsWith(">") || t.startsWith(";")) continue;
    seqLines.push(t);
  }
  return cleanSeq(seqLines.join(""));
}

/**
 * Build 1-based mutation markers from wild-type vs mutant sequences.
 * Prefer an explicit UniProt site hint when provided.
 */
function buildVariantSites(wtSeq, mutSeq, hint) {
  const wt = String(wtSeq || "");
  const mut = String(mutSeq || "");
  if (hint?.index != null && Number.isFinite(Number(hint.index))) {
    const index = Number(hint.index);
    return [
      {
        index,
        from: hint.from ?? wt[index - 1] ?? "",
        to: hint.to ?? mut[index - 1] ?? "",
        label:
          hint.label ||
          `${hint.from ?? wt[index - 1] ?? ""}${index}${hint.to ?? mut[index - 1] ?? "Δ"}`,
      },
    ];
  }
  const sites = [];
  const n = Math.max(wt.length, mut.length);
  for (let i = 0; i < n; i++) {
    const a = wt[i] || "";
    const b = mut[i] || "";
    if (a !== b) {
      sites.push({
        index: i + 1,
        from: a || "–",
        to: b || "Δ",
        label: `${a || ""}${i + 1}${b || "Δ"}`,
      });
    }
  }
  // Cap labels so a huge indel does not flood the viewer
  return sites.slice(0, 40);
}

function applyLoadedSequence(seq, sourceLabel, opts = {}) {
  const status = document.getElementById("predict-status");
  if (seq.length < 2) {
    predictStatus = `${sourceLabel}: no usable sequence (need ≥2 amino acids).`;
    if (status) status.textContent = predictStatus;
    return false;
  }
  if (seq.length > MAX_PDB_LEN) {
    predictStatus = `${sourceLabel}: ${seq.length.toLocaleString()} aa exceeds max ${MAX_PDB_LEN.toLocaleString()}.`;
    if (status) status.textContent = predictStatus;
    return false;
  }
  if (!seqIsStandardAA(seq)) {
    predictStatus = `${sourceLabel}: only the 20 standard amino acids are supported.`;
    if (status) status.textContent = predictStatus;
    return false;
  }

  if (Object.prototype.hasOwnProperty.call(opts, "variantSites")) {
    variantSites = Array.isArray(opts.variantSites) ? opts.variantSites : [];
  } else if (!opts.keepVariantSites) {
    variantSites = [];
  }

  query = seq;
  const input = document.getElementById("search");
  if (input) {
    // Keep huge sequences in JS state only (same as paste path)
    input.value = seq.length > 500 ? "" : seq;
    input.placeholder =
      seq.length > 500
        ? `Loaded ${seq.length.toLocaleString()} residues — ready to Predict`
        : `e.g. AGPVK or up to ${MAX_PDB_LEN} residues for PDB predict…`;
  }
  const fileMeta = document.getElementById("seq-file-meta");
  if (fileMeta) {
    fileMeta.textContent = `Loaded ${sourceLabel} · ${seq.length.toLocaleString()} residues`;
  }
  predictStatus = `Loaded ${seq.length.toLocaleString()} aa from ${sourceLabel}.`;
  if (status) status.textContent = predictStatus;

  if (seq.length <= PEPTIDE.maxLength) {
    if (trySelectFromQuery(seq)) {
      /* short peptide selected */
    } else {
      updateSuggestions();
    }
  } else {
    const host = document.getElementById("search-panel");
    if (host) host.innerHTML = "";
    pdbResult = null;
    refreshDetail();
  }
  return true;
}

async function loadSequenceFile(file) {
  if (!file) return;
  const name = file.name || "sequence.txt";
  const lower = name.toLowerCase();
  if (
    file.size > 2_000_000 &&
    !lower.endsWith(".txt") &&
    !lower.endsWith(".fasta") &&
    !lower.endsWith(".fa") &&
    !lower.endsWith(".faa") &&
    !lower.endsWith(".seq")
  ) {
    predictStatus = `File ${name} looks too large or wrong type. Use a .txt / FASTA sequence file.`;
    const status = document.getElementById("predict-status");
    if (status) status.textContent = predictStatus;
    return;
  }
  try {
    const text = await file.text();
    const seq = parseSequenceFileText(text);
    applyLoadedSequence(seq, name);
  } catch (e) {
    predictStatus = `Could not read ${name}: ${e.message || e}`;
    const status = document.getElementById("predict-status");
    if (status) status.textContent = predictStatus;
  }
}

function bindSequenceFileDrop() {
  const zone = document.getElementById("seq-drop");
  const input = document.getElementById("seq-file");
  const browse = document.getElementById("btn-seq-browse");
  if (!zone || !input) return;

  browse?.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    const f = input.files?.[0];
    if (f) loadSequenceFile(f);
    input.value = "";
  });

  zone.addEventListener("dragenter", (e) => {
    e.preventDefault();
    zone.classList.add("is-dragover");
  });
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("is-dragover");
  });
  zone.addEventListener("dragleave", (e) => {
    if (!zone.contains(e.relatedTarget)) zone.classList.remove("is-dragover");
  });
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("is-dragover");
    const f = e.dataTransfer?.files?.[0];
    if (f) loadSequenceFile(f);
  });
}

function variantRowHTML(v) {
  const change = escapeHtml(v.label || "");
  const desc = escapeHtml(v.description || "—");
  const id = escapeHtml(v.id || "");
  return `
    <li class="up-var">
      <div class="up-var__main">
        <code class="up-var__change">${change}</code>
        ${id ? `<span class="up-var__id">${id}</span>` : ""}
      </div>
      <p class="up-var__desc">${desc}</p>
      <button
        type="button"
        class="up-var__use"
        data-var-kind="${escapeHtml(v.kind)}"
        data-var-start="${v.start ?? ""}"
        data-var-end="${v.end ?? ""}"
        data-var-from="${escapeHtml(v.original || "")}"
        data-var-to="${escapeHtml(v.alternatives?.[0] ?? "")}"
        title="Load sequence and mark this site in 3D after Predict"
      >
        Use + mark in 3D
      </button>
    </li>`;
}

function uniprotHitsHTML() {
  if (uniprotBusy && !uniprotHits.length && !uniprotEntry) {
    return `<p class="up-status">Searching UniProtKB…</p>`;
  }
  if (!uniprotHits.length) {
    if (uniprotStatus && !uniprotEntry) {
      return `<p class="up-status">${escapeHtml(uniprotStatus)}</p>`;
    }
    return "";
  }
  return `
    <ul class="up-hits" role="listbox">
      ${uniprotHits
        .map((h) => {
          const active =
            uniprotEntry?.accession === h.accession ? " is-active" : "";
          const badge = h.reviewed ? "Swiss-Prot" : "TrEMBL";
          return `
            <li>
              <button type="button" class="up-hit${active}" data-accession="${escapeHtml(h.accession)}">
                <span class="up-hit__ids">
                  <strong>${escapeHtml(h.accession)}</strong>
                  <span>${escapeHtml(h.entryName || "")}</span>
                  <em>${badge}</em>
                </span>
                <span class="up-hit__name">${escapeHtml(h.proteinName || "—")}</span>
                <span class="up-hit__meta">
                  ${escapeHtml(h.gene || "—")}
                  · ${escapeHtml(h.organism || "—")}
                  ${h.length ? ` · ${h.length} aa` : ""}
                </span>
              </button>
            </li>`;
        })
        .join("")}
    </ul>`;
}

function uniprotEntryHTML() {
  const e = uniprotEntry;
  if (!e) {
    if (uniprotBusy) {
      return `<p class="up-status">Loading entry from UniProtKB…</p>`;
    }
    return `
      <p class="up-empty">
        Search UniProtKB by <strong>protein name</strong>, <strong>gene</strong>,
        <strong>Entry Name</strong> (e.g. P53_HUMAN), or <strong>accession</strong> (e.g. P04637).
      </p>`;
  }

  const list = variantTab === "mutant" ? e.mutant : e.natural;
  const filtered = filterVariants(list, variantFilter);
  const shown = filtered.slice(0, variantLimit);
  const remaining = Math.max(0, filtered.length - shown.length);
  const fn = e.functionText
    ? e.functionText.length > 420
      ? `${e.functionText.slice(0, 420)}…`
      : e.functionText
    : "No function annotation in this field set.";

  return `
    <div class="up-entry">
      <div class="up-info">
        <p class="up-info__eyebrow">UniProtKB · ${e.reviewed ? "reviewed" : "unreviewed"}</p>
        <h3 class="up-info__title">${escapeHtml(e.proteinName || e.entryName || e.accession)}</h3>
        <dl class="up-info__dl">
          <div><dt>Accession</dt><dd><code>${escapeHtml(e.accession)}</code></dd></div>
          <div><dt>Entry name</dt><dd><code>${escapeHtml(e.entryName || "—")}</code></dd></div>
          <div><dt>Gene</dt><dd>${escapeHtml(e.genes?.join(", ") || e.gene || "—")}</dd></div>
          <div><dt>Organism</dt><dd>${escapeHtml(e.organism || e.scientificName || "—")}</dd></div>
          <div><dt>Length</dt><dd>${e.length ? `${e.length} aa` : "—"}</dd></div>
          <div><dt>Evidence</dt><dd>${escapeHtml(e.proteinExistence || "—")}</dd></div>
          ${
            e.annotationScore != null
              ? `<div><dt>Annotation</dt><dd>${escapeHtml(String(e.annotationScore))}</dd></div>`
              : ""
          }
        </dl>
        <p class="up-info__fn"><span>Function</span>${escapeHtml(fn)}</p>
        <div class="up-info__actions">
          <button type="button" class="btn-up-seq" id="btn-up-use-wt">
            Use wild-type sequence
          </button>
          ${
            e.uniprotUrl
              ? `<a class="up-info__link" href="${escapeHtml(e.uniprotUrl)}" target="_blank" rel="noopener noreferrer">Open on UniProt</a>`
              : ""
          }
        </div>
        ${uniprotStatus ? `<p class="up-status up-status--inline">${escapeHtml(uniprotStatus)}</p>` : ""}
      </div>

      <div class="up-vars" id="up-vars-panel">
        <div class="up-vars__head">
          <p class="up-vars__title">Sequence variations</p>
          <div class="up-vars__tabs" role="tablist">
            <button
              type="button"
              class="up-vars__tab${variantTab === "natural" ? " is-active" : ""}"
              data-var-tab="natural"
              role="tab"
              aria-selected="${variantTab === "natural"}"
            >
              Natural (${e.natural.length})
            </button>
            <button
              type="button"
              class="up-vars__tab${variantTab === "mutant" ? " is-active" : ""}"
              data-var-tab="mutant"
              role="tab"
              aria-selected="${variantTab === "mutant"}"
            >
              Mutagenesis (${e.mutant.length})
            </button>
          </div>
        </div>
        <label class="up-vars__filter">
          <span>Filter</span>
          <input
            id="up-var-filter"
            type="search"
            placeholder="e.g. 175 · R175H · cancer · VAR_"
            value="${escapeHtml(variantFilter)}"
            autocomplete="off"
            spellcheck="false"
          />
        </label>
        <p class="up-vars__count">
          Showing ${shown.length} of ${filtered.length}
          ${variantFilter.trim() ? " (filtered)" : ""}
          · ${variantTab === "natural" ? "natural polymorphism / disease variants" : "experimental mutagenesis"}
        </p>
        <ul class="up-vars__list">
          ${
            shown.length
              ? shown.map(variantRowHTML).join("")
              : `<li class="up-vars__empty">No ${variantTab === "natural" ? "natural" : "mutagenesis"} variants match.</li>`
          }
        </ul>
        ${
          remaining > 0
            ? `<button type="button" class="up-vars__more" id="up-var-more">
                Show more (${Math.min(VARIANT_PAGE, remaining)} of ${remaining} left)
              </button>`
            : ""
        }
      </div>
    </div>`;
}

function uniprotPanelHTML() {
  return `
    <div id="uniprot-panel" class="uniprot">
      <div class="section-head">
        <h2>UniProtKB</h2>
        <p>Import by name, gene, Entry Name, or accession</p>
      </div>
      <form id="uniprot-form" class="up-search" autocomplete="off">
        <label class="search-box up-search__field">
          <span class="search-box__label">Protein search</span>
          <input
            id="uniprot-q"
            type="search"
            placeholder="e.g. TP53 · P04637 · P53_HUMAN · hemoglobin"
            value="${escapeHtml(uniprotQuery)}"
            autocomplete="off"
            spellcheck="false"
          />
        </label>
        <button type="submit" class="btn-up-search" id="btn-uniprot-search" ${uniprotBusy ? "disabled" : ""}>
          ${uniprotBusy ? "Working…" : "Search UniProt"}
        </button>
      </form>
      <div id="uniprot-hits">${uniprotHitsHTML()}</div>
      <div id="uniprot-entry">${uniprotEntryHTML()}</div>
    </div>`;
}

function refreshUniprotPanel() {
  const root = document.getElementById("uniprot-panel");
  if (!root) return;
  const keepFilter = document.activeElement?.id === "up-var-filter";
  const keepSearch = document.activeElement?.id === "uniprot-q";
  const caret = keepFilter
    ? document.getElementById("up-var-filter")?.selectionStart
    : keepSearch
      ? document.getElementById("uniprot-q")?.selectionStart
      : null;
  root.outerHTML = uniprotPanelHTML();
  bindUniprot();
  if (keepFilter) {
    const el = document.getElementById("up-var-filter");
    if (el) {
      el.focus();
      if (caret != null) el.setSelectionRange(caret, caret);
    }
  } else if (keepSearch) {
    const el = document.getElementById("uniprot-q");
    if (el) {
      el.focus();
      if (caret != null) el.setSelectionRange(caret, caret);
    }
  }
}

async function runUniProtSearch() {
  const q = uniprotQuery.trim();
  if (!q || uniprotBusy) return;
  uniprotBusy = true;
  uniprotStatus = "";
  uniprotHits = [];
  refreshUniprotPanel();
  let autoAccession = null;
  try {
    const { results } = await searchUniProt(q, 12);
    uniprotHits = results;
    uniprotStatus = results.length
      ? `Found ${results.length} hit(s). Select one to load details.`
      : "No UniProtKB entries matched that query.";
    if (results.length === 1) {
      autoAccession = results[0].accession;
    } else {
      const exact = results.find(
        (h) =>
          h.accession.toUpperCase() === q.toUpperCase() ||
          h.entryName.toUpperCase() === q.toUpperCase(),
      );
      if (exact) autoAccession = exact.accession;
    }
  } catch (err) {
    uniprotStatus = err.message || String(err);
  }
  uniprotBusy = false;
  refreshUniprotPanel();
  if (autoAccession) await loadUniProtEntry(autoAccession);
}

async function loadUniProtEntry(accession) {
  uniprotBusy = true;
  uniprotStatus = "";
  variantFilter = "";
  variantLimit = VARIANT_PAGE;
  variantTab = "natural";
  refreshUniprotPanel();
  try {
    uniprotEntry = await fetchUniProtEntry(accession);
    uniprotStatus = `Loaded ${uniprotEntry.accession} · ${uniprotEntry.natural.length} natural · ${uniprotEntry.mutant.length} mutagenesis`;
  } catch (err) {
    uniprotEntry = null;
    uniprotStatus = err.message || String(err);
  } finally {
    uniprotBusy = false;
    refreshUniprotPanel();
  }
}

function standardSeqOnly(raw) {
  return cleanSeq(raw).replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, "");
}

function useUniProtWildType() {
  if (!uniprotEntry?.sequence) {
    uniprotStatus = "No sequence on this entry.";
    refreshUniprotPanel();
    return;
  }
  const seq = standardSeqOnly(uniprotEntry.sequence);
  const ok = applyLoadedSequence(
    seq,
    `${uniprotEntry.accession} (${uniprotEntry.entryName || "UniProt"})`,
    { variantSites: [] },
  );
  if (ok) {
    const stripped = (uniprotEntry.sequence?.length || 0) - seq.length;
    uniprotStatus = stripped
      ? `Wild-type loaded (${seq.length} aa; removed ${stripped} non-standard residue(s)). Ready to Predict.`
      : `Wild-type sequence loaded (${seq.length} aa). Ready to Predict.`;
    refreshUniprotPanel();
  }
}

function useUniProtVariant(btn) {
  if (!uniprotEntry?.sequence) return;
  const start = Number(btn.dataset.varStart);
  const end = Number(btn.dataset.varEnd || btn.dataset.varStart);
  const from = btn.dataset.varFrom || "";
  const to = btn.dataset.varTo ?? "";
  const kind = btn.dataset.varKind || "natural";
  const variant = {
    start: Number.isFinite(start) ? start : null,
    end: Number.isFinite(end) ? end : null,
    original: from,
    alternatives: [to],
    label: `${from}${start}${to || "Δ"}`,
  };
  try {
    const wt = standardSeqOnly(uniprotEntry.sequence);
    const mutated = standardSeqOnly(applySequenceEdit(uniprotEntry.sequence, variant, 0));
    const label = `${uniprotEntry.accession} ${variant.label} (${kind})`;
    const sites = buildVariantSites(wt, mutated, {
      index: variant.start,
      from,
      to,
      label: variant.label,
    });
    const ok = applyLoadedSequence(mutated, label, { variantSites: sites });
    if (ok) {
      uniprotStatus = `Loaded ${kind} ${variant.label} (${mutated.length} aa) · marked in 3D after Predict.`;
      refreshUniprotPanel();
    }
  } catch (err) {
    uniprotStatus = err.message || String(err);
    refreshUniprotPanel();
  }
}

function bindUniprot() {
  const form = document.getElementById("uniprot-form");
  const input = document.getElementById("uniprot-q");
  form?.addEventListener("submit", (e) => {
    e.preventDefault();
    uniprotQuery = input?.value || "";
    runUniProtSearch();
  });
  input?.addEventListener("input", (e) => {
    uniprotQuery = e.target.value;
  });

  document.querySelectorAll(".up-hit").forEach((btn) => {
    btn.addEventListener("click", () => {
      const acc = btn.dataset.accession;
      if (acc) loadUniProtEntry(acc);
    });
  });

  document.getElementById("btn-up-use-wt")?.addEventListener("click", () => {
    useUniProtWildType();
  });

  document.querySelectorAll("[data-var-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      variantTab = btn.dataset.varTab === "mutant" ? "mutant" : "natural";
      variantLimit = VARIANT_PAGE;
      refreshUniprotPanel();
    });
  });

  const vf = document.getElementById("up-var-filter");
  vf?.addEventListener("input", (e) => {
    variantFilter = e.target.value;
    variantLimit = VARIANT_PAGE;
    refreshUniprotPanel();
  });

  document.getElementById("up-var-more")?.addEventListener("click", () => {
    variantLimit += VARIANT_PAGE;
    refreshUniprotPanel();
  });

  document.querySelectorAll(".up-var__use").forEach((btn) => {
    btn.addEventListener("click", () => useUniProtVariant(btn));
  });
}

function render() {
  app.innerHTML = `
    <div class="page">
      <header class="hero">
        <h1 class="brand">PairFold</h1>
        <p class="lede">
          Rule-based short peptides (2–5) plus a PDB-trained fragment network that
          segments longer sequences and assembles a 3D backbone.
        </p>
      </header>

      <section class="legend-panel">
        <div class="section-head">
          <h2>Amino acid colors</h2>
          <p>Each residue keeps one identity color across every combination.</p>
        </div>
        <ul class="legend">${legendHTML()}</ul>
      </section>

      ${detailHTML()}

      <section class="explorer">
        <div class="section-head">
          <h2>Find / predict</h2>
          <p>UniProtKB · paste · file · PDB up to ${MAX_PDB_LEN}</p>
        </div>

        ${uniprotPanelHTML()}

        <label class="search-box">
          <span class="search-box__label">Sequence</span>
          <input
            id="search"
            type="search"
            placeholder="e.g. AGPVK or up to ${MAX_PDB_LEN} residues for PDB predict…"
            value="${escapeHtml(query.length > 500 ? "" : query)}"
            autocomplete="off"
            spellcheck="false"
          />
        </label>

        <div
          id="seq-drop"
          class="seq-drop"
          tabindex="0"
          aria-label="Drop a sequence text file here"
        >
          <input
            id="seq-file"
            type="file"
            accept=".txt,.fasta,.fa,.faa,.seq,text/plain"
            hidden
          />
          <p class="seq-drop__title">Drop a sequence file here</p>
          <p class="seq-drop__hint">.txt / FASTA · or</p>
          <button type="button" id="btn-seq-browse" class="btn-seq-browse">
            Browse file
          </button>
          <p id="seq-file-meta" class="seq-drop__meta"></p>
        </div>

        <div class="predict-bar">
          <button type="button" id="btn-pdb" class="btn-pdb">Predict (PDB)</button>
          <button type="button" id="btn-cancel-predict" class="btn-cancel" hidden>
            Cancel / safety stop
          </button>
          <p id="predict-status" class="predict-status">${escapeHtml(predictStatus)}</p>
        </div>
        <div id="safety-alert" class="safety-alert" role="alert" hidden></div>

        <div id="predict-progress" class="predict-progress" hidden>
          <div class="predict-progress__track">
            <div id="predict-progress-fill" class="predict-progress__fill"></div>
          </div>
          <p id="predict-progress-meta" class="predict-progress__meta"></p>
        </div>

        <div id="search-panel">
          ${suggestionHTML(searchMatches(query))}
        </div>
      </section>

      <footer class="footer">
        <p>
          PDB path: high-res X-ray fragments → torsion Transformer → DP segmentation → assemble.
          This is a local-structure tool, not a full folding predictor.
        </p>
      </footer>
    </div>
  `;

  bind();
  mountViewer();
  bindSegPanel();
  bindUniprot();
}

function bind() {
  const search = document.getElementById("search");
  // Restore long sequences without baking them into innerHTML (avoids multi-MB DOM).
  if (search && query.length > 500 && !search.value) {
    search.value = query;
  }
  let suggestTimer = 0;
  search?.addEventListener("input", (e) => {
    query = e.target.value;
    clearTimeout(suggestTimer);
    // Skip suggestion work for long PDB pastes — it freezes the tab
    const nClean = cleanSeq(query).length;
    if (nClean > PEPTIDE.maxLength) {
      const host = document.getElementById("search-panel");
      if (host) host.innerHTML = "";
      return;
    }
    suggestTimer = window.setTimeout(() => updateSuggestions(), 80);
  });

  search?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const seq = cleanSeq(query);
      if (seq.length > PEPTIDE.maxLength) {
        predictPdb();
        return;
      }
      const matches = searchMatches(query);
      if (trySelectFromQuery(query)) return;
      if (matches[0]) selectPeptide(matches[0].codes);
    }
  });

  document.getElementById("btn-pdb")?.addEventListener("click", () => {
    predictPdb();
  });
  document.getElementById("btn-cancel-predict")?.addEventListener("click", () => {
    abortPredict("Prediction cancelled by user.");
  });

  bindSequenceFileDrop();
  bindResults();
}

function bindResults() {
  document.querySelectorAll(".search-result").forEach((btn) => {
    btn.addEventListener("click", () => {
      const pair = btn.dataset.pair;
      query = pair;
      const input = document.getElementById("search");
      if (input) input.value = pair;
      selectPeptide([...pair]);
    });
  });
}

void activeView;

try {
  if (typeof window.__pairfoldClearBootTimer === "function") {
    window.__pairfoldClearBootTimer();
  }
  render();
} catch (err) {
  console.error(err);
  if (app) {
    app.innerHTML = `
      <div class="page">
        <h1 class="brand">PairFold</h1>
        <p class="empty">Page failed to load: ${escapeHtml(err.message || String(err))}</p>
      </div>
    `;
  }
}
