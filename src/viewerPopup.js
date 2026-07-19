import "./style.css";
import { destroyPeptide3D, mountPeptide3D } from "./viewer3D.js";

function loadPayload() {
  try {
    if (window.opener && typeof window.opener.__pairfoldGet3DPayload === "function") {
      const fromOpener = window.opener.__pairfoldGet3DPayload();
      if (fromOpener?.phis?.length) return fromOpener;
    }
  } catch {
    /* cross-origin or closed */
  }
  try {
    const raw = sessionStorage.getItem("pairfold_3d_payload");
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed?.phis?.length) return parsed;
    }
  } catch {
    /* ignore */
  }
  return null;
}

function boot() {
  const payload = loadPayload();
  const title = document.getElementById("viewer-title");
  const host = document.getElementById("viewer3d");
  if (!host) return;

  if (!payload) {
    host.innerHTML =
      '<p class="empty">No structure available. Open 3D from the main PairFold page.</p>';
    return;
  }

  const n = payload.phis?.length || payload.sequence?.length || 0;
  const nAtoms = payload.structure?.atoms?.length || 0;
  if (title) {
    title.textContent = n
      ? nAtoms > 0
        ? `PairFold 3D · ${n} residues · ${nAtoms} atoms`
        : `PairFold 3D · ${n} residues`
      : "PairFold 3D";
  }

  // Ensure layout has a real size before WebGL init
  requestAnimationFrame(() => {
    try {
      destroyPeptide3D(host);
      // Prefer client all-atom so Stage-2/3 shows even if API omitted sidechains
      const n = payload.phis?.length || 0;
      const viewInput =
        n > 0 && n <= 256
          ? { ...payload, structure: null, caTrace: false }
          : payload;
      mountPeptide3D(host, viewInput);
      if (!host.querySelector("canvas")) {
        host.innerHTML =
          '<p class="empty">3D canvas failed to start. Try closing and opening again.</p>';
      }
    } catch (err) {
      host.innerHTML = `<p class="empty">3D failed: ${String(err.message || err)}</p>`;
      console.error(err);
    }
  });
}

boot();

window.addEventListener("beforeunload", () => {
  destroyPeptide3D();
});
