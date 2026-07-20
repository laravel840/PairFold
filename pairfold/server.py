"""FastAPI inference server for PDB-trained fragment model.

Also serves the built web UI from ``dist/`` (run ``npm run build`` first).
Open http://127.0.0.1:8000/ — do not open index.html via file://.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from pathlib import Path
from queue import Empty, Queue
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import MAX_QUERY_LEN, TERTIARY_MAX_LEN, VIEW_3D_MAX_LEN
from .predict import FragmentPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"

app = FastAPI(
    title="PairFold PDB Fragment Predictor",
    description=(
        "Short-peptide torsion model trained on high-resolution PDB fragments. "
        "Longer sequences are segmented into 2–5 residue windows and assembled. "
        "This is NOT AlphaFold."
    ),
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Optional[FragmentPredictor] = None


def get_predictor() -> FragmentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FragmentPredictor()
    return _predictor


class PredictRequest(BaseModel):
    sequence: str = Field(..., min_length=2, max_length=MAX_QUERY_LEN)


class SegmentOut(BaseModel):
    start: int
    end: int
    seq: str
    confidence: float


class PredictResponse(BaseModel):
    sequence: str
    mode: str
    segmentation: List[SegmentOut]
    phis: List[float]
    psis: List[float]
    confidence: List[float]
    structure: dict
    model: str
    device: str
    note: str


@app.get("/health")
def health():
    try:
        p = get_predictor()
        cp = p._get_contact_predictor()
        return {
            "ok": True,
            "device": str(p.dev),
            "ckpt": p.ckpt_path,
            "contact_ckpt": getattr(cp, "ckpt_path", "") if cp else "",
            "contact_enabled": bool(cp and cp.enabled),
            "max_query_len": MAX_QUERY_LEN,
            "view_3d_max_len": VIEW_3D_MAX_LEN,
            "tertiary_max_len": TERTIARY_MAX_LEN,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        out = get_predictor().predict_sequence(req.sequence)
        return out
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/predict/stream")
def predict_stream(req: PredictRequest):
    """
    NDJSON stream: progress lines then a final result/error line.
      {"type":"progress","pct":12.5,"message":"...","eta_s":30.0,...}
      {"type":"result","data":{...}}
      {"type":"error","detail":"..."}
    """
    q: Queue = Queue()

    def on_progress(ev: dict) -> None:
        q.put(("progress", ev))

    def worker() -> None:
        from .mem_guard import MemoryGuardError, guard_rss, release_caches

        try:
            guard_rss("predict_stream_start")
            out = get_predictor().predict_sequence(req.sequence, progress=on_progress)
            q.put(("done", out))
        except FileNotFoundError as e:
            q.put(("error", str(e)))
        except ValueError as e:
            q.put(("error", str(e)))
        except MemoryGuardError as e:
            release_caches()
            q.put(("error", str(e)))
        except Exception as e:
            release_caches()
            q.put(("error", str(e)))

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        # Heartbeats keep the browser stall-watchdog alive during heavy phases
        # (early contact fold / clash assembly) that may not emit for >25s.
        last_progress = {
            "pct": 1.0,
            "message": "Starting…",
            "elapsed_s": None,
            "eta_s": None,
            "n": len(req.sequence or ""),
        }
        deadline = time.time() + 14400
        while True:
            remaining = max(0.1, deadline - time.time())
            try:
                kind, payload = q.get(timeout=min(5.0, remaining))
            except Empty:
                if time.time() >= deadline:
                    yield json.dumps({"type": "error", "detail": "Prediction timed out"}) + "\n"
                    break
                hb = dict(last_progress)
                hb["heartbeat"] = True
                if hb.get("message") and "(working…)" not in str(hb["message"]):
                    hb["message"] = f"{hb['message']} (working…)"
                yield json.dumps({"type": "progress", **hb}) + "\n"
                continue
            if kind == "progress":
                last_progress = {k: v for k, v in payload.items()}
                yield json.dumps({"type": "progress", **payload}) + "\n"
            elif kind == "done":
                n = len(payload.get("sequence") or "")
                # Keep the client stall-watchdog alive before a large JSON dump
                yield json.dumps(
                    {
                        "type": "progress",
                        "pct": 99.0,
                        "message": "Sending result…",
                        "elapsed_s": None,
                        "eta_s": 0.0,
                        "n": n,
                    }
                ) + "\n"
                yield json.dumps({"type": "result", "data": payload}) + "\n"
                break
            else:
                yield json.dumps({"type": "error", "detail": payload}) + "\n"
                break

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


def _mount_ui() -> None:
    """Serve the Vite-built HTML app from dist/ (same origin as the API)."""
    index_html = DIST_DIR / "index.html"
    if not index_html.is_file():
        @app.get("/")
        def ui_missing():
            return {
                "ok": True,
                "api": "PairFold",
                "ui": "missing",
                "hint": "Run npm run build (or double-click Web/Open PairFold.vbs) then reopen http://127.0.0.1:8000/",
            }

        return

    assets = DIST_DIR / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/")
    def ui_index():
        # Avoid stale UI after npm run build while the server stays up
        return FileResponse(
            index_html,
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    viewer = DIST_DIR / "viewer.html"
    if viewer.is_file():

        @app.get("/viewer.html")
        def ui_viewer():
            return FileResponse(
                viewer,
                headers={"Cache-Control": "no-cache, must-revalidate"},
            )


_mount_ui()


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="PairFold API + HTML app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the HTML app in a browser",
    )
    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run("pairfold.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
