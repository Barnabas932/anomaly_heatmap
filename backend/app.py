from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import shutil
import os

import torch
import timm

from backend.vit_heatmaps import run_heatmap_pair  # lentebb adom

APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = (APP_DIR.parent / "frontend" / "dist").resolve()

app = FastAPI()

# --- serve built frontend ---
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

@app.get("/api/health")
def health():
    return {"ok": True}

def load_model(device: str = "cpu"):
    base = timm.create_model("vit_base_patch16_224", pretrained=True)
    base.reset_classifier(0)  # remove classifier
    model = torch.nn.Sequential(
        base,
        torch.nn.Linear(base.num_features, 2)
    )
    model.eval()
    model.to(device)
    return model

MODEL = None

@app.on_event("startup")
def _startup():
    global MODEL
    device = "cpu"
    MODEL = load_model(device=device)

@app.post("/api/heatmap")
async def heatmap(
    ok_files: list[UploadFile] = File(...),
    nok_file: UploadFile = File(...),
    mode: str = Form("gradcam"),   # "gradcam" | "attn"
    alpha: float = Form(0.45),
):
    """
    ok_files: legalább 10 OK kép
    nok_file: 1 db NOK kép
    mode: gradcam / attn
    """
    if len(ok_files) < 10:
        return JSONResponse({"error": "Legalább 10 OK képet tölts fel."}, status_code=400)

    job_id = str(uuid.uuid4())
    workdir = Path("/tmp") / f"heatmap_{job_id}"
    inp_ok = workdir / "ok"
    inp_nok = workdir / "nok"
    out_dir = workdir / "out"
    inp_ok.mkdir(parents=True, exist_ok=True)
    inp_nok.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save uploads
    for f in ok_files:
        p = inp_ok / f.filename
        with p.open("wb") as w:
            shutil.copyfileobj(f.file, w)

    nok_path = inp_nok / nok_file.filename
    with nok_path.open("wb") as w:
        shutil.copyfileobj(nok_file.file, w)

    out_img_path = out_dir / f"{nok_path.stem}_{mode}.png"
    try:
        run_heatmap_pair(
            model=MODEL,
            img_path=nok_path,
            out_path=out_img_path,
            mode=mode,
            alpha=float(alpha),
            device_str="cpu"
        )
    except Exception as e:
        return JSONResponse({"error": f"Heatmap hiba: {e}"}, status_code=500)

    return {
        "job_id": job_id,
        "result_png": f"/api/result/{job_id}/{out_img_path.name}",
        "mode": mode,
        "alpha": float(alpha),
        "ok_count": len(ok_files),
        "nok_name": nok_file.filename,
    }

@app.get("/api/result/{job_id}/{filename}")
def get_result(job_id: str, filename: str):
    p = Path("/tmp") / f"heatmap_{job_id}" / "out" / filename
    if not p.exists():
        return JSONResponse({"error": "Nincs ilyen eredmény."}, status_code=404)
    return FileResponse(str(p), media_type="image/png", filename=filename)
