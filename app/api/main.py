import base64
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.db.database import get_all_predictions, get_prediction_by_id, init_db, save_prediction
from app.llm.report_generator import generate_report
from app.xai.gradcam import run_gradcam

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
GRADCAM_DIR = STATIC_DIR / "gradcam"

for d in (STATIC_DIR, UPLOAD_DIR, GRADCAM_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

app = FastAPI(title="Medical AI Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    prediction_id: int
    predicted_class: str
    confidence: float
    gradcam_image: str
    llm_report: str


class HistoryItem(BaseModel):
    id: int
    timestamp: str
    image_filename: str
    predicted_class: str
    confidence: float
    gradcam_image_path: str | None
    llm_report: str | None


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    uid = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{uid}{ext}"
    contents = await file.read()
    upload_path.write_bytes(contents)

    try:
        pred_label, confidence, gradcam_img = run_gradcam(str(upload_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM inference failed: {e}")

    gradcam_path = GRADCAM_DIR / f"{uid}.png"
    gradcam_img.save(gradcam_path)

    try:
        llm_report = generate_report(pred_label, confidence)
    except Exception as e:
        llm_report = f"(Report generation failed: {e})"

    pred_id = save_prediction(
        image_filename=file.filename,
        predicted_class=pred_label,
        confidence=confidence,
        gradcam_image_path=str(gradcam_path.relative_to(STATIC_DIR)),
        llm_report=llm_report,
    )

    gradcam_b64 = base64.b64encode(gradcam_path.read_bytes()).decode()

    return PredictionResponse(
        prediction_id=pred_id,
        predicted_class=pred_label,
        confidence=confidence,
        gradcam_image=gradcam_b64,
        llm_report=llm_report,
    )


@app.get("/history")
def history():
    preds = get_all_predictions()
    return [
        HistoryItem(
            id=p.id,
            timestamp=p.timestamp.isoformat(),
            image_filename=p.image_filename,
            predicted_class=p.predicted_class,
            confidence=p.confidence,
            gradcam_image_path=p.gradcam_image_path,
            llm_report=p.llm_report,
        )
        for p in preds
    ]


@app.get("/history/{pred_id}")
def history_detail(pred_id: int):
    p = get_prediction_by_id(pred_id)
    if not p:
        raise HTTPException(status_code=404, detail="Prediction not found")

    gradcam_b64 = None
    if p.gradcam_image_path:
        full_path = STATIC_DIR / p.gradcam_image_path
        if full_path.exists():
            gradcam_b64 = base64.b64encode(full_path.read_bytes()).decode()

    return {
        "id": p.id,
        "timestamp": p.timestamp.isoformat(),
        "image_filename": p.image_filename,
        "predicted_class": p.predicted_class,
        "confidence": p.confidence,
        "gradcam_image": gradcam_b64,
        "llm_report": p.llm_report,
    }


@app.get("/gradcam/{filename}")
def serve_gradcam(filename: str):
    path = GRADCAM_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")
