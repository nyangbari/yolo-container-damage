from __future__ import annotations

import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("ultralytics is not installed. Run: pip install -r requirements.txt") from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models" / "best.pt"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONFIDENCE = 0.25
DEFAULT_INFERENCE_CONFIDENCE = 0.01

app = FastAPI(title="Container Damage API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: YOLO | None = None
_model_path: Path | None = None


def resolve_model_path() -> Path:
    override = os.environ.get("YOLO_MODEL_PATH")
    return Path(override).expanduser().resolve() if override else DEFAULT_MODEL_PATH


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def parse_probability(raw_value: float | None, *, default: float, minimum: float) -> float:
    if raw_value is None:
        return default
    return min(1.0, max(minimum, float(raw_value)))


def get_model() -> YOLO:
    global _model, _model_path

    model_path = resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train first so models/best.pt exists."
        )

    if _model is None or _model_path != model_path:
        _model = YOLO(str(model_path))
        _model_path = model_path

    return _model


def make_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> dict:
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)

    return {
        "x": round(x1, 2),
        "y": round(y1, 2),
        "width": round(box_width, 2),
        "height": round(box_height, 2),
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "normalized": {
            "x": round(x1 / width, 6),
            "y": round(y1 / height, 6),
            "width": round(box_width / width, 6),
            "height": round(box_height / height, 6),
            "x1": round(x1 / width, 6),
            "y1": round(y1 / height, 6),
            "x2": round(x2 / width, 6),
            "y2": round(y2 / height, 6),
        },
    }


def run_inference(
    image_path: Path,
    *,
    requested_confidence: float,
    inference_confidence: float,
) -> dict:
    model = get_model()
    results = model.predict(source=str(image_path), verbose=False, conf=inference_confidence)
    result = results[0]
    image_height, image_width = result.orig_shape

    detections: list[dict] = []
    max_confidence = 0.0
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        for xyxy, cls_id, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
            class_id = int(cls_id)
            confidence_score = float(score)
            max_confidence = max(max_confidence, confidence_score)
            x1, y1, x2, y2 = xyxy

            detections.append(
                {
                    "label": result.names.get(class_id, str(class_id)),
                    "class_id": class_id,
                    "confidence": round(confidence_score, 6),
                    "confidence_percent": round(confidence_score * 100, 2),
                    "bbox": make_bbox(x1, y1, x2, y2, image_width, image_height),
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)

    return {
        "image": {
            "width": image_width,
            "height": image_height,
        },
        "thresholds": {
            "requested": requested_confidence,
            "requested_percent": round(requested_confidence * 100, 2),
            "inference": inference_confidence,
            "inference_percent": round(inference_confidence * 100, 2),
            "client_side_filtering": True,
        },
        "summary": {
            "all_detections": len(detections),
            "max_confidence": round(max_confidence, 6),
            "max_confidence_percent": round(max_confidence * 100, 2),
        },
        "detections": detections,
    }


@app.get("/")
def root() -> dict:
    model_path = resolve_model_path()
    return {
        "service": "container-damage-api",
        "framework": "fastapi",
        "model_ready": model_path.exists(),
        "model_path": str(model_path),
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "detect": "POST /api/detect",
        },
    }


@app.get("/health")
def health() -> dict:
    model_path = resolve_model_path()
    return {
        "ok": True,
        "model_ready": model_path.exists(),
        "model_path": str(model_path),
    }


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    confidence: float = Form(DEFAULT_CONFIDENCE),
    inference_confidence: float = Form(DEFAULT_INFERENCE_CONFIDENCE),
) -> dict:
    if not image.filename:
        raise HTTPException(status_code=400, detail="image file is required")

    if not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="supported formats: jpg, jpeg, png, bmp, webp")

    confidence = parse_probability(confidence, default=DEFAULT_CONFIDENCE, minimum=0.0)
    inference_confidence = parse_probability(
        inference_confidence,
        default=DEFAULT_INFERENCE_CONFIDENCE,
        minimum=0.0,
    )
    inference_confidence = min(inference_confidence, confidence)

    filename = Path(image.filename).name
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await image.read())

    try:
        payload = run_inference(
            temp_path,
            requested_confidence=confidence,
            inference_confidence=inference_confidence,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
        await image.close()

    return {
        "ok": True,
        "filename": filename,
        **payload,
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8787, reload=True)
