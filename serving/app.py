"""
serving/app.py
FastAPI service — Generic Image Classifier
Load model từ MLflow Registry, expose /predict endpoint.
Kèm Prometheus metrics để monitor.
"""

import os
import ast
import time
import io
import uuid
import random
import yaml
import logging
from contextlib import asynccontextmanager

import boto3
from botocore.client import Config

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import mlflow.pytorch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config từ environment variables ──────────────────────────────────────────
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME    = os.getenv("MODEL_NAME", "image-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "Production")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load params.yaml để lấy img_size và experiment_name ──────────────────────
try:
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)
    IMG_SIZE        = cfg["data"]["img_size"]
    EXPERIMENT_NAME = f"{cfg['mlflow']['experiment_base_name']}/{cfg['pipeline']['task_name']}"
except Exception:
    IMG_SIZE        = 32      # fallback
    EXPERIMENT_NAME = "image-generic"

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Tổng số request predict",
    ["status"]
)
REQUEST_LATENCY = Histogram(
    "predict_latency_seconds",
    "Thời gian xử lý mỗi request",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)
PREDICTION_DIST = Counter(
    "prediction_class_total",
    "Số lần predict mỗi class",
    ["class_name"]
)

# ── MinIO config ─────────────────────────────────────────────────────────────
MINIO_URL        = os.getenv("MINIO_URL", "http://minio-service:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MINIO_BUCKET     = os.getenv("MINIO_BUCKET", "raw-images")
SAMPLE_RATE     = 0.14   # 14% cố định

_minio_client = None

def _get_minio():
    global _minio_client
    if _minio_client is None:
        _minio_client = boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )
    return _minio_client

def _maybe_save_image(contents: bytes, filename: str):
    """Lưu ảnh vào MinIO với xác suất ~14%."""
    if random.random() >= SAMPLE_RATE:
        return
    try:
        key = f"inference-samples/{uuid.uuid4().hex}_{filename}"
        _get_minio().put_object(
            Bucket=MINIO_BUCKET,
            Key=key,
            Body=contents,
            ContentLength=len(contents),
        )
        logger.info(f"Sampled: {key}")
    except Exception as e:
        logger.warning(f"Sample save failed: {e}")

# ── Global model state ────────────────────────────────────────────────────────
model_state = {"model": None, "class_names": [], "version": None}

# ── Transform — đọc img_size từ config ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # grayscale→3ch, RGB không đổi
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# ── Load model khi app khởi động ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading '{MODEL_NAME}' version='{MODEL_VERSION}' from {MLFLOW_URI}")
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
        os.environ["MLFLOW_ARTIFACT_URI"] = MLFLOW_URI

        # Load model từ Registry
        if MODEL_VERSION.isdigit():
            uri = f":/{MODEL_NAME}/{MODEL_VERSION}"
        else:
            uri = f"models:/{MODEL_NAME}@{MODEL_VERSION}"
        model = mlflow.pytorch.load_model(uri, map_location=DEVICE)
        model.eval()

        # Lấy class_names từ run mới nhất của experiment
        client = mlflow.MlflowClient()
        exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs   = client.search_runs(
            [exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        class_names = ast.literal_eval(
            runs[0].data.params.get("classes", "[]")
        )

        model_state["model"]       = model
        model_state["class_names"] = class_names
        model_state["version"]     = MODEL_VERSION

        logger.info(f"Model loaded! Classes: {class_names}")

    except Exception as e:
        logger.error(f"Không load được model: {e}")

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Generic Vision API",
    description="Generic Vision — MLOps Demo",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model chưa load được")
    return {
        "status":   "ok",
        "model":    MODEL_NAME,
        "version":  model_state["version"],
        "classes":  model_state["class_names"],
        "img_size": IMG_SIZE,
        "device":   DEVICE,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    if not file.content_type.startswith("image/"):
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh")

    t_start = time.time()

    try:
        contents = await file.read()
        _maybe_save_image(contents, file.filename or "image.jpg")
        img      = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor   = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model_state["model"](tensor)
            probs  = F.softmax(logits, dim=1)[0]

        top3_probs, top3_idx = probs.topk(min(3, len(model_state["class_names"])))
        class_names = model_state["class_names"]

        top3 = [
            {"class": class_names[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top3_probs, top3_idx)
        ]
        best    = top3[0]
        latency = time.time() - t_start

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(latency)
        PREDICTION_DIST.labels(class_name=best["class"]).inc()

        return JSONResponse({
            "prediction": best["class"],
            "confidence": best["confidence"],
            "top3":       top3,
            "latency_ms": round(latency * 1000, 1),
        })

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        logger.error(f"Predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    return {"message": "Image Classifier API", "docs": "/docs"}