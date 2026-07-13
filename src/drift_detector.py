"""
drift_detector.py
=================
1. Download reference embeddings (lúc train) từ MinIO
2. Download inference samples từ raw-images/inference-samples/
3. Extract embeddings, tính cosine similarity
4. Log vào MLflow
5. Gửi email thông báo
6. Move inference samples → raw-images/raw_images/ (để gắn nhãn sau)

Threshold:
  >= 0.9  → OK
  <  0.9  → DRIFT
"""

import io
import os
import random
import smtplib
import tempfile
from datetime import datetime
from email.mime.text import MIMEText

import boto3
import mlflow
import numpy as np
import torch
import yaml
from botocore.client import Config

# ── Config ─────────────────────────────────────────────────────────────────────
MINIO_URL             = os.getenv("MINIO_URL", "http://minio-service:9000")
MINIO_ACCESS_KEY      = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY      = os.getenv("MINIO_SECRET_KEY", "")
RAW_BUCKET            = os.getenv("MINIO_BUCKET", "raw-images")
REGISTRY_BUCKET       = os.getenv("REGISTRY_BUCKET", "mlops-registry")
MLFLOW_URI            = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
PARAMS_PATH           = os.getenv("PARAMS_PATH", "/app/params.yaml")
MAX_INFERENCE_SAMPLES = int(os.getenv("MAX_INFERENCE_SAMPLES", "500"))

SMTP_HOST     = "smtp.gmail.com"
SMTP_PORT     = 587
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
NOTIFY_EMAIL  = os.getenv("NOTIFY_EMAIL", "")

DRIFT_THRESHOLD = 0.9


def _s3():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def _load_params() -> dict:
    try:
        with open(PARAMS_PATH) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _download_reference_embeddings(s3) -> np.ndarray:
    try:
        obj  = s3.get_object(Bucket=REGISTRY_BUCKET, Key="current-train-emb/reference_embeddings.npy")
        data = obj["Body"].read()
        return np.load(io.BytesIO(data))
    except s3.exceptions.NoSuchKey:
        raise RuntimeError(
            "Chưa có reference embeddings. Chạy training trước để tạo "
            "mlops-registry/current-train-emb/reference_embeddings.npy"
        )
    except Exception as e:
        raise RuntimeError(f"Không download được reference embeddings: {e}")


def _list_inference_keys(s3) -> list:
    """List tất cả keys trong inference-samples/ — có phân trang (>1000 objects)."""
    keys  = []
    token = None
    while True:
        kwargs = {"Bucket": RAW_BUCKET, "Prefix": "inference-samples/"}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        keys.extend(
            obj["Key"] for obj in resp.get("Contents", [])
            if not obj["Key"].endswith("/")
        )
        if not resp.get("IsTruncated"):
            break
        token = resp["NextContinuationToken"]
    return keys


def _download_to_tmp(s3, keys: list, tmpdir: str) -> list:
    paths = []
    for i, key in enumerate(keys):
        ext   = os.path.splitext(key)[1] or ".jpg"
        local = os.path.join(tmpdir, f"{i}{ext}")
        s3.download_file(RAW_BUCKET, key, local)
        paths.append(local)
    return paths


def _move_to_raw_images(s3, keys: list):
    """Copy inference-samples/ → raw_images/ rồi xóa bản gốc."""
    print(f"\nMoving {len(keys)} images → raw_images/...")
    for key in keys:
        filename = os.path.basename(key)
        dest     = f"raw_images/{filename}"
        s3.copy_object(
            Bucket=RAW_BUCKET,
            CopySource={"Bucket": RAW_BUCKET, "Key": key},
            Key=dest,
        )
        s3.delete_object(Bucket=RAW_BUCKET, Key=key)
    print(f"  Done.")


def _cosine_similarity(ref_emb: np.ndarray, inf_emb: np.ndarray) -> float:
    ref_c = ref_emb.mean(axis=0)
    inf_c = inf_emb.mean(axis=0)
    ref_c /= (np.linalg.norm(ref_c) + 1e-8)
    inf_c /= (np.linalg.norm(inf_c) + 1e-8)
    return float(np.dot(ref_c, inf_c))


def _send_email(score: float, is_drift: bool, task_name: str,
                n_ref: int, n_inf: int):
    if not SMTP_USER or not NOTIFY_EMAIL:
        print("  SMTP chưa cấu hình — bỏ qua email.")
        return

    status  = "⚠️ DRIFT DETECTED" if is_drift else "✅ No Drift"
    subject = f"[MLOps] Data Drift Report — {task_name} — {status}"
    body    = f"""
Data Drift Detection Report
===========================
Task       : {task_name}
Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status     : {status}

Cosine Similarity : {score:.4f}
Threshold         : {DRIFT_THRESHOLD}

Reference samples : {n_ref:,}
Inference samples : {n_inf:,}

{"⚠️  Score dưới ngưỡng. Nên xem xét thu thập thêm data và retrain." if is_drift else "✅  Distribution inference tương đồng với training data."}

Xem chi tiết tại MLflow: {MLFLOW_URI}
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = NOTIFY_EMAIL

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"  Email sent → {NOTIFY_EMAIL}")
    except Exception as e:
        print(f"  Email failed: {e}")


def main():
    cfg       = _load_params()
    data_cfg  = cfg.get("data", {})
    emb_cfg   = cfg.get("embedding", {})

    model_name = emb_cfg.get("model", "MobileNetV3Small")
    img_size   = data_cfg.get("img_size", 224)
    batch_size = data_cfg.get("batch_size", 32)
    task_name  = cfg.get("pipeline", {}).get("task_name", "unknown")
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    s3 = _s3()

    # ── 1. Reference embeddings ────────────────────────────────────────────────
    print("Downloading reference embeddings...")
    ref_emb = _download_reference_embeddings(s3)
    print(f"  shape: {ref_emb.shape}")

    # ── 2. Inference images ────────────────────────────────────────────────────
    all_keys = _list_inference_keys(s3)
    if not all_keys:
        print("Không có inference samples — bỏ qua.")
        return

    keys = (random.sample(all_keys, MAX_INFERENCE_SAMPLES)
            if len(all_keys) > MAX_INFERENCE_SAMPLES else all_keys)
    print(f"Inference samples: {len(keys)} / {len(all_keys)}")

    # ── 3. Extract embeddings ──────────────────────────────────────────────────
    from embedding_utils import extract_embeddings

    with tempfile.TemporaryDirectory() as tmpdir:
        paths   = _download_to_tmp(s3, keys, tmpdir)
        inf_emb = extract_embeddings(paths, model_name, img_size, batch_size, device).numpy()

    print(f"  shape: {inf_emb.shape}")

    # ── 4. Cosine similarity ───────────────────────────────────────────────────
    score    = _cosine_similarity(ref_emb, inf_emb)
    is_drift = score < DRIFT_THRESHOLD

    print(f"\nCosine similarity : {score:.4f}  (threshold={DRIFT_THRESHOLD})")
    print("⚠️  DRIFT DETECTED!" if is_drift else "✅ No significant drift.")

    # ── 5. Log to MLflow ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(f"drift-monitoring/{task_name}")

    with mlflow.start_run(run_name=f"drift_{datetime.now().strftime('%y%m%d_%H%M%S')}"):
        mlflow.log_params({
            "backbone":    model_name,
            "n_reference": ref_emb.shape[0],
            "n_inference": inf_emb.shape[0],
            "threshold":   DRIFT_THRESHOLD,
        })
        mlflow.log_metric("cosine_similarity", score)
        mlflow.set_tag("drift_status", "DRIFT" if is_drift else "OK")

    print("Logged to MLflow.")

    # ── 6. Gửi email ──────────────────────────────────────────────────────────
    print("\nSending email notification...")
    _send_email(score, is_drift, task_name, ref_emb.shape[0], inf_emb.shape[0])

    # ── 7. Move inference samples → raw_images/ ───────────────────────────────
    _move_to_raw_images(s3, all_keys)


if __name__ == "__main__":
    main()
