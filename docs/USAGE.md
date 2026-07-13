# Hướng Dẫn Sử Dụng

---

## Cấu hình `params.yaml`

Không sửa trực tiếp file trong `src/config/`. Copy ra rồi chỉnh:

```bash
cp src/config/params_config_cls.yaml src/params.yaml   # Classification
cp src/config/params_config_dct.yaml src/params.yaml   # Detection
```

### Classification
```yaml
pipeline:
  model_type: "classification"
  task_name: "ten_bai_toan"           # Tên đăng ký trên MLflow

data:
  batch_size: 32
  img_size: 224

embedding:
  model: "MobileNetV3Small"           # Dùng cho drift detection

optuna_tuning:
  n_trials: 50
  max_num_epochs: 50
  storage_db: "sqlite:///optuna_mlops.db"
  pruner: "median"

training:
  optimizer_choices: ["Adam", "SGD", "RMSprop", "AdamW"]
  lr_min: 1.0e-5
  lr_max: 1.0e-2

models:
  - "ResNet18"
  - "ResNet50"
  - "MobileNetV3Small"
  - "EfficientNetB0"
  - "SimpleCNN"

mlflow:
  experiment_base_name: "generic-vision-pipeline"
```

### Detection
```yaml
pipeline:
  model_type: "detection"
  task_name: "ten_bai_toan"

data:
  batch_size: 32
  img_size: 640

optuna_tuning:
  n_trials: 50
  max_num_epochs: 50
  storage_db: "sqlite:///optuna_mlops.db"
  pruner: "median"

training:
  optimizer_choices: ["AdamW", "SGD", "Adam"]
  lr_min: 1.0e-5
  lr_max: 1.0e-2

models:
  - "YOLOv8n"
  - "YOLOv8s"
  - "YOLOv8m"

mlflow:
  experiment_base_name: "generic-vision-pipeline"
```

---

## Khi Có Data Mới

### Cách 1 — Upload qua Labeling Tool (http://localhost:3001)

Mở tab **Upload**, chọn folder. Tool tự detect format:

**Classification — ImageFolder:**
```
my_dataset/
  cat/img1.jpg       ← subfolder = class name
  dog/img2.jpg
  bird/img3.jpg
```
→ Tự động gắn nhãn theo tên subfolder. Ảnh ở root folder → UNLABELED, gắn tay sau.

**Detection — YOLO format:**
```
my_dataset/
  classes.txt        ← tên class theo thứ tự index (1 dòng/class)
  images/img1.jpg    ← ảnh
  labels/img1.txt    ← class_idx cx cy w h (normalized 0-1)
```
→ Tự động tạo class, import bbox, mark LABELED. Ảnh không có label → UNLABELED.

**File thường (không có folder):** upload nhiều file → tất cả UNLABELED, gắn nhãn sau trong tab Label.

### Cách 2 — Gắn nhãn mới qua Labeling Tool

```bash
# 1. Mở Labeling Tool: http://localhost:3001
#    Upload ảnh → gắn nhãn → submit

# 2. Tạo snapshot — bấm trong app: tab Progress → "Tạo Snapshot"
# → tạo snapshot_vX.json + dataset_vX.json + test_vX.json trên MinIO

# 3. Cập nhật version + push → CI/CD tự chạy
echo "v1" > dataset_version.txt
git add dataset_version.txt
git commit -m "dataset v1"
git tag v1.0
git push origin main --tags
```

---

## Khi Có Code / Config Mới

```bash
# Sửa bất kỳ file nào trong src/, src/params.yaml, hoặc src/models/*.py
git add .
git commit -m "mo ta thay doi"
git push origin main
# → CI/CD detect thay đổi → train lại → deploy API mới
```

**Thêm model custom** — không cần rebuild image:

1. Tạo `src/models/custom2.py` với class model
2. `git add src/models/custom2.py && git commit && git push`
3. CI/CD cập nhật ConfigMap `model-code` từ `src/models/` → inject vào container lúc runtime

```yaml
# Thêm vào models list trong params.yaml
models:
  - "ResNet18"
  - "MyCustomModel"   # tên class trong custom.py hoặc custom2.py
```

---

## Chạy Thủ Công

Dùng khi muốn test nhanh mà không push lên GitHub:

```bash
# Cập nhật ConfigMaps
kubectl create configmap model-code --from-file=src/models/ -n mlops --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap model-params --from-file=params.yaml=src/params.yaml -n mlops --dry-run=client -o yaml | kubectl apply -f -
DATASET_VERSION=$(cat dataset_version.txt | tr -d '[:space:]')
kubectl create configmap dataset-version --from-literal=DATASET_VERSION=$DATASET_VERSION -n mlops --dry-run=client -o yaml | kubectl apply -f -

# Xóa trainer Job cũ rồi deploy lại
kubectl delete job trainer -n mlops --ignore-not-found
helm upgrade mlops mlops_chart/ --namespace mlops \
  --set trainer.enabled=true --set api.enabled=true

# Theo dõi log
kubectl logs -n mlops -l job-name=trainer -c trainer -f

# Sau khi trainer xong, restart API
kubectl rollout restart deployment/api -n mlops
```

---

## CI/CD Pipeline

```
git push origin main
        ↓
GitHub Actions trigger khi:
  src/**, src/params.yaml, dataset_version.txt thay đổi
  hoặc bấm tay trên GitHub UI (workflow_dispatch)
        ↓
Kiểm tra dataset_version.txt
  ├── Rỗng → skip (v0.0, chưa có snapshot)
  └── Có version → chạy full pipeline:
        1. Build Docker images nếu stable files thay đổi
        2. Load images vào Minikube
        3. Tạo ConfigMaps: model-code, model-params, dataset-version
        4. Xóa trainer Job cũ
        5. helm upgrade → trainer Job chạy:
           download dataset → train (Optuna × N trials) → MLflow
        6. Đợi trainer complete
        7. Restart API → health check
        ↓
Model mới đang serve tại http://localhost:8000
```

> **⚠️ GitHub Actions mặc định cancel job sau 6 giờ** — kể cả khi không khai báo `timeout-minutes` trong workflow. Đây là giới hạn cứng của GitHub, không phải do bạn set. Training nhiều trials / dataset lớn rất dễ bị cancel. Phải khai báo rõ để ghi đè:
>
> ```yaml
> # .github/workflows/mlops_pipeline.yml
> jobs:
>   train:
>     timeout-minutes: 720   # 12 giờ — đổi tuỳ nhu cầu, hoặc để số lớn như 1440 (24h)
> ```

---

## Drift Detection

API tự động lưu ~14% ảnh predict vào `raw-images/inference-samples/`. Mỗi giờ, CronJob `drift-detector` chạy:

1. Tải reference embeddings (lưu lúc train) từ MinIO
2. Extract embeddings các ảnh inference
3. Tính cosine similarity giữa 2 distribution
4. Gửi email kết quả — dù drift hay không
5. Move toàn bộ ảnh inference → `raw-images/raw_images/` (để gắn nhãn)

**Threshold:** `< 0.9` = drift, `≥ 0.9` = OK.

### Bật drift detection

Trong `mlops_chart/values.yaml`:

```yaml
driftDetector:
  enabled: true
  schedule: "0 * * * *"        # mỗi giờ — đổi tuỳ ý (VD: "0 */2 * * *" = mỗi 2h)
  notifyEmail: "you@gmail.com"  # email nhận thông báo

secret:
  smtpUser: "sender@gmail.com"      # Gmail gửi đi
  smtpPassword: "xxxx xxxx xxxx xxxx"  # Gmail App Password
```

### Tạo Gmail App Password

Gmail không cho dùng password thường để gửi mail qua SMTP. Phải dùng **App Password**:

1. Vào [myaccount.google.com](https://myaccount.google.com) → **Security**
2. Bật **2-Step Verification** (bắt buộc)
3. Tìm **App passwords** (gõ vào ô search nếu không thấy)
4. Chọn app: `Mail` → Generate
5. Copy 16 ký tự (dạng `xxxx xxxx xxxx xxxx`) → điền vào `smtpPassword`

> App Password ≠ password Gmail. Mất thì generate lại, không ảnh hưởng tài khoản.

### Deploy sau khi cấu hình

```bash
helm upgrade mlops mlops_chart/ --namespace mlops -f mlops_chart/values.yaml
```

### Chạy thủ công (không chờ schedule)

```bash
kubectl create job drift-manual --from=cronjob/drift-detector -n mlops

# Xem log
kubectl logs -n mlops -l job-name=drift-manual -f
```

### Xem kết quả trong MLflow

Experiment: `drift-monitoring/<task_name>` — mỗi lần chạy log `cosine_similarity` và tag `drift_status`.

---

## Luồng MLOps Tổng Thể

```
[Đội gắn nhãn]
  Labeling Tool → MinIO (raw-images/raw_images/) + PostgreSQL (metadata)
                          │
                   POST /snapshots → MinIO registry (snapshot/dataset/test vX)
                          │
[Code/Config] → GitHub Actions
                          │
              ConfigMaps: model-code | model-params | dataset-version
                          │
                   [Trainer Job]
                   registry_loader: download MinIO + PostgreSQL
                   Optuna × N trials → MLflow nested runs
                   Save reference_embeddings.npy → MinIO registry
                          │
                   [MLflow] → alias "champion"
                          │
                   [FastAPI] → serve /predict
                   ~14% ảnh → MinIO (raw-images/inference-samples/)
                          │
                   [Prometheus + Grafana] → monitor latency / error rate
                          │
                   [Drift Detector CronJob — mỗi giờ]
                   Cosine similarity: inference vs reference embeddings
                   ├── < 0.9 → ⚠️ DRIFT  ─┐
                   └── ≥ 0.9 → ✅ OK      ─┤→ Email → notifyEmail
                                            │
                   Move inference-samples/ → raw_images/ (chờ gắn nhãn)
                          │
                   Drift? → gắn nhãn mới → snapshot mới → push → CI/CD → retrain
```

---

## Quản Lý Version Data

| Tag | Ý nghĩa |
|-----|---------|
| `v0.0` | Project skeleton, chưa có data |
| `v1.0` | Dataset đầu tiên |
| `v2.0` | Dataset cập nhật lần 2 |

**Rollback về dataset cũ:**

```bash
git checkout v1.0 -- dataset_version.txt
git commit -m "rollback to dataset v1"
git push origin main
# → CI/CD tự trigger train lại với snapshot v1
```

---

## Dùng Template Cho Project Mới

```bash
git clone https://github.com/mnhut0550/mlops_1.git my_project
cd my_project

git remote remove origin
git remote add origin https://github.com/<you>/my_project.git

git checkout --orphan fresh
git add .
git commit -m "init from mlops template"
git push origin fresh:main

cp src/config/params_config_cls.yaml src/params.yaml
# Chỉnh task_name, models, credentials trong values.yaml

.\setup.ps1   # Windows
./setup.sh    # Linux/macOS
```

---

## Giới Hạn Demo

Đây là template demo cơ bản, thiết kế cho **một bài toán duy nhất** (hoặc Classification hoặc Detection) trong vòng đời của project.

**Chưa hỗ trợ đổi loại task** (ví dụ từ Classification sang Detection) sau khi đã setup và có data. Để đổi loại cần xóa toàn bộ data (MinIO + PostgreSQL) và setup lại từ đầu — chưa có script tự động cho việc này.

Nếu muốn thử loại khác, cách đơn giản nhất là dùng lại template cho một project mới riêng biệt.
