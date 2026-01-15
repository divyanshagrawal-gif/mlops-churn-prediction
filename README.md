# MLOps Churn Prediction Pipeline

A production-ready MLOps pipeline for customer churn prediction, demonstrating industry best practices for data versioning, experiment tracking, feature engineering, model serving, and CI/CD.

## 🎯 Overview

This project implements a complete MLOps workflow that ensures:
- **Reproducibility**: Every model can be reproduced with the exact same dataset and code
- **Feature Consistency**: The same feature logic is used in training and inference, preventing training-serving skew
- **Experiment Tracking**: All experiments are logged with MLflow for comparison and model versioning
- **Data Versioning**: Processed datasets are versioned with DVC, tied to Git commits
- **Production Ready**: Containerized API service with input validation and feature alignment

## 🏗️ Architecture

### Technology Stack

- **Git** → Versions code
- **DVC** → Versions data (content-addressed storage)
- **MLflow** → Versions experiments & models
- **FastAPI** → Production API service
- **Docker** → Containerization
- **GitHub Actions** → CI/CD pipeline

### Project Structure

```
mlops-churn-prediction/
├── data/
│   ├── raw/                    # Raw data (not versioned)
│   │   └── churn.csv
│   ├── processed/              # Processed data (DVC versioned)
│   │   ├── train.csv.dvc      # DVC pointer file
│   │   └── train.csv           # Actual data (in DVC cache)
│   └── process_data.py         # Data preprocessing script
├── features/
│   ├── schema.py               # Feature definitions (single source of truth)
│   └── builder.py              # Feature engineering pipeline
├── training/
│   ├── config.yaml             # Training configuration
│   └── train.py                # Training pipeline with MLflow
├── serving/
│   ├── app.py                  # FastAPI application
│   ├── predictor.py            # Model prediction logic
│   └── schemas.py              # Pydantic request/response models
├── models/                     # Trained models and feature names
│   ├── model.joblib
│   └── feature_names.joblib    # Saved feature column names
├── infra/
│   └── docker/
│       └── Dockerfile          # Container definition
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── requirements.txt            # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Git
- DVC (installed via requirements.txt)
- Docker (for containerization)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlops-churn-prediction
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize DVC** (if not already initialized)
   ```bash
   dvc init
   ```

5. **Process raw data**
   ```bash
   python -m data.process_data
   ```

6. **Version processed data with DVC**
   ```bash
   dvc add data/processed/train.csv
   git add data/processed/train.csv.dvc .gitignore
   git commit -m "Add processed training data with DVC"
   ```

7. **Train the model**
   ```bash
   python -m training.train training/config.yaml
   ```

8. **Start the API server**
   ```bash
   uvicorn serving.app:app --reload
   ```

   The API will be available at `http://localhost:8000`

## 📊 Data Pipeline

### Data Processing

The `data/process_data.py` script performs minimal data cleaning:
- Converts column names to lowercase
- Maps churn labels: "Yes" → 1, "No" → 0
- Handles empty strings in numeric columns

**Run data processing:**
```bash
python -m data.process_data
```

### Data Versioning with DVC

**Why DVC?**
- Raw data is often too large for Git
- Processed data is what training actually uses
- DVC stores content-addressed blobs, indexed by hash
- Each dataset version is tied to a Git commit

**DVC Workflow:**

1. **Add data to DVC**
   ```bash
   dvc add data/processed/train.csv
   ```
   This creates `data/processed/train.csv.dvc` (a pointer file) and stores the actual CSV in DVC cache.

2. **Commit DVC pointer**
   ```bash
   git add data/processed/train.csv.dvc .gitignore
   git commit -m "Version processed training data with DVC"
   ```

3. **Reproducibility test**
   ```bash
   # Simulate data loss
   rm data/processed/train.csv
   
   # Restore from DVC
   dvc checkout
   ```

**Important Notes:**
- The `.dvc` folder contains metadata, not the cache
- The actual data is stored in DVC local cache
- If cache is deleted, data is lost even if `.csv.dvc` is in Git
- For production, configure a DVC remote (S3, Google Drive, etc.)

## 🔬 Feature Engineering

### Feature Schema

Features are explicitly defined in `features/schema.py` - this is the **single source of truth**:

**Numeric Features:**
- `tenure`
- `monthlycharges`
- `totalcharges`

**Categorical Features:**
- `contract`
- `paymentmethod`
- `internetservice`

### Feature Builder

The `features/builder.py` module:
- Handles missing values (numeric → 0, categorical → "unknown")
- Converts empty strings to NaN in numeric columns
- Applies one-hot encoding to categorical features
- **Aligns prediction features to match training features** (prevents feature mismatch errors)

**Key Feature:**
- During training: `build_features(df)` creates features based on data
- During prediction: `build_features(df, expected_columns=feature_names)` ensures all expected columns exist, filling missing ones with 0

This prevents the common error: *"Feature names seen at fit time, yet now missing"*

## 🎓 Training Pipeline

### Configuration

Training is config-driven via `training/config.yaml`:

```yaml
data:
  input_path: data/processed/train.csv
  target: churn

model:
  type: logistic_regression
  params:
    max_iter: 500

training:
  test_size: 0.2
  random_state: 42
```

### Training Process

The training pipeline (`training/train.py`):

1. Loads configuration
2. Reads processed data
3. Builds features using shared feature pipeline
4. Splits data (train/test)
5. Trains Logistic Regression model
6. Evaluates and logs metrics to MLflow
7. Saves model and feature names locally
8. Registers model in MLflow

**Run training:**
```bash
python -m training.train training/config.yaml
```

### MLflow Integration

- **Experiment Tracking**: All runs logged to `mlruns/` directory
- **Model Registry**: Models registered as "ChurnModel"
- **Reproducibility**: Each run tracks:
  - Git commit
  - DVC data hash
  - Model parameters
  - Metrics (accuracy)
  - Model artifacts

**View MLflow UI:**
```bash
mlflow ui
# Open http://localhost:5000
```

## 🚢 Model Serving

### FastAPI Application

The serving layer (`serving/app.py`) provides:
- Health check endpoint: `GET /health`
- Prediction endpoint: `POST /predict`

### Request Schema

```json
{
  "tenure": 12,
  "monthlycharges": 70.5,
  "totalcharges": 845.0,
  "contract": "Month-to-month",
  "paymentmethod": "Electronic check",
  "internetservice": "Fiber optic"
}
```

### Response Schema

```json
{
  "churn_probability": 0.234
}
```

### Feature Consistency

The `ChurnPredictor` class:
1. Loads the trained model
2. Loads expected feature names (saved during training)
3. Builds features using the same pipeline as training
4. Aligns features to match training-time columns
5. Returns churn probability

**Test the API:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "monthlycharges": 70.5,
    "totalcharges": 845.0,
    "contract": "Month-to-month",
    "paymentmethod": "Electronic check",
    "internetservice": "Fiber optic"
  }'
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t mlops-churn-api -f infra/docker/Dockerfile .
```

### Run Container

```bash
docker run -p 8000:8000 mlops-churn-api
```

The API will be available at `http://localhost:8000`

### Dockerfile Details

- Base image: `python:3.12-slim` (reduces image size)
- Working directory: `/app`
- Exposes port: `8000`
- Includes model, feature pipeline, and all dependencies

## 🔄 CI/CD Pipeline

### GitHub Actions

The CI pipeline (`.github/workflows/ci.yml`) runs on every push/PR:

1. ✅ Sets up Python 3.12
2. ✅ Installs dependencies
3. ✅ Runs training sanity check
4. ✅ Builds Docker image

**Current Limitation:**
The CI pipeline will fail on the training step because data is not fetched. To fix this, add a DVC pull step (see DVC Remote Setup below).

## 📦 DVC Remote Setup (Google Drive)

For production use, configure a DVC remote to store data in cloud storage.

### Setup Google Drive Remote

1. **Install Google Drive support**
   ```bash
   pip install "dvc[gdrive]"
   ```

2. **Create Google Service Account**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project
   - Enable Google Drive API
   - Create a Service Account
   - Download JSON key file (e.g., `gdrive-sa.json`)
   - ⚠️ **DO NOT COMMIT THIS FILE** - add to `.gitignore`

3. **Create folder in Google Drive**
   - Create a folder (e.g., `mlops-dvc-data`)
   - Right-click → Share
   - Share with: `<service-account-email>@<project>.iam.gserviceaccount.com`
   - Give **Editor** access

4. **Configure DVC remote**
   ```bash
   # Get folder ID from Drive URL: https://drive.google.com/drive/folders/<FOLDER_ID>
   dvc remote add -d gdrive gdrive://<FOLDER_ID>
   dvc remote modify gdrive gdrive_use_service_account true
   dvc remote modify gdrive gdrive_service_account_json_file_path gdrive-sa.json
   ```

5. **Push data to Google Drive**
   ```bash
   dvc push
   ```

6. **Update CI pipeline**
   Add to `.github/workflows/ci.yml`:
   ```yaml
   - name: Pull data with DVC
     env:
       AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
       AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
     run: |
       dvc pull
   ```

## 🔄 Reproducibility

### What Makes This Reproducible?

1. **Git** versions code and configuration
2. **DVC** versions processed datasets (content-addressed by hash)
3. **MLflow** tracks experiments with:
   - Model parameters
   - Metrics
   - Code version (Git commit)
   - Data version (DVC hash)

### Reproducing a Model

Every model in MLflow can be reproduced because:
- Code is tied to a Git commit
- Data is tied to a DVC hash
- Configuration is versioned in Git
- Feature pipeline is shared between training and serving

### Reproducibility Test

```bash
# Remove processed data
rm data/processed/train.csv

# Restore from DVC
dvc checkout

# Re-run training (should produce identical results)
python -m training.train training/config.yaml
```

## 📝 Key Principles

### Feature Store Thinking

- ✅ Features explicitly defined in `features/schema.py`
- ✅ No hidden columns
- ✅ One source of truth (Git-versioned)
- ✅ Reusable everywhere (training + serving)
- ✅ Prevents training-serving skew

### Model Lifecycle

1. **Data Processing** → Clean and prepare data
2. **Data Versioning** → Track with DVC
3. **Feature Engineering** → Shared pipeline
4. **Training** → MLflow tracks everything
5. **Serving** → Same feature pipeline ensures consistency
6. **Deployment** → Dockerized for portability

## 🛠️ Development

### Running Tests

```bash
# Training sanity check
python -m training.train training/config.yaml

# API health check
curl http://localhost:8000/health
```

### Project Dependencies

See `requirements.txt` for full list:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `pyyaml` - Configuration parsing
- `joblib` - Model serialization
- `mlflow` - Experiment tracking
- `dvc` - Data versioning
- `fastapi` - API framework
- `uvicorn` - ASGI server

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

