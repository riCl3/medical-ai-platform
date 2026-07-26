# MediScan AI — Medical AI Diagnostic Platform

> Chest X-ray analysis platform powered by deep learning, explainable AI (Grad-CAM), and large language models for automated radiological reporting.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.60-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Live Demo:** [mediscan-ai.streamlit.app](https://mediscan-ai.streamlit.app)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technical Deep Dive](#technical-deep-dive)
  - [Training Pipeline](#training-pipeline)
  - [Inference Pipeline](#inference-pipeline)
  - [Explainability (Grad-CAM)](#explainability-grad-cam)
  - [LLM Report Generation](#llm-report-generation)
  - [Database Schema](#database-schema)
- [Deployment](#deployment)
- [Local Development](#local-development)
- [API Reference](#api-reference)
- [Testing](#testing)

---

## Overview

MediScan AI is an end-to-end medical imaging platform that:

1. **Classifies** chest X-rays as NORMAL or PNEUMONIA using a fine-tuned ResNet18 model
2. **Explains** predictions via Grad-CAM heatmaps showing which regions influenced the decision
3. **Generates** structured clinical reports using Llama 3.3 (70B) via Groq API
4. **Stores** prediction history in a SQLite database for review and audit

The platform is designed as a single-process Streamlit application suitable for free deployment on Streamlit Cloud.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                           │
│                  (app/frontend/app.py — port 8501)                  │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────┐  │
│  │  Upload   │──▶│  Grad-CAM    │──▶│  LLM Report │──▶│  SQLite  │  │
│  │  Image    │   │  Inference   │   │  Generator  │   │  Database│  │
│  └──────────┘   └──────────────┘   └─────────────┘   └──────────┘  │
│                        │                   │               │         │
│                        ▼                   ▼               ▼         │
│               ┌──────────────┐   ┌─────────────┐   ┌──────────┐     │
│               │  ResNet18    │   │  Groq API   │   │  SQLite  │     │
│               │  model.pth   │   │  Llama 3.3  │   │  .db     │     │
│               └──────────────┘   └─────────────┘   └──────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow:**

```
User uploads X-ray
       │
       ▼
┌─────────────────┐
│  Preprocessing  │  Resize 224×224, ImageNet normalization
└────────┬────────┘
         ▼
┌─────────────────┐
│  ResNet18       │  Forward pass → sigmoid → P(NORMAL) vs P(PNEUMONIA)
│  Forward Pass   │
└────────┬────────┘
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Grad-CAM       │────▶│  Heatmap Overlay │  Base64-encoded PNG
│  Generation     │     └─────────────────┘
└────────┬────────┘
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Groq API       │────▶│  Clinical Report │  Findings / Impression / Recommendation
│  (Llama 3.3)    │     └─────────────────┘
└────────┬────────┘
         ▼
┌─────────────────┐
│  SQLite DB      │  Store prediction + base64 heatmap + report
└─────────────────┘
```

---

## Project Structure

```
medical-ai-platform/
├── app/
│   ├── api/                    # FastAPI backend (used for local multi-service mode)
│   │   └── main.py             # REST endpoints: /predict, /history, /health
│   ├── db/
│   │   ├── database.py         # SQLAlchemy engine, session, CRUD operations
│   │   └── models.py           # Prediction ORM model
│   ├── frontend/
│   │   └── app.py              # Streamlit app (main entrypoint for deployment)
│   ├── llm/
│   │   └── report_generator.py # Groq LLM integration
│   └── xai/
│       └── gradcam.py          # Grad-CAM inference module
├── models/
│   └── model.pth               # Trained ResNet18 weights (42.7 MB)
├── training/
│   └── train_pneumonia.ipynb   # Colab training notebook
├── tests/
│   ├── test_database.py        # DB smoke test
│   ├── test_gradcam.py         # Grad-CAM inference test
│   └── test_report_generator.py# LLM report test
├── .streamlit/
│   └── config.toml             # Streamlit dark theme + server config
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Model Performance

Trained on [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset (5,856 images).

### Test Set Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 92.95% |
| Precision | 93.03% |
| Recall    | 95.90% |
| F1 Score  | 94.44% |

### Confusion Matrix

```
                Predicted
              NORMAL  PNEUMONIA
Actual NORMAL  [ 206     28  ]
      PNEUMONIA [  16    374  ]
```

### Training Configuration

| Parameter       | Value                    |
|-----------------|--------------------------|
| Architecture    | ResNet18 (ImageNet pretrained) |
| Final Layer     | `Linear(512, 1)` + Sigmoid |
| Loss Function   | BCEWithLogitsLoss        |
| Optimizer       | Adam (lr=1e-4)           |
| Batch Size      | 32                       |
| Epochs          | 8                        |
| Image Size      | 224 × 224                |
| Hardware        | Google Colab T4 GPU      |

---

## Technical Deep Dive

### Training Pipeline

The training notebook (`training/train_pneumonia.ipynb`) implements:

1. **Data augmentation** — Random horizontal flip, random rotation (15°), ImageNet normalization
2. **Transfer learning** — ResNet18 pretrained on ImageNet, final FC layer replaced for binary output
3. **Training loop** — 8 epochs with train/val loss and accuracy tracking
4. **Evaluation** — Test set metrics (accuracy, precision, recall, F1) and confusion matrix
5. **Model export** — Saves `model.pth` state dict

### Inference Pipeline

```python
# app/xai/gradcam.py
def run_gradcam(image_path: str, device: str = "cpu") -> Tuple[str, float, PIL.Image]:
    # 1. Load cached ResNet18 model
    model, target_layer = _load_model(device)

    # 2. Preprocess: resize, normalize
    input_tensor = transform(Image.open(image_path))

    # 3. Forward pass → sigmoid probability
    logit = model(input_tensor)
    prob = torch.sigmoid(logit).item()

    # 4. Generate Grad-CAM heatmap
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor, targets=[BinaryClassifierOutputTarget(target_class)])[0]

    # 5. Overlay heatmap on original image
    overlay = show_cam_on_image(img_np, grayscale_cam)

    return predicted_class, confidence, overlay
```

### Explainability (Grad-CAM)

**Grad-CAM** (Gradient-weighted Class Activation Mapping) produces a heatmap highlighting the image regions most influential to the model's prediction.

- **Target layer:** `model.layer4[-1].conv2` — the last convolutional layer of ResNet18
- **Output:** A 2D heatmap overlaid on the original X-ray, showing which anatomical regions (e.g., lung fields, opacities) drove the classification
- **Library:** [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

### LLM Report Generation

Uses **Llama 3.3 70B** via the Groq API to generate structured clinical reports:

**Prompt structure:**
```
System: You are a medical report assistant. Given a chest X-ray prediction,
        write a structured report with: Findings, Impression, Recommendation.
        Always end with a medical disclaimer.

User:   Prediction: PNEUMONIA
        Model confidence: 93.00%
        Generate the structured report.
```

**Output format:**
```markdown
## Findings
The chest X-ray demonstrates bilateral opacities...

## Impression
The findings are consistent with pneumonia.

## Recommendation
1. Clinical correlation recommended
2. Follow-up imaging in 2-4 weeks
3. Consider laboratory workup

This is an AI-generated report for educational/demonstration purposes only
and is not a substitute for professional medical diagnosis.
```

### Database Schema

```sql
CREATE TABLE predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_filename      VARCHAR(512) NOT NULL,
    predicted_class     VARCHAR(50) NOT NULL,
    confidence          FLOAT NOT NULL,
    gradcam_image_path  VARCHAR(1024),
    gradcam_blob        TEXT,          -- base64-encoded heatmap PNG
    llm_report          TEXT
);
```

The `gradcam_blob` column stores the Grad-CAM heatmap as a base64-encoded PNG directly in the database, ensuring persistence across application restarts on ephemeral file systems (e.g., Streamlit Cloud).

---

## Deployment

### Streamlit Cloud (Recommended — Free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo, branch `main`, script path `app/frontend/app.py`
4. Advanced Settings → Secrets:
   ```
   GROQ_API_KEY = "gsk_..."
   ```
5. Deploy

The app auto-redeploys on every push to `main`.

### Local Multi-Service Mode (Development)

```bash
# Terminal 1 — FastAPI backend
uvicorn app.api.main:app --reload --port 8000

# Terminal 2 — Streamlit frontend
streamlit run app/frontend/app.py
```

---

## Local Development

### Prerequisites

- Python 3.10+
- 4 GB+ RAM (for PyTorch)
- Groq API key ([get one here](https://console.groq.com))

### Setup

```bash
# Clone
git clone https://github.com/riCl3/medical-ai-platform.git
cd medical-ai-platform

# Virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run
streamlit run app/frontend/app.py
```

### Dependencies

| Package                | Purpose                          |
|------------------------|----------------------------------|
| `torch`                | Deep learning framework          |
| `torchvision`          | Image transforms, model zoo      |
| `grad-cam`             | Grad-CAM explainability          |
| `streamlit`            | Web UI framework                 |
| `sqlalchemy`           | ORM and database management      |
| `groq`                 | LLM API client (Llama 3.3)       |
| `python-dotenv`        | Environment variable loading     |
| `opencv-python-headless` | Image processing (Grad-CAM dep) |
| `pillow`               | Image I/O                        |
| `numpy`                | Numerical operations             |
| `pandas`               | Data manipulation                |
| `scikit-learn`         | Metrics (precision, recall, F1)  |
| `matplotlib`           | Plotting (training notebook)     |

---

## API Reference

> **Note:** The FastAPI backend is used in local multi-service mode. The Streamlit app calls functions directly.

| Method | Endpoint             | Description                                      |
|--------|----------------------|--------------------------------------------------|
| `GET`  | `/health`            | Health check                                     |
| `POST` | `/predict`           | Upload image → inference + report + save to DB   |
| `GET`  | `/history`           | List all past predictions                        |
| `GET`  | `/history/{id}`      | Get single prediction with base64 Grad-CAM       |
| `GET`  | `/gradcam/{filename}`| Serve Grad-CAM PNG file                          |

### POST /predict

**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "prediction_id": 1,
  "predicted_class": "PNEUMONIA",
  "confidence": 0.9342,
  "gradcam_image": "<base64-encoded PNG>",
  "llm_report": "## Findings\n..."
}
```

---

## Testing

```bash
# Test Grad-CAM inference
python tests/test_gradcam.py

# Test database operations
python tests/test_database.py

# Test LLM report generation
python tests/test_report_generator.py
```

Each test is a standalone script (not pytest-based). Place sample X-ray images in `sample_data/` for Grad-CAM testing.

---

## Disclaimer

This platform is for **educational and demonstration purposes only**. It is not intended for clinical use and should not be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider.

---

## License

MIT
