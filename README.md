# Fake News Detection

> Hybrid Deep Learning System using BERT + GNN for Misinformation Detection

## Overview

A hybrid deep learning system for fake news detection that combines:

- **BERT** (Transformer-based NLP) - Deep semantic understanding of news text
- **GNN** (Graph Neural Network) - Captures relational and contextual patterns
- **Fusion Model** - Combines both representations for robust classification

The system includes a modern web interface and REST API, fully containerized with Docker.

---

## Architecture

```
Input News Text
       ↓
BERT Encoder (TensorFlow) → 768-dim embedding
       ↓
Graph Neural Network (PyTorch) → 128-dim embedding
       ↓
Fusion Classifier (768 + 128 + 64 = 960-dim)
       ↓
Fake/Real Probability
```

---

## Project Structure

```
fake-news-detection/
├── api/                    # FastAPI application
│   └── app.py              # API endpoints
├── data/                   # Dataset files
│   ├── processed/          # Processed data
│   └── raw/liar/           # LIAR dataset (train/test/valid)
├── explainability/         # Feature ablation analysis
├── fusion_model/           # Fusion classifier (PyTorch)
├── gnn_model/              # Graph Neural Network (PyTorch)
├── inference/              # Prediction pipeline
├── static/                 # Frontend (HTML/CSS/JS)
├── text_model/             # BERT encoder (TensorFlow)
├── training/               # Training scripts
├── user_model/             # User behavior MLP
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── run.sh                  # Linux/macOS run script
├── run.bat                 # Windows CMD run script
└── run.ps1                 # Windows PowerShell run script
```

---

## Quick Start

### Prerequisites

- Python 3.10.11
- pip (Python package manager)
- Git

---

### Step 1: Clone the Repository

```bash

cd Fake-News-Detection
```

---

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** This may take several minutes as it downloads BERT models and PyTorch libraries.

---

### Step 4: Run the Server

```bash
uvicorn api.app:app --reload
```

---

### Step 5: Open in Browser

- **Web Interface:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

---

## Alternative: Using Run Scripts

Instead of Steps 2-4, you can use the provided run scripts that automate everything:

**Windows (PowerShell):**
```powershell
.\run.ps1
```

**Windows (CMD):**
```cmd
run.bat
```

**Linux/macOS:**
```bash
chmod +x run.sh
./run.sh
```

---

## Alternative: Using Docker

```bash
# Build image
docker build -t fake-news-api .

# Run container
docker run -p 8000:8000 fake-news-api
```

---

## Usage

### Web Interface
Open http://localhost:8000 in your browser to access the frontend.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| GET | `/api/health` | Health check |
| POST | `/predict` | Analyze news text |
| GET | `/docs` | Swagger documentation |

### API Request Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists confirm drinking coffee cures all diseases instantly"}'
```

### API Response

```json
{
  "fake_probability": 0.82,
  "label": "Fake"
}
```

---

## Technologies

| Category | Technologies |
|----------|-------------|
| **ML/AI** | TensorFlow, PyTorch, HuggingFace Transformers, PyTorch Geometric |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Deployment** | Docker |
| **Data** | NumPy, Pandas, Scikit-learn |

---

## Model Details

| Component | Input | Output | Technology |
|-----------|-------|--------|------------|
| BERT Encoder | Text | 768-dim | TensorFlow + HuggingFace |
| GNN | 768-dim | 128-dim | PyTorch Geometric (GCNConv) |
| User Features | - | 64-dim | Placeholder (zeros) |
| Fusion | 960-dim | Probability | PyTorch (Linear + Sigmoid) |

---

## Dataset

Uses the **LIAR dataset** with 10,270 training samples:
- **Fake labels**: pants-fire, false, barely-true
- **Real labels**: half-true, mostly-true, true

---

## Notes

- The model predicts misinformation, not opinions
- Subjective/praise statements are typically classified as Real
- Sensational or false factual claims are more likely classified as Fake

---

## License

MIT License

