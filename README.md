📰 Fake News Detection using BERT 🧠 + GNN 🌐 (Hybrid Deep Learning)

📌 Project Overview

This project implements a hybrid deep learning system for Fake News Detection by combining:

🤖 BERT (Transformer-based NLP model) for deep semantic understanding of news text

🌐 Graph Neural Networks (GNN) for capturing relational and contextual patterns

🔗 Fusion Neural Network for final classification

The system is deployed as a FastAPI 🚀 web service and fully Dockerized 🐳, ensuring reproducibility across different machines.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🧠 Motivation

Fake news is not just about misleading text — it often spreads through networks, relationships, and context.
Traditional text-only models fail to capture this behavior.

This project addresses the problem by:

📝 Understanding what the news says using BERT

🧩 Modeling contextual / relational reasoning using GNN

🔀 Fusing both representations for a more robust prediction



------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
🏗️ System Architecture

📰 Input News Text
        ↓
🧠 BERT Encoder (TensorFlow) → 768-dim embedding
        ↓
🌐 Graph Neural Network (PyTorch) → 128-dim embedding
        ↓
🔗 Fusion Neural Network
        ↓
✅ Fake / Real Probability

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

⚙️ Technologies Used
🤖 Machine Learning & AI

TensorFlow (BERT Encoder)

PyTorch (GNN & Fusion Model)

HuggingFace Transformers

PyTorch Geometric

🌐 Backend & Deployment

FastAPI

Uvicorn

Docker

🛠️ Utilities

NumPy

Pandas

Scikit-learn

SHAP (Explainability)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 Running the Project (Recommended: Docker 🐳)
✅ Prerequisite

Docker Desktop installed and running
.Step 1: Clone the Repository

git clone https://github.com/vijai-ganeshh/fake-news-detection.git
cd fake-news-detection

.Step 2: Build the Docker Image
docker build -t fake-news-api .

.Step 3: Run the Docker Container
docker run -p 8000:8000 fake-news-api

.Step 4: Open FastAPI in Browser 🌍
http://localhost:8000/docs

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧪 Testing the /predict Endpoint
📥 Sample Request

{
  "text": "Government confirms aliens landed yesterday and signed a secret agreement"
}

📤 Sample Response

{
  "fake_probability": 0.82,
  "label": "Fake"
}


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧠 Important Notes on Predictions ⚠️

🗣️ The model predicts Fake News, not opinions

👍 Praise or subjective statements are usually classified as Real

🚨 Sensational or false factual claims are more likely to be classified as Fake

✅ This behavior is expected and correct.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📦 Reproducibility ♻️

The entire application is Dockerized, which guarantees:

Same Python version 🐍

Same library versions 📦

No dependency conflicts ❌

Any user can run this project using only Docker, without manually installing ML libraries.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🔮 Future Enhancements

🌍 Real-world graph construction from social media data

📰 Source credibility modeling

🌐 Multilingual fake news detection

☁️ Cloud deployment

🎨 Frontend web interface
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

