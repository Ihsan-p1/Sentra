# Sentra - AI Election Chatbot 🗳️

**Sentra** is an advanced **RAG-based (Retrieval Augmented Generation)** chatbot designed to analyze media framing in Indonesian political news. It uniquely employs a **Comparative Verification Framework**, contrasting Neural Machine Learning models against Statistical Baselines to ensure robust and explainable answers.

---

## 🚀 Key Features

### 1. RAG-Based Knowledge Engine
*   **Source:** 100+ Curated Articles (Jakarta Post, Tempo, Antara, Jakarta Globe).
*   **Engine:** PostgreSQL with `pgvector` for high-speed semantic retrieval.
*   **Gating:** Intelligent **Relevance Thresholding (<0.35)** prevents hallucinations on out-of-domain queries.

### 2. A/B Model Comparison System
Sentra evaluates every response using two competing approaches:

| Component | Model A (Neural / ML) 🧠 | Model B (Statistical / Baseline) 📊 |
|:---|:---|:---|
| **Framing Analysis** | **Fine-Tuned DistilBERT** (Style Classification) | **TF-IDF Keyword Extraction** (Frequency Analysis) |
| **Hallucination** | **Logistic Regression** (Embedding Features) | **N-gram Overlap** (Token Matching) |
| **Confidence** | **Random Forest** (Retrieval Density) | **Weighted Heuristic** (Simple Sum) |

### 3. Gemini 2.0 Flash Integration
*   Powered by Google's latest **Gemini 2.0 Flash** model for high-speed, context-aware generation.
*   Uses a strict **Markdown-structured prompt** to separate factual summary from interpretive analysis.

### 4. Academic-Grade Transparency
*   **Sidebar Analytics:** Real-time dashboard showing Model A vs Model B scores.
*   **Refined Terminology:** Distinguishes between "Hallucination Risk" and "Framing Inference".

---

## 🛠️ Tech Stack

*   **LLM:** Google Gemini 2.0 Flash (`google-generativeai`)
*   **Backend:** FastAPI, Uvicorn
*   **Database:** PostgreSQL (`pgvector` extension)
*   **NLP & ML:** PyTorch, Transformers (Hugging Face), Scikit-Learn
*   **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
*   **Frontend:** Vanilla JS + TailwindCSS (Glassmorphism UI)

---

## 📦 Installation

### 1. Clone & Setup
```bash
git clone https://github.com/Ihsan-p1/Sentra.git
cd Sentra
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install google-generativeai  # Required for Gemini
```

### 3. Environment Variables
Create a `.env` file based on `.env.example`:
```ini
DATABASE_URL=postgresql://user:pass@localhost:5432/Sentra1
GOOGLE_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-2.0-flash
```

### 4. Initialize Database
Scrape and ingest data into PostgreSQL:
```bash
python refresh_database.py
```

### 5. Train Models (Optional)
The repo comes with pre-trained models, but you can retrain them:
```bash
python scripts/train_framing_bert.py  # Fine-tune DistilBERT
python scripts/train_models.py        # Train Hallucination/Confidence models
```

---

## ▶️ Usage

### Start the Server
```bash
python -m uvicorn api.main:app --reload --port 8000
```
Open **http://localhost:8000** in your browser.

### Sample Questions
1.  *"How did the media frame Prabowo's cabinet selection?"* (Triggers Framing Analysis)
2.  *"What are the economic implications of the new policies?"*
3.  *"Who won the 2014 world cup?"* (Should trigger **Refusal Gating**)

---

## 📂 Project Structure

```
Sentra/
├── api/                # FastAPI routes
├── chatbot/            # Engine, LLM Client, Gating Logic
├── config/             # Settings & Env vars
├── data/               # Raw JSON data & Datasets
│   └── models/         # Saved ML models (BERT .bin, .pkl)
├── database/           # PostgreSQL connection & Schema
├── models/             # The Analysis Modules
│   ├── framing/        # DistilBERT & TF-IDF Analyzers
│   ├── hallucination/  # Logistic Regression Detector
│   └── confidence/     # Random Forest Scorer
├── rag/                # Retrieval (Embeddings + Pgvector)
├── scripts/            # Training & Scraper scripts
├── web/                # HTML/JS Frontend
└── requirements.txt    # Python deps
```

---

