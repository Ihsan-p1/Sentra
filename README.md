# Sentra - AI Election Chatbot 🗳️

**Sentra** is an advanced RAG-based (Retrieval Augmented Generation) chatbot designed to answer questions about the Indonesian Election 2024 using trusted news sources (The Jakarta Post, Tempo, ANTARA, Jakarta Globe).

What sets Sentra apart is its **Dual-Model Evaluation System** which compares its own AI judgments (Model A) against baseline heuristics (Model B) in real-time.

![Sentra Dashboard Screenshot](https://via.placeholder.com/800x400?text=Sentra+AI+Dashboard+Placeholder)

---

## 🚀 Key Features

### 1. RAG-Based Knowledge
Answers are grounded in a curated database of 100+ news articles, minimizing hallucinations.

### 2. A/B Model Comparison System
Sentra doesn't just answer; it evaluates itself using two competing approaches for every response:

| Component | Model A (Machine Learning) 🧠 | Model B (Baseline) 📊 |
|:---|:---|:---|
| **Hallucination** | **Logistic Regression** (Trained on semantic features) | **Keyword Overlap** (Simple heuristic) |
| **Confidence** | **Random Forest** (Trained on retrieval metrics) | **Chunk Count** (Quantity heuristic) |
| **Framing** | **TF-IDF + NER** (Statistical significance) | **Word Count** (Frequency baseline) |

### 3. Real-Time Analytics Dashboard
The frontend displays a live comparison of how Model A and Model B evaluate the current answer, showing verify/unverified claims side-by-side.

---

## 🛠️ Tech Stack

*   **Backend**: Python, FastAPI
*   **AI/ML**: Scikit-Learn (Custom Models), ChromaDB (Vector Store), Ollama/OpenAI (LLM)
*   **Embedding**: Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **Frontend**: Vanilla JS + TailwindCSS
*   **Database**: SQLite (Metadata) + ChromaDB (Vectors)

---

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Ihsan-p1/Sentra.git
    cd Sentra
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup Environment Variables**
    Copy `.env.example` to `.env` and configure your LLM provider (Ollama or OpenAI).
    ```bash
    cp .env.example .env
    ```

5.  **Initialize Database**
    Run the ingestion script to populate ChromaDB with the news articles.
    ```bash
    python refresh_database.py
    ```

---

## ▶️ Usage

1.  **Start the Server**
    ```bash
    python -m uvicorn api.main:app --reload --port 8000
    ```

2.  **Open Web Interface**
    Go to `http://localhost:8000` in your browser.

3.  **Ask Questions**
    Try asking:
    *   *"What is Prabowo's free meal program?"*
    *   *"How are the cabinet appointments viewed by critics?"*
    *   *"Did aliens land in Jakarta?"* (To test hallucination detection)

---

## � Customizing the Knowledge Base

Want to use your own data? It's easy!

1.  **Modify the Data Source**
    Open `data/election_articles.py` and replace the content with your own articles.
    ```python
    ELECTION_ARTICLES = [
        {
            "title": "My Custom Article",
            "content": "This is the text that the bot will read...",
            "media_source": "my_source",
            "url": "https://example.com",
            "published_date": datetime(2024, 1, 1)
        },
        # Add more articles...
    ]
    ```

2.  **Refresh the Database**
    Run the script to clear the old database and ingest your new data:
    ```bash
    python refresh_database.py
    ```
    *This will rebuild the Vector Store (ChromaDB) with your new content.*

3.  **No Retraining Needed!** 🚀
    The AI models (Hallucination & Confidence) automatically adapt to your new data without any code changes.

---

```bash
python scripts/evaluate_models.py
```

*See `evaluation_report.md` for detailed benchmarks.*

---

## 📂 Project Structure

```
Sentra/
├── api/                # FastAPI routes and main app
├── chatbot/            # Core RAG logic and engine
├── config/             # Configuration settings
├── data/               # Raw news data and synthetic datasets
├── database/           # DB connection and schema
├── models/             # The A/B Models
│   ├── baseline/       # Model B (Heuristics)
│   ├── confidence/     # Model A (Random Forest)
│   ├── framing/        # Model A (TF-IDF)
│   └── hallucination/  # Model A (Logistic Regression)
├── pipeline/           # Ingestion and embedding pipeline
├── rag/                # Retrieval logic
├── scripts/            # Training and evaluation scripts
├── web/                # Frontend (HTML/JS)
└── requirements.txt    # Dependencies
```

---

**Author**: Ihsan  
**Project**: College Project - AI Sem 5