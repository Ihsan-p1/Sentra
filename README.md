# Sentra
Media Framing Analysis Chatbot

Sentra is a Retrieval-Augmented Generation (RAG) chatbot designed to analyze and compare media framing in Indonesian news. It uses a hybrid AI approach, combining Large Language Models (Google Gemini 2.0 Flash) with custom-trained machine learning models to provide grounded, fact-checked, and confidence-scored answers.

## Install

1.  Clone the repository and navigate to the project directory.

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Configure environment variables:
    Create a `.env` file in the root directory (see `.env.example`) and add your API keys:
    ```ini
    GOOGLE_API_KEY=your_gemini_api_key
    DATABASE_URL=postgresql://user:password@localhost/sentra_db
    ```

## Usage

### Starting the Application
Run the FastAPI server:
```bash
python -m uvicorn api.main:app --reload
```

The API will start at `http://localhost:8000`.
The web interface is served at the root URL `http://localhost:8000/`.

### Features
*   **Media Framing Analysis**: Compares how different media outlets frame political events using TF-IDF and DistilBERT.
*   **A/B Model Comparison**: Compares results between "Model A" (Custom ML) and "Model B" (Baseline/Heuristic) for hallucination detection and confidence scoring.
*   **Hallucination Detection**: Verifies generated answers against retrieved news chunks using Logistic Regression.
*   **Confidence Scoring**: Predicts answer reliability using a Random Forest model based on retrieval metrics.

## Architecture

The system uses a multi-stage pipeline:

1.  **Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` embeds queries and retrieves relevant news chunks (stored in PostgreSQL).
2.  **Generation**: Google Gemini 2.0 Flash generates the response and comparative analysis.
3.  **Evaluation (Model A)**:
    *   **Hallucination Detector**: Logistic Regression model trained on similarity features.
    *   **Confidence Scorer**: Random Forest Regressor trained on retrieval metrics.
    *   **Framing Analyzer**: DistilBERT fine-tuned for media style classification.
4.  **Evaluation (Model B - Baseline)**:
    *   Keyword Overlap for fact-checking.
    *   Heuristic scoring for confidence.
    *   TF-IDF for keyword framing.

> **Note**: The custom ML models (Model A) are currently trained on synthetic/proof-of-concept data. The architecture is designed to scale with larger, real-world datasets when available.

## Storage Layout

```text
Sentra/
├── api/                # FastAPI routes and server logic
├── chatbot/            # Core RAG engine and prompt management
├── data/               # Datasets and raw articles
│   └── models/         # Pickle files for custom trained models
├── database/           # Database connection and schema
├── models/             # ML model definitions (Framing, Confidence, Hallucination)
├── pipeline/           # Data ingestion and embedding pipeline
├── rag/                # Vector retrieval logic
├── scraper/            # News scrapers (ANTARA, Tempo, ABC News)
├── scripts/            # Training and evaluation scripts
├── web/                # Frontend static files (HTML/JS/CSS)
├── requirements.txt    # Python dependencies
└── setup_vector.py     # Script to initialize vector database
```

## Data Sources

The system currently supports the following news sources:
- **ANTARA News** (en.antaranews.com) - Indonesian national news agency
- **Tempo English** (en.tempo.co) - Indonesian investigative journalism
- **ABC News** (abc.net.au) - Australian international perspective

## Contributing

1.  Fork the project.
2.  Create your feature branch.
3.  Commit your changes.
4.  Push to the branch.
5.  Open a Pull Request.

## Status

MVP/Prototype. The project demonstrates a working proof-of-concept for media framing analysis in Indonesian political news. Training data is currently synthetic; the architecture supports expansion to larger datasets.
