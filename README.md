# Sentra
Media Framing Analysis Chatbot

![tests](https://github.com/Ihsan-p1/Sentra/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10-blue)
![license](https://img.shields.io/badge/status-prototype-orange)

Sentra is a Retrieval-Augmented Generation (RAG) chatbot that analyzes and compares media framing in English-language Indonesian news. It runs entirely on local infrastructure: a 3B parameter LLM served by Ollama, local sentence-transformer embeddings, and PostgreSQL. Nothing calls an external API at inference time.

Every answer is evaluated twice: once by custom-trained models ("Model A") and once by rule-based baselines ("Model B"). Both results are shown side by side. That A/B comparison is the point of the project: it makes the cost of a simple heuristic visible instead of assuming the trained model is better.

## Pipeline

```mermaid
flowchart LR
    Q[User query] --> L{Small talk?}
    L -->|greeting| SMALL[Direct reply<br/>no retrieval]
    L -->|question| R[Retrieval<br/>MiniLM + cosine]
    R -->|max sim &lt; 0.35| REF[Refuse to answer]
    R --> G[Generation<br/>qwen2.5:3b via Ollama]
    R --> F[Framing<br/>DistilBERT / TF-IDF]
    G --> H[Hallucination check<br/>LogReg / keyword overlap]
    H --> C[Confidence<br/>RandomForest / heuristic]
    F --> OUT[Answer + A/B panel]
    C --> OUT
```

The retrieval gate matters: when the best-matching source scores below threshold, Sentra refuses rather than generating an unsupported answer.

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

4.  Install [Ollama](https://ollama.com), then build the serving model:
    ```bash
    ollama pull qwen2.5:3b
    ollama create sentra-qwen:3b -f ollama/Modelfile
    ```

    `qwen2.5:3b` is a ~1.9 GB multilingual model that runs on CPU. The Modelfile
    step is not optional: Ollama defaults to a 2048-token context window, and
    Sentra's prompts (system template + chunks from three outlets + history)
    exceed that. Ollama truncates silently, which looks like a retrieval bug.
    `ollama/Modelfile` bakes `num_ctx 8192` into the model so the fix survives
    however the server is started.

    To use a different model, edit `ollama/Modelfile` and `LLM_MODEL` in `.env`.

5.  Configure environment variables:
    Copy `.env.example` to `.env` and adjust:
    ```ini
    DATABASE_URL=postgresql://user:password@localhost:5432/Sentra1
    LLM_BASE_URL=http://localhost:11434/v1
    LLM_MODEL=llama3.1
    ```
    No API key is required for a local endpoint. To use a hosted provider instead,
    point `LLM_BASE_URL` at it (e.g. `https://openrouter.ai/api/v1`) and set
    `OPENROUTER_API_KEY`.

6.  Populate the database and train the models (the trained artifacts are
    gitignored, so a fresh clone must regenerate them):
    ```bash
    python scripts/tools/refresh_database.py   # ingest the article corpus
    python scripts/train_models.py             # Model A: hallucination + confidence
    python scripts/train_framing_bert.py       # Model A: DistilBERT framing
    ```

## Usage

### Starting the application
Run the FastAPI server:
```bash
python -m uvicorn api.main:app --reload
```

The API starts at `http://localhost:8000`, and the web interface is served at the root URL.

### Verifying the setup
```bash
python scripts/debug/verify_ollama.py        # is the model server reachable?
python scripts/debug/verify_model_direct.py  # does the configured model answer?
```

### Testing
```bash
pytest                    # 449 unit tests (offline; no server, database or model weights)
pytest -m integration     # needs a database; see the warning below
python scripts/uat.py     # 36-scenario user acceptance test against a live server
```

Three layers, each catching what the layer below cannot:

| Layer | Count | Needs | Catches |
|---|---|---|---|
| Unit | 449 | nothing | logic, copy, config invariants, guard behaviour |
| Integration (schema) | 14 | PostgreSQL | that `schema.sql` executes and provenance columns populate |
| Integration (API) | 10 | running server | that the endpoints answer over HTTP |
| UAT | 36 | server + Ollama | that the product behaves for a user end to end |

The UAT exercises the product the way a user does: greetings, Indonesian and
English questions, in-corpus and out-of-corpus queries, strict mode, multi-turn
memory, and latency, each with explicit pass criteria. It has earned its
keep: it found the all-greetings-detected-as-English bug, the Indonesian-body /
English-headings bug, the `"app"` substring that hijacked *"cabinet
appointments"*, and one scenario of my own that was simply wrong. Latest run:
36/36 passed ([report](evaluation_results/uat_report.json)).

> Warning: `tests/test_schema_integration.py` truncates the article tables. It
> refuses to run unless the database name contains `test`, `_ci`, `ci_` or
> `scratch`; point it somewhere disposable with `SENTRA_TEST_DATABASE_URL`.
> Running it unguarded emptied a freshly embedded 154-article corpus in under a
> second, which is why the guard exists and is itself tested.

CI runs the unit tests, the schema/provenance integration job against a real
PostgreSQL service, and a hardcoded-credential scan on every push
(`.github/workflows/tests.yml`).

### Features
*   Media Framing Analysis: Compares how different media outlets frame political events using TF-IDF and DistilBERT.
*   A/B Model Comparison: Compares "Model A" (Custom ML) against "Model B" (Baseline/Heuristic) for hallucination detection, confidence scoring, and framing.
*   Hallucination Detection: Verifies generated answers against retrieved news chunks using Logistic Regression over similarity features.
*   Confidence Scoring: Predicts answer reliability using a Random Forest model based on retrieval metrics.
*   Retrieval Gating: Refuses to answer when the best-matching source falls below a calibrated similarity threshold, rather than generating an unsupported answer.
*   Bilingual: Ask in Indonesian, get an Indonesian answer (headings, disclaimers, and refusals included) from an English corpus. Ask in English and get English.
*   Small-talk routing: Greetings are answered directly instead of being pushed through retrieval and refused.
*   Self-explaining: Asking *"apa fungsi dari Analysis Results?"* or *"what features does this app have?"* returns a plain-language explanation of the panel or feature, no jargon, no retrieval.

## Architecture

The system uses a multi-stage pipeline:

0.  Intent routing: greetings and thanks are answered directly; only real questions reach retrieval.
1.  Retrieval: `paraphrase-multilingual-MiniLM-L12-v2` embeds the query; chunks are ranked by cosine similarity computed in NumPy. Articles and their embeddings live in PostgreSQL as `FLOAT8[]`; pgvector is not used. The query is prefixed before embedding (see *Query expansion* below).
2.  Generation: a local LLM served by Ollama (default `sentra-qwen:3b`) generates the response and comparative analysis over an OpenAI-compatible HTTP API. The prompt declares the retrieved outlet names as a closed list, because small models otherwise abbreviate or invent them (observed: `ANTARANEWS` → `ANTV`).
3.  Evaluation (Model A):
    *   Hallucination Detector: Logistic Regression on 5 similarity/overlap features.
    *   Confidence Scorer: Random Forest Regressor trained on retrieval metrics.
    *   Framing Analyzer: DistilBERT fine-tuned for media style classification.
4.  Evaluation (Model B - Baseline):
    *   Keyword Overlap for fact-checking.
    *   Deterministic heuristic scoring for confidence.
    *   Word-frequency / TF-IDF for keyword framing.

> Note: The custom ML models (Model A) are trained on synthetic/proof-of-concept data. The architecture is designed to scale with larger, real-world datasets when available.

## Results

Generated by `python scripts/evaluate_models.py`; full report in
[`evaluation_results/latest_evaluation.md`](evaluation_results/latest_evaluation.md).

Hallucination detection (16 labelled claim/source pairs)

| Metric | Model A (Logistic Regression) | Model B (keyword overlap) |
|---|---|---|
| Accuracy | **75.00%** | 56.25% |
| F1 | **80.00%** | 69.57% |

Confidence scoring

| Metric | Model A (Random Forest) | Model B (heuristic) |
|---|---|---|
| In-range accuracy | **88.89%** | 44.44% |
| Mean absolute error | **0.089** | 0.180 |

Framing keyword extraction (top-15 keywords per outlet, no manual labels)

| Metric | TF-IDF | Word frequency |
|---|---|---|
| Distinctiveness (mean pairwise Jaccard, lower is better) | **17.92%** | 26.81% |
| Generic-term rate (share also in corpus top-50, lower is better) | **76.67%** | 86.67% |
| Source attribution accuracy | 51.35% | 45.95% |
| *Majority-class baseline* | *59.46%* | *59.46%* |

The framing row is a negative result, reported as one. TF-IDF produces
noticeably more distinctive and less generic keyword sets than raw counts, but
*neither extractor beats the majority-class baseline* at identifying which outlet
wrote a held-out article. Keyword frequency alone does not carry outlet identity.
The fine-tuned DistilBERT classifier reaches 79% accuracy on that same task
(`scripts/train_framing_bert.py`), which is the argument for using it.

> Caveat: the DistilBERT figure comes from its own stratified split over
> title/content segments, not the article-level split used for the keyword
> metrics, so treat it as indicative rather than a strict head-to-head.

Model A's hallucination detector and confidence scorer are trained on synthetic
data. Their numbers demonstrate that the architecture works end to end; they are
not a benchmark against real annotations.

### A fix that measured well, and was wrong anyway

Indonesian questions were being refused. Similarity scores across the whole
corpus looked compressed, so a `"News about: "` prefix was added to the query
before embedding, on the theory that a paraphrase model wants a sentence rather
than a bare phrase. It measured as a large improvement: distribution separation
went from −0.022 to +0.039, and three false refusals became zero.

That measurement was worthless. The real fault was that every stored embedding
had been produced by a different model than the query embeddings.
`EMBEDDING_MODEL` had been switched from an English-only model to a multilingual
one without re-ingesting. Both output 384 dimensions, so nothing crashed and no
error was ever logged: retrieval simply compared vectors across two unrelated
embedding spaces. A chunk whose true similarity was 0.606 scored 0.350.

Re-embedding the corpus fixed the root cause. With correct vectors, the prefix
gave identical error rates and slightly worse separation, so it was removed.
The threshold calibrated against the broken state (0.28) turned out to admit
5 of 8 deliberately unanswerable queries once retrieval worked, and was raised
to 0.35.

> A fix validated against a broken baseline measures the breakage, not the fix.

Three things came out of it, and they are the reason this section exists:

*   `article_chunks.embedding_model` now records which model produced every
    vector, and `api/main.py` refuses to start quietly on a mismatch.
*   Every request logs one `[TRACE]` line with the similarity spread
    (`max/median/min`). All three scores sliding down together is the signature
    of an embedding mismatch; a low max alone is just a hard query. Diagnosing
    this without that line took five throwaway scripts.
*   `print()` logging was reconfigured to line-buffered. Piped to a file, the
    startup diagnostics were sitting unflushed in an 8 KB buffer, so the guard was
    running and reporting nothing.

### When the answer is a data problem, not a code problem

After retrieval was fixed, `siapa pemenang pemilu 2024?` ("who won the 2024
election?") was still refused. It was tempting to read that as a cross-lingual
retrieval weakness and lower the threshold again.

It was neither. The corpus consisted of *aftermath* coverage and named the winner
exactly once, in passing, inside a single quick-count article. One incidental
mention cannot be ranked above 280 other chunks, and no threshold, prompt rule or
reranker changes that. The most basic question in the domain was unanswerable
because the answer was barely in the data.

The fix was five articles reporting the official result: the KPU declaration,
the vote counts, and the inauguration, sourced from ANTARA News and Tempo with URLs
checked against the live sites. Retrieval now scores 0.551 on the Indonesian
phrasing and the answer names Prabowo Subianto with the official 58.59% share.

`tests/test_corpus.py` asserts the fix cannot silently regress, including that at
least one article *opens* by naming the winner. Chunking is positional, so a
fact buried at the end of a long article lands in a later chunk and competes with
the whole corpus again.

The same test file caught two defects that had been in the corpus from the start:
two articles duplicated another's URL (the same source ingested twice, which
double-counted one outlet's vocabulary in the framing comparison), and 67
typographic characters made `Indonesia's` and `quick-count` different tokens from
their ASCII spellings, silently splitting TF-IDF term counts.

### Reproducibility

All stochastic components use fixed seeds (`LLM_SEED`, `random_state=42`, `torch.manual_seed`, seeded train/test splits), and the Model B baselines contain no randomness, so an A/B evaluation run is repeatable given the same corpus.

"The same corpus" is checkable rather than assumed: `data/corpus.py` fingerprints
the article list (`indonesia-election-2024@cecce290d0ff`), ingestion stores that
fingerprint on every article row, and startup reports which version the database
holds, with a warning if a partial re-ingest left more than one.

### Honest failure states

Three places were reporting success they had not earned. Each is now capable of
saying "no":

*   `/api/health` returned a literal `{"status": "ok", "models": "loaded"}`:
    healthy before initialisation had run, with the database unreachable, and with
    Model A missing. It now reports `ok` or `degraded` with the failing checks
    named, plus the corpus version, chunk count and live session counters.
*   The A/B sidebar rendered `0/0` as green "Strong Alignment", claiming a
    verification result that had never been computed, and left the previous
    answer's scores on screen through a refusal. Both now read "Not evaluated".
*   A dead model server produced `"Connection Error: could not reach
    localhost:11434"` as the answer, then split it into sentences, scored it with the
    hallucination detector, given a confidence percentage, and written into
    conversation memory. It now raises `LLMUnavailable`, returns the retrieved
    sources with an empty comparison, and is never remembered.

### Security

*   `POST /api/ingest` writes into the corpus the chatbot cites. Set `INGEST_API_KEY` in `.env` and send it as `X-API-Key` to gate it; unset, the endpoint stays open for local development and startup logs a warning.
*   CORS is restricted to localhost origins by default (`CORS_ALLOW_ORIGINS`).
*   No secrets in source: CI fails the build if an API key pattern appears in tracked files.
*   Error responses no longer echo `str(e)` to the client; tracebacks go to the server log.
*   The destructive half of the integration suite refuses to run against a database whose name is not marked disposable. It emptied the development corpus once.

### Known limits

*   Retrieval loads every chunk for a media source and scores it in Python on each request. That is fine for the 154-article corpus here; it is not a design for large collections.
*   The corpus is a single news event (the 2024 election and its aftermath), so framing differences between outlets are narrow by construction.
*   Conversation memory is in-process: restarting the server clears every session.
*   The in-corpus and out-of-corpus similarity distributions overlap by 0.014. Two valid queries score 0.329 and are refused at the calibrated 0.35 threshold; lowering it to admit them leaks 3 of 8 unanswerable queries. No threshold satisfies both, and the UAT does not paper over it.
*   Model A's hallucination detector flags most of the *Comparative Framing Analysis* section as unsupported. That is correct, since the prompt asks the model to infer tone and emphasis, which is not literally in the sources. It does mean the headline support rate mixes factual claims with interpretive ones and reads lower than the factual grounding alone would.
*   `data/election_articles.json` stores article text ABC News could not be re-verified against: abc.net.au blocks automated access, so the five added result articles come from ANTARA News and Tempo only.

## Storage layout

```text
Sentra/
├── api/                # FastAPI routes and server logic
├── chatbot/            # Core RAG engine, LLM client, prompts, conversation memory
├── config/             # Pydantic settings loaded from .env
├── data/               # Article corpus (JSON) + versioning loader
├── database/           # Connection manager and schema
├── evaluation_results/ # Generated A/B evaluation report (committed)
├── ollama/             # Modelfile for the serving model
├── models/             # ML model definitions + trained artifacts (gitignored)
│   ├── baseline/       # Model B (rule-based)
│   ├── confidence/     # Model A confidence scorer
│   ├── framing/        # TF-IDF + DistilBERT analyzers, evaluation metrics
│   ├── hallucination/  # Model A hallucination detector
│   └── framing_bert/   # Fine-tuned DistilBERT weights (generated)
├── pipeline/           # Preprocessing, embeddings, ingestion
├── rag/                # Vector retrieval logic
├── scraper/            # News scrapers (ANTARA, Tempo, ABC News)
├── scripts/            # Training, evaluation, and debug utilities
├── tests/              # pytest suite (unit + integration)
├── utils/              # Per-request tracing
├── web/                # Frontend static files (HTML/JS/CSS)
├── requirements.txt    # Pinned Python dependencies
└── .env.example        # Environment template
```

## Data sources

The corpus covers the 2024 Indonesian presidential election and its aftermath:
154 English-language articles in `data/election_articles.json`:

| Source | Articles | Perspective |
|---|---|---|
| ANTARA News (en.antaranews.com) | 89 | Indonesian national news agency |
| Tempo English (en.tempo.co) | 34 | Indonesian investigative journalism |
| ABC News (abc.net.au) | 31 | Australian international perspective |

Rebuild the database from it with `python scripts/tools/refresh_database.py`,
which re-embeds every chunk and verifies the vectors it wrote.

Source identifiers (`antaranews`, `tempo`, `abc_news`) are defined once in
`settings.SUPPORTED_MEDIA` and reused by ingestion, retrieval, and training.

## Contributing

1.  Fork the project.
2.  Create your feature branch.
3.  Commit your changes (`pytest` must pass).
4.  Push to the branch.
5.  Open a Pull Request.

## Status

Prototype. The project demonstrates a working proof-of-concept for media framing analysis in Indonesian political news. Training data is currently synthetic; the architecture supports expansion to larger datasets.
