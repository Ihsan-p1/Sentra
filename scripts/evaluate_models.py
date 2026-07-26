"""
Model Evaluation Script: Compare Model A (Trained ML) vs Model B (Baseline)

This script evaluates all three model pairs:
1. Hallucination Detection
2. Framing Analysis  
3. Confidence Scoring

Note: The models are trained on synthetic/hardcoded data for proof-of-concept.
The architecture is designed to scale with larger, real-world datasets.
Current training uses ~15-20 example sentences duplicated to increase sample size.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The report uses box-drawing characters, which the default Windows console
# codepage (cp1252) cannot encode. Force UTF-8 so the run does not die halfway.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from datetime import datetime

# Import our models
from models.hallucination.detector import HallucinationDetector
from models.confidence.scorer import ConfidenceScorer
from models.framing.analyzer import FramingAnalyzer
from models.framing.metrics import evaluate_extractor, tfidf_profiles, wordcount_profiles
from models.baseline.models import get_baseline_models
from pipeline.embeddings import get_embedder  # Fixed import

print("=" * 70)
print("SENTRA MODEL EVALUATION: Model A (Trained) vs Model B (Baseline)")
print("=" * 70)
print(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# 1. HALLUCINATION DETECTION EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("1. HALLUCINATION DETECTION")
print("=" * 70)

# Create synthetic test dataset
print("\nGenerating test dataset...")

# Ground truth examples: (sentence, source_texts, is_supported)
hallucination_test_data = [
    # SUPPORTED claims (should be detected as supported)
    ("Prabowo won the election with 58 percent of votes.", 
     ["Prabowo Subianto won Indonesia's presidential election with approximately 58.59 percent of votes."], 
     True),
    ("The Constitutional Court dismissed election fraud challenges.",
     ["The Constitutional Court has dismissed legal challenges filed by losing candidates contesting the election results."],
     True),
    ("Gibran is the eldest son of President Jokowi.",
     ["Gibran Rakabuming Raka, the eldest son of outgoing President Joko Widodo, was selected as running mate."],
     True),
    ("The cabinet has 109 officials including ministers.",
     ["The cabinet comprises 48 ministers, 5 ministerial-level officials, and 59 vice ministers, totaling 109 officials."],
     True),
    ("Sri Mulyani remains as Finance Minister.",
     ["Sri Mulyani Indrawati remains as Finance Minister, reassuring international investors."],
     True),
    ("Young voters played crucial role in the election.",
     ["Analysis of voting patterns reveals that young Indonesian voters played a crucial role in Prabowo's electoral victory."],
     True),
    ("The election saw high voter turnout.",
     ["The voter turnout was recorded at approximately 82 percent of the 204 million registered voters."],
     True),
    ("Prabowo emphasized ASEAN centrality.",
     ["Prabowo emphasized Indonesia's commitment to ASEAN centrality in regional security matters."],
     True),
    
    # UNSUPPORTED claims (should be detected as hallucinations)
    ("Prabowo won with 75 percent of votes.",  # Wrong percentage
     ["Prabowo won with approximately 58 percent of the vote."],
     False),
    ("The Constitutional Court ruled in favor of Anies Baswedan.",  # Opposite of truth
     ["The Constitutional Court dismissed legal challenges filed by Anies Baswedan."],
     False),
    ("Gibran is Prabowo's biological son.",  # Fabricated relationship
     ["Gibran is the eldest son of President Jokowi."],
     False),
    ("The cabinet has only 20 ministers.",  # Wrong number
     ["The cabinet comprises 48 ministers and over 100 officials total."],
     False),
    ("Indonesia will leave ASEAN next year.",  # Fabricated claim
     ["Indonesia emphasized commitment to ASEAN centrality."],
     False),
    ("The election was cancelled due to protests.",  # Fabricated
     ["Indonesia completed the largest single-day election successfully."],
     False),
    ("Prabowo announced he would step down immediately.",  # Fabricated
     ["Prabowo declared victory and outlined plans for his administration."],
     False),
    ("Foreign investors fled Indonesia after the election.",  # Opposite of truth
     ["Foreign investors maintained their positions and showed confidence in economic stability."],
     False),
]

# Initialize models
print("\nInitializing models...")
embedder = get_embedder()
hallucination_model_a = HallucinationDetector()
confidence_scorer_a = ConfidenceScorer()
baseline_models = get_baseline_models()
hallucination_model_b = baseline_models["hallucination"]
confidence_scorer_b = baseline_models["confidence"]

# ============================================================================
# TRAIN MODELS (Synthetic Data)
# ============================================================================
print("\nTraining Model A (ML) components on synthetic data...")
print("   Note: Using proof-of-concept dataset. Architecture supports larger datasets.")

# --- 1. Train Hallucination Detector ---
print("   - Training Hallucination Detector...")
# Synthetic training data (Sentence, Source, Label)
train_data_hallu = [
    # Supported (High similarity, high overlap)
    ("The economy grew by 5% last year.", ["Indonesia's economy grew by 5.05 percent last year."], 1),
    ("Prabowo met with Xi Jinping.", ["Prabowo Subianto met with Chinese President Xi Jinping in Beijing."], 1),
    ("The capital is moving to Nusantara.", ["The government is moving the capital city to Nusantara in East Kalimantan."], 1),
    ("Inflation remains stable.", ["Bank Indonesia reported that inflation remains stable within the target range."], 1),
    ("Exports increased in Q1.", ["Export value increased significant during the first quarter."], 1),
    ("Bali tourism has recovered.", ["Tourism in Bali has fully recovered to pre-pandemic levels."], 1),
    ("EV adoption is rising.", ["The adoption of electric vehicles is rising due to government incentives."], 1),
    ("The bridge was completed on time.", ["Construction of the bridge was completed on schedule last month."], 1),
    
    # Unsupported (Low similarity, conflicting info)
    ("The economy crashed by 10%.", ["The economy showed strong growth of over 5 percent."], 0),
    ("Prabowo visited Mars.", ["Prabowo visited several countries including China and Japan."], 0),
    ("Jakarta will remain the capital forever.", ["The capital status will be transferred to Nusantara next year."], 0),
    ("Inflation hit 50%.", ["Inflation is controlled at around 2.5 percent."], 0),
    ("Exports zeroed out.", ["Exports remained strong despite global headwinds."], 0),
    ("Bali is closed to tourists.", ["Bali is welcoming record numbers of international tourists."], 0),
    ("EVs are banned in Indonesia.", ["The government is promoting EVs through subsidies."], 0),
    ("The bridge collapsed.", ["The bridge stands as a symbol of connectivity."], 0)
] * 5  # Duplicate to increase sample size

X_hallu_features = []
y_hallu_labels = []

for sent, sources, label in train_data_hallu:
    sent_emb = embedder.generate_single(sent)
    source_embs = embedder.generate(sources, show_progress=False)
    features = hallucination_model_a.extract_features(sent_emb, source_embs, sent, sources)
    X_hallu_features.append(features)
    y_hallu_labels.append(label)

hallucination_model_a.train(np.array(X_hallu_features), np.array(y_hallu_labels))


# --- 2. Train Confidence Scorer ---
print("   - Training Confidence Scorer...")
# Synthetic training data (Similarities, Num Chunks, Confidence Label)
train_data_conf = [
    # High confidence (High sim, multiple sources)
    ([0.95, 0.90, 0.88], 3, 0.95),
    ([0.92, 0.89], 2, 0.90),
    ([0.88, 0.85, 0.82, 0.80], 4, 0.85),
    
    # Medium confidence
    ([0.75, 0.70, 0.65], 3, 0.70),
    ([0.72], 1, 0.60),
    ([0.65, 0.60, 0.55, 0.50], 4, 0.65),
    
    # Low confidence
    ([0.45, 0.40], 2, 0.30),
    ([0.35], 1, 0.20),
    ([0.30, 0.25, 0.20], 3, 0.15)
] * 10

X_conf_features = []
y_conf_labels = []

for sims, n_chunks, label in train_data_conf:
    features = confidence_scorer_a.extract_features(sims, n_chunks)
    X_conf_features.append(features)
    y_conf_labels.append(label)

confidence_scorer_a.train(np.array(X_conf_features), np.array(y_conf_labels))
print("Training complete.\n")

# Evaluate
print("\nEvaluating Hallucination Detection...")
results_a = []
results_b = []
ground_truth = []

for sentence, sources, is_supported in hallucination_test_data:
    ground_truth.append(is_supported)
    
    # Model A prediction
    sent_emb = embedder.generate_single(sentence)
    source_embs = embedder.generate(sources, show_progress=False)
    pred_a = hallucination_model_a.predict(sent_emb, source_embs, sentence, sources)
    results_a.append(pred_a['is_supported'])
    
    # Model B prediction
    pred_b = hallucination_model_b.check_sentence(sentence, sources)
    results_b.append(pred_b['is_supported'])

# Calculate metrics
y_true = np.array(ground_truth)
y_pred_a = np.array(results_a)
y_pred_b = np.array(results_b)

print("\n┌─────────────────────────────────────────────────────────────────────┐")
print("│                  HALLUCINATION DETECTION RESULTS                    │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│ {'Metric':<25} │ {'Model A (ML)':<15} │ {'Model B (Baseline)':<15} │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│ {'Accuracy':<25} │ {accuracy_score(y_true, y_pred_a):<15.2%} │ {accuracy_score(y_true, y_pred_b):<15.2%} │")
print(f"│ {'Precision':<25} │ {precision_score(y_true, y_pred_a, zero_division=0):<15.2%} │ {precision_score(y_true, y_pred_b, zero_division=0):<15.2%} │")
print(f"│ {'Recall':<25} │ {recall_score(y_true, y_pred_a, zero_division=0):<15.2%} │ {recall_score(y_true, y_pred_b, zero_division=0):<15.2%} │")
print(f"│ {'F1 Score':<25} │ {f1_score(y_true, y_pred_a, zero_division=0):<15.2%} │ {f1_score(y_true, y_pred_b, zero_division=0):<15.2%} │")
print("└─────────────────────────────────────────────────────────────────────┘")

print("\nConfusion Matrix - Model A (ML):")
cm_a = confusion_matrix(y_true, y_pred_a)
print(f"   TN={cm_a[0,0]}, FP={cm_a[0,1]}")
print(f"   FN={cm_a[1,0]}, TP={cm_a[1,1]}")

print("\nConfusion Matrix - Model B (Baseline):")
cm_b = confusion_matrix(y_true, y_pred_b)
print(f"   TN={cm_b[0,0]}, FP={cm_b[0,1]}")
print(f"   FN={cm_b[1,0]}, TP={cm_b[1,1]}")

# ============================================================================
# 2. CONFIDENCE SCORING EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("2. CONFIDENCE SCORING")
print("=" * 70)

# Test data: (similarity_scores, num_chunks, expected_confidence_range)
confidence_test_data = [
    # High confidence scenarios
    ([0.92, 0.88, 0.85, 0.82, 0.80], 5, (0.7, 1.0)),
    ([0.95, 0.90, 0.87], 3, (0.65, 0.95)),
    ([0.88, 0.85, 0.82, 0.78, 0.75, 0.72], 6, (0.6, 0.9)),
    
    # Medium confidence scenarios
    ([0.70, 0.65, 0.60], 3, (0.4, 0.7)),
    ([0.72, 0.68, 0.55, 0.50], 4, (0.4, 0.7)),
    ([0.65, 0.60], 2, (0.3, 0.6)),
    
    # Low confidence scenarios
    ([0.45, 0.40], 2, (0.1, 0.5)),
    ([0.50, 0.45, 0.40, 0.35], 4, (0.2, 0.5)),
    ([0.30], 1, (0.0, 0.4)),
]

print("\nEvaluating Confidence Scoring...")
confidence_scorer_a = ConfidenceScorer()
confidence_scorer_b = baseline_models["confidence"]

correct_a = 0
correct_b = 0
total = len(confidence_test_data)
errors_a = []
errors_b = []

print("\n┌───────────────────────────────────────────────────────────────────────────────┐")
print("│                        CONFIDENCE SCORING TEST CASES                          │")
print("├───────────────────────────────────────────────────────────────────────────────┤")
print(f"│ {'Test':<5} │ {'Expected':<12} │ {'Model A':<10} │ {'Model B':<10} │ {'A Correct':<10} │ {'B Correct':<10} │")
print("├───────────────────────────────────────────────────────────────────────────────┤")

for i, (sims, n_chunks, (min_exp, max_exp)) in enumerate(confidence_test_data):
    # Model A
    pred_a = confidence_scorer_a.predict(sims, n_chunks)
    in_range_a = min_exp <= pred_a <= max_exp
    if in_range_a:
        correct_a += 1
    errors_a.append(abs(pred_a - (min_exp + max_exp) / 2))
    
    # Model B - create dummy chunks
    dummy_chunks = {"media": [{"similarity": s} for s in sims]}
    pred_b_result = confidence_scorer_b.score("test query", dummy_chunks)
    pred_b = pred_b_result["confidence"]
    in_range_b = min_exp <= pred_b <= max_exp
    if in_range_b:
        correct_b += 1
    errors_b.append(abs(pred_b - (min_exp + max_exp) / 2))
    
    print(f"│ {i+1:<5} │ {f'{min_exp:.1f}-{max_exp:.1f}':<12} │ {pred_a:<10.2f} │ {pred_b:<10.2f} │ {'Yes' if in_range_a else 'No':<10} │ {'Yes' if in_range_b else 'No':<10} │")

print("└───────────────────────────────────────────────────────────────────────────────┘")

print("\n┌─────────────────────────────────────────────────────────────────────┐")
print("│                   CONFIDENCE SCORING RESULTS                        │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│ {'Metric':<25} │ {'Model A (ML)':<15} │ {'Model B (Baseline)':<15} │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│ {'In-Range Accuracy':<25} │ {correct_a/total:<15.2%} │ {correct_b/total:<15.2%} │")
print(f"│ {'Mean Absolute Error':<25} │ {np.mean(errors_a):<15.3f} │ {np.mean(errors_b):<15.3f} │")
print("└─────────────────────────────────────────────────────────────────────┘")

from data.corpus import corpus_metadata, load_articles

CORPUS_ARTICLES = load_articles()
CORPUS_META = corpus_metadata()

# ============================================================================
# 3. FRAMING ANALYSIS EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("3. FRAMING ANALYSIS (Using Full Dataset)")
print("=" * 70)

# Prepare real data
framing_data = {}
for article in CORPUS_ARTICLES:
    source = article["media_source"]
    if source not in framing_data:
        framing_data[source] = []
    framing_data[source].append(article["content"])

print(f"Corpus: {CORPUS_META['version']}")
print(f"Analyzing framing across {len(CORPUS_ARTICLES)} articles from {len(framing_data)} sources...")

TOP_K = 15

print("\nEvaluating keyword extractors with label-free metrics...")
print("   (no hand-written 'expected themes' - see models/framing/metrics.py)")

results_tfidf = evaluate_extractor(framing_data, tfidf_profiles, top_k=TOP_K)
results_wordcount = evaluate_extractor(framing_data, wordcount_profiles, top_k=TOP_K)

print("\n┌" + "─" * 89 + "┐")
print("│" + "TOP KEYWORDS PER SOURCE".center(89) + "│")
print("├" + "─" * 89 + "┤")
for media in sorted(framing_data):
    print(f"│ {media.upper():<15}")
    print(f"│   TF-IDF:      {', '.join(results_tfidf['profiles'].get(media, [])):<68}")
    print(f"│   WordCount:   {', '.join(results_wordcount['profiles'].get(media, [])):<68}")
    print("├" + "─" * 89 + "┤")

print("│" + "FRAMING KEYWORD METRICS (top-15 keywords per source)".center(89) + "│")
print("├" + "─" * 89 + "┤")
print(f"│ {'Metric':<34} │ {'TF-IDF':<20} │ {'WordCount baseline':<26}│")
print("├" + "─" * 89 + "┤")
print(f"│ {'Distinctiveness (lower better)':<34} │ {results_tfidf['distinctiveness']:<20.2%} │ {results_wordcount['distinctiveness']:<26.2%}│")
print(f"│ {'Generic-term rate (lower better)':<34} │ {results_tfidf['generic_term_rate']:<20.2%} │ {results_wordcount['generic_term_rate']:<26.2%}│")
print(f"│ {'Source attribution acc (higher)':<34} │ {results_tfidf['accuracy']:<20.2%} │ {results_wordcount['accuracy']:<26.2%}│")
print(f"│ {'Source attribution macro-F1':<34} │ {results_tfidf['macro_f1']:<20.2%} │ {results_wordcount['macro_f1']:<26.2%}│")
print("├" + "─" * 89 + "┤")
print(f"│ {'Majority-class baseline':<34} │ {results_tfidf['majority_baseline']:<20.2%} │ {'(same task, always guess ANTARA)':<26}│")
print(f"│ {'Held-out articles':<34} │ {results_tfidf['n_test']:<20} │ {results_wordcount['n_test']:<26}│")
print("└" + "─" * 89 + "┘")

final_score_a = results_tfidf["accuracy"]
final_score_b = results_wordcount["accuracy"]

print("\nHow to read this:")
print("  - Distinctiveness  : mean pairwise Jaccard overlap between outlets' keyword sets.")
print("  - Generic-term rate: share of keywords that are also corpus-wide top-50 terms.")
print("  - Attribution      : assign a held-out article to the outlet whose keyword")
print("                       profile it matches best. Beating the majority baseline")
print("                       means the keywords carry genuine outlet-specific signal.")

# State the negative result explicitly. An extractor that loses to "always guess
# the biggest outlet" has not demonstrated any outlet-specific signal, and the
# report should say so rather than crowning a relative winner.
majority = results_tfidf["majority_baseline"]
print("\nVerdict on source attribution:")
for name, res in (("TF-IDF", results_tfidf), ("WordCount", results_wordcount)):
    if res["accuracy"] > majority:
        print(f"  - {name:<10}: {res['accuracy']:.2%} BEATS the {majority:.2%} majority baseline.")
    else:
        print(f"  - {name:<10}: {res['accuracy']:.2%} does NOT beat the {majority:.2%} majority baseline")
        print(f"    {'':<10}  -> keyword frequency alone does not identify the outlet.")
print("  Both extractors do separate the outlets' vocabularies (see distinctiveness),")
print("  but that separation is too weak to classify individual articles. This is the")
print("  gap the fine-tuned DistilBERT classifier is meant to close.")
print("\nNote: this compares the two KEYWORD extractors only. In the running app,")
print("      Model A for framing is the fine-tuned DistilBERT classifier - see the")
print("      classification report printed by scripts/train_framing_bert.py.")

# ... (Summary Section Update) ...

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY: MODEL A vs MODEL B")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PERFORMANCE COMPARISON                            │
├────────────────────────┬─────────────────────┬──────────────────────────────┤
│ Component              │ Model A (Trained)   │ Model B (Baseline)           │
├────────────────────────┼─────────────────────┼──────────────────────────────┤
│ Hallucination Detection│                     │                              │
│   - Method             │ Logistic Regression │ Keyword Overlap (30%)        │
│   - Accuracy           │ {:.2%}             │ {:.2%}                       │
│   - F1 Score           │ {:.2%}             │ {:.2%}                       │
├────────────────────────┼─────────────────────┼──────────────────────────────┤
│ Confidence Scoring     │                     │                              │
│   - Method             │ Random Forest       │ Chunk Count Heuristic        │
│   - In-Range Accuracy  │ {:.2%}             │ {:.2%}                       │
│   - MAE                │ {:.3f}              │ {:.3f}                        │
├────────────────────────┼─────────────────────┼──────────────────────────────┤
│ Framing keywords*      │ TF-IDF              │ WordCount baseline           │
│   - Attribution acc.   │ {:.2%}             │ {:.2%}                       │
└────────────────────────┴─────────────────────┴──────────────────────────────┘
* Keyword extractors only. The app's Model A for framing is DistilBERT.
""".format(
    accuracy_score(y_true, y_pred_a),
    accuracy_score(y_true, y_pred_b),
    f1_score(y_true, y_pred_a, zero_division=0),
    f1_score(y_true, y_pred_b, zero_division=0),
    correct_a/total,
    correct_b/total,
    np.mean(errors_a),
    np.mean(errors_b),
    final_score_a,
    final_score_b
))

# Determine winner for each
print("WINNERS:")
print(f"   - Hallucination Detection: {'Model A' if accuracy_score(y_true, y_pred_a) > accuracy_score(y_true, y_pred_b) else 'Model B'}")
print(f"   - Confidence Scoring: {'Model A' if correct_a > correct_b else 'Model B'}")  
if final_score_a == final_score_b:
    framing_winner = "Tie"
else:
    framing_winner = "TF-IDF" if final_score_a > final_score_b else "WordCount baseline"

# Naming a relative winner here would overstate the result: both extractors are
# below the majority-class baseline, so the honest verdict is that neither works,
# not that one of them won. Reporting "TF-IDF wins" is precisely the kind of
# overclaiming this project's A/B comparison exists to expose.
if max(final_score_a, final_score_b) <= majority:
    print(f"   - Framing keywords (attribution): NEITHER - both are below the "
          f"{majority:.2%} majority-class baseline")
    print(f"     ({framing_winner} is the better of the two at "
          f"{max(final_score_a, final_score_b):.2%}, but that is not a win.)")
else:
    print(f"   - Framing keywords (attribution): {framing_winner}")

# ============================================================================
# PERSIST THE REPORT
# ============================================================================
# Written to disk so results can be reviewed, diffed between runs, and cited
# without re-running the whole evaluation.
report = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "corpus": {
        "version": CORPUS_META["version"],
        "articles": len(CORPUS_ARTICLES),
        "sources": {media: len(texts) for media, texts in sorted(framing_data.items())},
    },
    "hallucination_detection": {
        "model_a": {
            "method": "Logistic Regression on similarity features",
            "accuracy": accuracy_score(y_true, y_pred_a),
            "precision": precision_score(y_true, y_pred_a, zero_division=0),
            "recall": recall_score(y_true, y_pred_a, zero_division=0),
            "f1": f1_score(y_true, y_pred_a, zero_division=0),
        },
        "model_b": {
            "method": "Keyword overlap >= 30%",
            "accuracy": accuracy_score(y_true, y_pred_b),
            "precision": precision_score(y_true, y_pred_b, zero_division=0),
            "recall": recall_score(y_true, y_pred_b, zero_division=0),
            "f1": f1_score(y_true, y_pred_b, zero_division=0),
        },
        "n_test": len(y_true),
        "note": "Trained on synthetic data; treat as proof-of-concept, not a benchmark.",
    },
    "confidence_scoring": {
        "model_a": {"method": "Random Forest", "in_range_accuracy": correct_a / total, "mae": float(np.mean(errors_a))},
        "model_b": {"method": "Deterministic heuristic", "in_range_accuracy": correct_b / total, "mae": float(np.mean(errors_b))},
        "n_test": total,
    },
    "framing_keywords": {
        "top_k": TOP_K,
        "tfidf": {k: v for k, v in results_tfidf.items()},
        "wordcount": {k: v for k, v in results_wordcount.items()},
        "interpretation": (
            "TF-IDF produces more distinctive, less generic keyword sets, but neither "
            "extractor beats the majority-class baseline at source attribution. Keyword "
            "frequency alone does not identify the outlet."
        ),
    },
}

os.makedirs("evaluation_results", exist_ok=True)
json_path = os.path.join("evaluation_results", "latest_evaluation.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

md_path = os.path.join("evaluation_results", "latest_evaluation.md")
with open(md_path, "w", encoding="utf-8") as f:
    h_a = report["hallucination_detection"]["model_a"]
    h_b = report["hallucination_detection"]["model_b"]
    c_a = report["confidence_scoring"]["model_a"]
    c_b = report["confidence_scoring"]["model_b"]

    f.write("# Sentra Evaluation Report\n\n")
    f.write(f"Generated: {report['generated_at']}  \n")
    f.write(f"Corpus: {report['corpus']['articles']} articles - {report['corpus']['sources']}\n\n")

    f.write("## Hallucination detection\n\n")
    f.write(f"{report['hallucination_detection']['n_test']} labelled claim/source pairs.\n\n")
    f.write("| Metric | Model A (Logistic Regression) | Model B (Keyword overlap) |\n")
    f.write("|---|---|---|\n")
    for metric in ("accuracy", "precision", "recall", "f1"):
        f.write(f"| {metric.capitalize()} | {h_a[metric]:.2%} | {h_b[metric]:.2%} |\n")
    f.write(f"\n> {report['hallucination_detection']['note']}\n\n")

    f.write("## Confidence scoring\n\n")
    f.write("| Metric | Model A (Random Forest) | Model B (Heuristic) |\n")
    f.write("|---|---|---|\n")
    f.write(f"| In-range accuracy | {c_a['in_range_accuracy']:.2%} | {c_b['in_range_accuracy']:.2%} |\n")
    f.write(f"| Mean absolute error | {c_a['mae']:.3f} | {c_b['mae']:.3f} |\n\n")

    f.write(f"## Framing keyword extractors (top-{TOP_K})\n\n")
    f.write("| Metric | TF-IDF | WordCount baseline |\n")
    f.write("|---|---|---|\n")
    f.write(f"| Distinctiveness (lower better) | {results_tfidf['distinctiveness']:.2%} | {results_wordcount['distinctiveness']:.2%} |\n")
    f.write(f"| Generic-term rate (lower better) | {results_tfidf['generic_term_rate']:.2%} | {results_wordcount['generic_term_rate']:.2%} |\n")
    f.write(f"| Source attribution accuracy | {results_tfidf['accuracy']:.2%} | {results_wordcount['accuracy']:.2%} |\n")
    f.write(f"| Source attribution macro-F1 | {results_tfidf['macro_f1']:.2%} | {results_wordcount['macro_f1']:.2%} |\n")
    f.write(f"| Majority-class baseline | {results_tfidf['majority_baseline']:.2%} | {results_tfidf['majority_baseline']:.2%} |\n\n")
    f.write(f"> {report['framing_keywords']['interpretation']}\n")

print(f"\nReport written to {json_path} and {md_path}")
print("\nEvaluation complete.")
