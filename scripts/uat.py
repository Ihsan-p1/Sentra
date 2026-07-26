"""
User Acceptance Test for Sentra.

Unit tests prove the parts work. This proves the *product* behaves the way a
user expects, against a live server, a live database, and a live LLM.

Each scenario states what a user does and what must be true of the response.
Failures print the actual answer so a regression is diagnosable without a rerun.

Usage:
    python -m uvicorn api.main:app        # terminal 1
    python scripts/uat.py                 # terminal 2

    python scripts/uat.py --base http://localhost:8000
    python scripts/uat.py --only language     # run one section
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ALLOWED_SOURCES = {"ANTARANEWS", "TEMPO", "ABC_NEWS"}

PASS = "PASS"
FAIL = "FAIL"


class Client:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.session_id = None

    def health(self):
        return requests.get(f"{self.base}/api/health", timeout=10)

    def ask(self, message, mode="default", keep_session=False):
        payload = {"message": message, "mode": mode}
        if keep_session and self.session_id:
            payload["session_id"] = self.session_id
        started = time.time()
        response = requests.post(f"{self.base}/api/chat", json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        data["_latency"] = time.time() - started
        self.session_id = data.get("session_id")
        return data

    def new_session(self):
        self.session_id = None


# --- Checks -----------------------------------------------------------------
# Each returns (ok, detail). Kept tiny and independent so a scenario can compose
# several and report every failing condition, not just the first.

def answered(data):
    text = (data.get("answer") or "").strip()
    return bool(text), f"answer length {len(text)}"


def refused(data):
    sources = data.get("sources") or {}
    total = sum(len(v) for v in sources.values())
    quality = data.get("source_quality") or {}
    is_refusal = total == 0 or "refusal_reason" in quality
    return is_refusal, f"{total} chunks retrieved"


def has_sources(minimum=1):
    def check(data):
        total = sum(len(v) for v in (data.get("sources") or {}).values())
        return total >= minimum, f"{total} chunks"
    return check


def language_is(expected):
    def check(data):
        actual = data.get("language")
        return actual == expected, f"language={actual}"
    return check


def reads_as_indonesian(data):
    lowered = f" {(data.get('answer') or '').lower()} "
    markers = [" yang ", " dan ", " dengan ", " adalah ", " tidak ", " pada ", " untuk ", " dari "]
    hits = [m.strip() for m in markers if m in lowered]
    return len(hits) >= 2, f"indonesian markers: {hits or 'none'}"


def reads_as_english(data):
    lowered = f" {(data.get('answer') or '').lower()} "
    markers = [" the ", " and ", " of ", " that ", " with ", " is ", " to "]
    hits = [m.strip() for m in markers if m in lowered]
    return len(hits) >= 2, f"english markers: {hits or 'none'}"


def headings_in_language(expected):
    """
    Section headings must match the answer's language too.

    Regression: the body came back in Indonesian while the headings stayed
    "## Direct Answer" / "## Summary", because the model copied the English
    headings it could see in the prompt template.
    """
    english_headings = {"direct answer", "summary", "media coverage by source",
                        "comparative framing analysis", "factual summary"}
    indonesian_headings = {"jawaban langsung", "ringkasan", "liputan media per sumber",
                           "analisis framing komparatif", "ringkasan faktual"}

    def check(data):
        found = {h.strip().lower().rstrip(":") for h in
                 re.findall(r"^#{2,3}\s*(.+?)\s*$", data.get("answer") or "", re.M)}
        en_hits = found & english_headings
        id_hits = found & indonesian_headings

        if expected == "id":
            return not en_hits, f"english headings present: {sorted(en_hits) or 'none'}"
        return not id_hits, f"indonesian headings present: {sorted(id_hits) or 'none'}"

    return check


def no_invented_sources(data):
    headings = re.findall(r"^#{2,4}\s*([A-Za-z_ ]+)\s*$", data.get("answer") or "", re.M)
    candidates = [h.strip().upper().replace(" ", "_") for h in headings]
    invented = [
        h for h in candidates
        if h not in ALLOWED_SOURCES
        and any(tok in h for tok in ("NEWS", "ANT", "TEMPO", "ABC", "POST", "GLOBE", "TV"))
    ]
    return not invented, f"invented: {invented or 'none'}"


def intent_is(expected):
    def check(data):
        return data.get("intent") == expected, f"intent={data.get('intent')}"
    return check


def has_ab_comparison(data):
    comparison = data.get("model_comparison") or {}
    missing = [
        f"{stage}.{model}"
        for stage in ("hallucination", "confidence", "framing")
        for model in ("model_a", "model_b")
        if model not in (comparison.get(stage) or {})
    ]
    return not missing, f"missing: {missing or 'none'}"


def framing_model_a_populated(data):
    keywords = ((data.get("model_comparison") or {}).get("framing") or {}).get("model_a", {}).get("keywords", {})
    populated = {m: v for m, v in keywords.items() if v}
    return len(populated) >= 1, f"sources with style match: {sorted(populated)}"


def verification_is_coherent(data):
    summary = data.get("verification_summary") or {}
    if not summary:
        return True, "no verification (refusal)"
    supported, total = summary.get("supported_count", 0), summary.get("total_checked", 0)
    rate = summary.get("support_rate", 0)
    ok = supported <= total and 0.0 <= rate <= 1.0 and summary.get("level") in {"high", "moderate", "low"}
    return ok, f"{supported}/{total} level={summary.get('level')}"


def unsupported_ratio_is_valid(data):
    hallucination = (data.get("model_comparison") or {}).get("hallucination") or {}
    if not hallucination:
        return True, "no comparison (refusal)"
    details = []
    for model in ("model_a", "model_b"):
        raw = hallucination.get(model, {}).get("unsupported_ratio", "0/0")
        unsupported, total = (int(x) for x in raw.split("/"))
        if unsupported > total:
            return False, f"{model} {raw} exceeds total"
        details.append(f"{model}={raw}")
    return True, " ".join(details)


def faster_than(seconds):
    def check(data):
        return data["_latency"] < seconds, f"{data['_latency']:.1f}s"
    return check


def mentions(*required):
    """
    The answer must contain every given string (case-insensitive).

    Most checks here verify shape - that something was answered, in the right
    language, with the panels populated. That is not enough for the one question
    every user asks first. "Who won?" was answered fluently and at length while
    never naming the winner, because the corpus mentioned him exactly once, in
    passing. Only a content assertion catches that.
    """
    def check(data):
        answer = (data.get("answer") or "").lower()
        missing = [term for term in required if term.lower() not in answer]
        return not missing, f"missing: {missing or 'none'}"
    return check


def evaluated_or_refused(data):
    """
    Verification either ran on a non-zero number of sentences, or the reply is a
    refusal that correctly reports nothing.

    Guards the sidebar honesty fix from the API side: a "0/0" comparison used to
    be rendered as green "Strong Alignment", claiming a verification result that
    had never been computed.
    """
    hallucination = (data.get("model_comparison") or {}).get("hallucination") or {}
    if not hallucination:
        return True, "no comparison (refusal or small talk)"

    totals = {}
    for model in ("model_a", "model_b"):
        raw = hallucination.get(model, {}).get("unsupported_ratio", "0/0")
        totals[model] = int(raw.split("/")[1])

    ok = all(total > 0 for total in totals.values())
    return ok, f"checked {totals}"


# --- Scenarios --------------------------------------------------------------

SCENARIOS = [
    # section, name, query, mode, checks
    ("smalltalk", "Indonesian greeting is answered conversationally", "halo", "default",
     [answered, intent_is("greeting"), refused, language_is("id")]),
    ("smalltalk", "English greeting is answered conversationally", "hello", "default",
     [answered, intent_is("greeting"), refused, language_is("en")]),
    ("smalltalk", "Asking what the bot is does not hit retrieval", "kamu bisa apa", "default",
     [answered, intent_is("greeting"), refused]),
    ("smalltalk", "Thanks is acknowledged", "terima kasih", "default",
     [answered, intent_is("thanks"), refused]),

    ("help", "Asking what Analysis Results does is answered",
     "apa fungsi dari analysis results?", "default",
     [answered, intent_is("help"), refused, language_is("id")]),
    ("help", "Asking for the feature list is answered",
     "apa saja fitur project ini", "default",
     [answered, intent_is("help"), refused, language_is("id")]),
    ("help", "Model A vs Model B is explained", "what is the difference between model a and model b",
     "default", [answered, intent_is("help"), refused, language_is("en")]),
    ("help", "Hallucination method is explained", "bagaimana cara deteksi halusinasi bekerja",
     "default", [answered, intent_is("help"), refused]),
    ("help", "Framing keywords panel is explained", "apa itu framing keywords", "default",
     [answered, intent_is("help"), refused]),
    ("help", "A framing research question is NOT hijacked by the help panel",
     "how did the outlets frame the election differently", "default",
     [answered, has_sources(), intent_is(None)]),

    ("language", "Indonesian question gets an Indonesian answer",
     "siapa yang memenangkan pemilu indonesia 2024", "default",
     [answered, language_is("id"), reads_as_indonesian, has_sources(),
      headings_in_language("id")]),
    ("language", "English question gets an English answer",
     "who won the 2024 indonesian election", "default",
     [answered, language_is("en"), reads_as_english, has_sources(),
      headings_in_language("en")]),
    ("language", "Indonesian follow-up stays Indonesian",
     "bagaimana media meliput pembentukan kabinet", "default",
     [answered, language_is("id"), reads_as_indonesian, headings_in_language("id")]),
    ("language", "Indonesian strict mode keeps Indonesian headings",
     "apa yang dikatakan prabowo tentang swasembada pangan", "reduce_hallucination",
     [answered, language_is("id"), headings_in_language("id")]),

    ("retrieval", "Previously-refused Indonesian person query now answers",
     "siapa itu prabowo subianto", "default",
     [answered, has_sources(), language_is("id")]),
    # The corpus previously named the 2024 winner once, in passing, inside one
    # article. Retrieval could not surface a single passing mention, so the most
    # basic question in the domain was unanswerable. The fix was corpus data -
    # articles reporting the official result - not a threshold or prompt change.
    ("retrieval", "English winner question names the winner",
     "who won the 2024 indonesian election", "default",
     [answered, has_sources(), mentions("prabowo")]),
    ("retrieval", "Indonesian winner question names the winner",
     "siapa pemenang pemilu 2024", "default",
     [answered, has_sources(), language_is("id"), mentions("prabowo")]),
    ("retrieval", "Official vote count is retrievable",
     "how many votes did prabowo get in the 2024 election", "default",
     [answered, has_sources(), mentions("96,214,691")]),
    ("retrieval", "Inauguration date is answerable",
     "when was prabowo inaugurated as president", "default",
     [answered, has_sources(), mentions("october")]),
    ("retrieval", "Core corpus topic is answerable", "prabowo cabinet appointments", "default",
     [answered, has_sources()]),
    ("retrieval", "Constitutional court topic is answerable",
     "what did the constitutional court decide about the election", "default",
     [answered, has_sources()]),
    ("retrieval", "Out-of-corpus question is refused", "what is the price of bitcoin today", "default",
     [answered, refused]),
    ("retrieval", "Unrelated recipe question is refused", "how do i cook fried rice", "default",
     [answered, refused]),

    ("grounding", "No invented media outlets in the answer",
     "how did each outlet cover the election result", "default",
     [answered, no_invented_sources, has_sources()]),
    # NOTE: "compare how the outlets framed the cabinet formation" is deliberately
    # not used here. It scores 0.329, inside the overlap between answerable and
    # unanswerable queries, so it is refused at the calibrated 0.35 threshold.
    # Lowering the threshold to admit it would leak 3 of 8 unanswerable queries.
    # That trade-off is a documented limitation, not something a UAT should mask.
    ("grounding", "A/B comparison is fully populated",
     "how did antaranews and tempo differ in covering the cabinet appointments", "default",
     [answered, has_ab_comparison, framing_model_a_populated]),
    ("grounding", "Verification summary is internally consistent",
     "what were the main developments after the election", "default",
     [answered, verification_is_coherent, unsupported_ratio_is_valid, evaluated_or_refused]),
    ("grounding", "A refusal reports no verification rather than a perfect score",
     "what is the price of bitcoin today", "default",
     [answered, refused, evaluated_or_refused]),

    ("strict", "Strict mode rejects a too-short query", "who?", "reduce_hallucination",
     [answered, refused]),
    ("strict", "Strict mode answers a well-specified question",
     "what did prabowo say about food self sufficiency", "reduce_hallucination",
     [answered]),
    # NOTE: "weather in Jakarta" is deliberately NOT used as an out-of-corpus
    # probe. The corpus contains four ANTARANEWS articles on Jakarta flooding
    # and the rainy season, so retrieving for it is correct behaviour.
    ("strict", "Strict mode refuses a genuinely unrelated question",
     "bagaimana cara membuat aplikasi android", "reduce_hallucination",
     [answered, refused, language_is("id")]),

    ("performance", "Warm query returns within 90 seconds",
     "who is gibran rakabuming", "default",
     [answered, faster_than(90)]),
]


def run_multiturn(client, results):
    """Multi-turn memory cannot be expressed as a single-query scenario."""
    section = "memory"
    client.new_session()

    first = client.ask("tell me about the 2024 presidential election", keep_session=True)
    session_id = first.get("session_id")

    second = client.ask("tell me more", keep_session=True)

    checks = [
        ("session id is issued", bool(session_id), str(session_id)),
        ("session id is stable across turns", second.get("session_id") == session_id,
         f"{second.get('session_id')}"),
        ("follow-up produces an answer", bool((second.get("answer") or "").strip()),
         f"{len((second.get('answer') or ''))} chars"),
        ("follow-up retrieved sources",
         sum(len(v) for v in (second.get("sources") or {}).values()) > 0,
         f"{sum(len(v) for v in (second.get('sources') or {}).values())} chunks"),
    ]
    for name, ok, detail in checks:
        results.append((section, f"Multi-turn: {name}", PASS if ok else FAIL, detail, ""))

    cleared = requests.post(
        f"{client.base}/api/chat/clear", json={"session_id": session_id}, timeout=30
    ).json()
    ok = cleared.get("status") == "cleared"
    results.append((section, "Multi-turn: session can be cleared", PASS if ok else FAIL,
                    str(cleared.get("status")), ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.getenv("SENTRA_API_BASE", "http://localhost:8000"))
    parser.add_argument("--only", help="run a single section")
    parser.add_argument("--json", help="write machine-readable results to this path")
    args = parser.parse_args()

    client = Client(args.base)

    print("=" * 78)
    print("SENTRA - USER ACCEPTANCE TEST")
    print("=" * 78)
    print(f"Target : {args.base}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        health = client.health()
        if health.status_code != 200:
            print(f"Server unhealthy: HTTP {health.status_code}")
            return 1
    except requests.exceptions.RequestException as e:
        print(f"Cannot reach {args.base}: {e}")
        print("Start it with: python -m uvicorn api.main:app")
        return 1

    scenarios = [s for s in SCENARIOS if not args.only or s[0] == args.only]
    results = []
    current_section = None

    for section, name, query, mode, checks in scenarios:
        if section != current_section:
            print(f"\n--- {section.upper()} " + "-" * (72 - len(section)))
            current_section = section

        client.new_session()
        try:
            data = client.ask(query, mode=mode)
        except Exception as e:
            print(f"  [FAIL] {name}\n         request failed: {e}")
            results.append((section, name, FAIL, f"request failed: {e}", query))
            continue

        failures = []
        details = []
        for check in checks:
            ok, detail = check(data)
            details.append(detail)
            if not ok:
                failures.append(f"{getattr(check, '__name__', 'check')}: {detail}")

        status = FAIL if failures else PASS
        results.append((section, name, status, "; ".join(details), query))

        print(f"  [{status}] {name}  ({data['_latency']:.1f}s)")
        if failures:
            print(f"         query : {query}")
            for failure in failures:
                print(f"         FAILED: {failure}")
            snippet = " ".join((data.get("answer") or "").split())[:220]
            print(f"         answer: {snippet}")

    if not args.only or args.only == "memory":
        print(f"\n--- MEMORY " + "-" * 67)
        try:
            run_multiturn(client, results)
            for section, name, status, detail, _ in results:
                if section == "memory":
                    print(f"  [{status}] {name}  ({detail})")
        except Exception as e:
            print(f"  [FAIL] multi-turn scenario crashed: {e}")
            results.append(("memory", "Multi-turn scenario", FAIL, str(e), ""))

    passed = sum(1 for r in results if r[2] == PASS)
    failed = len(results) - passed

    print("\n" + "=" * 78)
    print(f"RESULT: {passed} passed, {failed} failed, {len(results)} total")
    print("=" * 78)

    if failed:
        print("\nFailed scenarios:")
        for section, name, status, detail, query in results:
            if status == FAIL:
                print(f"  - [{section}] {name}")
                print(f"      {detail}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "target": args.base,
                    "passed": passed,
                    "failed": failed,
                    "results": [
                        {"section": s, "scenario": n, "status": st, "detail": d, "query": q}
                        for s, n, st, d, q in results
                    ],
                },
                f,
                indent=2,
            )
        print(f"\nJSON report: {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
