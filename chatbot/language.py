"""
Language detection and conversational intent handling.

Two problems this solves:

1. Sentra's corpus is English, but users ask in Indonesian. The answer should
   come back in the language the user wrote in, not the language of the sources.
   The embedding model is cross-lingual (a faithful Indonesian rendering of an
   English source still scores ~0.86 against it), so answering in Indonesian
   does not break hallucination verification.

2. A greeting is not a research question. Sending "halo" through retrieval
   produces "Insufficient source coverage", which reads as a broken bot. Small
   talk is answered directly, without touching the corpus.

Detection is a deliberately dependency-free word-marker heuristic: the inputs are
short chat messages, and a statistical language detector would add a dependency
for a decision between exactly two known languages.
"""
import re
from typing import Dict, List

INDONESIAN = "id"
ENGLISH = "en"

LANGUAGE_NAMES = {
    INDONESIAN: "Indonesian (Bahasa Indonesia)",
    ENGLISH: "English",
}

# Function words that rarely appear in the other language. Content words
# (politik, presiden, media) are excluded on purpose - they are near-identical
# across both languages and would make detection noisier, not sharper.
ID_MARKERS = {
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan", "itu", "ini",
    "adalah", "apa", "siapa", "kenapa", "mengapa", "bagaimana", "kapan", "dimana",
    "tidak", "bukan", "saya", "aku", "kamu", "anda", "kita", "kami", "mereka",
    "bisa", "boleh", "ada", "akan", "sudah", "belum", "juga", "atau", "tapi",
    "tetapi", "karena", "jika", "kalau", "tolong", "coba", "jelaskan", "sebutkan",
    "ceritakan", "bagaimanakah", "apakah", "berapa", "banyak", "lebih", "sangat",
    "gimana", "kok", "nih", "dong", "ya", "iya", "nya", "sih", "aja", "saja",
}

EN_MARKERS = {
    "the", "and", "of", "to", "in", "for", "with", "that", "this", "is", "are",
    "was", "were", "what", "who", "why", "how", "when", "where", "which", "not",
    "you", "your", "they", "their", "can", "could", "would", "should", "will",
    "have", "has", "had", "does", "did", "about", "tell", "explain", "please",
    "more", "between", "compare", "does", "there", "these", "those",
}

# Greetings carry no function words, so language detection has to key off the
# greeting itself: "halo" is Indonesian, "hello" is English.
ID_GREETINGS = {
    "halo", "hallo", "hai", "oi", "woi", "permisi", "salam",
    "assalamualaikum", "assalamu'alaikum",
    "pagi", "siang", "sore", "malam", "selamat",
}

EN_GREETINGS = {
    "hi", "hey", "hello", "helo", "yo",
    "morning", "afternoon", "evening", "good",
}

GREETINGS = ID_GREETINGS | EN_GREETINGS

GREETING_PHRASES = [
    "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
    "good morning", "good afternoon", "good evening", "good day",
    "apa kabar", "gimana kabarnya", "how are you", "how's it going",
    "kamu siapa", "siapa kamu", "who are you", "what are you",
    "kamu bisa apa", "bisa apa", "what can you do", "help me", "bantu saya",
]

THANKS_PHRASES = [
    "terima kasih", "makasih", "trims", "thanks", "thank you", "thx", "nuhun",
]

FAREWELL_PHRASES = [
    "bye", "goodbye", "dadah", "sampai jumpa", "see you", "selamat tinggal",
]

# Intent labels
INTENT_GREETING = "greeting"
INTENT_THANKS = "thanks"
INTENT_FAREWELL = "farewell"
INTENT_HELP = "help"
INTENT_QUESTION = "question"

# --- Questions about the app itself -----------------------------------------
# "apa fungsi dari analysis results?" is not a question about the news corpus,
# so retrieval finds nothing and the bot refuses - which makes it look broken.
#
# Routing is keyed on terms that name the INTERFACE, never on topic words alone.
# "framing" appears in real research questions ("how did outlets frame the
# election"); "framing keywords" is the sidebar label. Only the latter routes here.
HELP_PATTERNS = [
    (["analysis result", "analysis results", "hasil analisis", "panel analisis",
      "sidebar", "panel samping"], "analysis_results"),
    (["model a", "model b", "perbedaan model", "beda model", "bedanya model",
      "a vs b", "a dan b"], "model_ab"),
    (["hallucination detection", "deteksi halusinasi", "hallucination method",
      "metode halusinasi", "cara deteksi halusinasi", "unsupported ratio",
      "strong alignment", "weak alignment"], "hallucination"),
    (["framing keyword", "kata kunci framing", "keyword framing",
      "style match"], "framing"),
    (["confidence score", "skor kepercayaan", "trust meter", "nilai confidence",
      "arti confidence", "persen di atas"], "confidence"),
    (["reduce hallucination", "mode ketat", "strict mode", "toggle"], "modes"),
    (["sources panel", "panel sumber", "daftar sumber", "sumber di kanan"], "sources"),
    (["fitur apa", "apa saja fitur", "fitur dari", "fiturnya apa", "fitur project",
      "fitur aplikasi", "what features", "what can this app", "what does this app",
      "apa fungsi aplikasi", "fungsi aplikasi", "apa itu sentra", "what is sentra",
      "cara kerja", "how does this work", "how do you work", "bantuan", "help me use"],
     "overview"),
]

# Terms that mark a message as being about the interface rather than the news.
# A message must contain one of these to be treated as a help request at all.
#
# Matched on WORD BOUNDARIES, not as raw substrings: a bare "app" in this list
# once matched "cabinet appointments" and routed a real research question to a
# canned help reply.
APP_SCOPE_TERMS = [
    "analysis result", "analysis results", "hasil analisis", "model a", "model b",
    "sidebar", "panel", "confidence score", "skor kepercayaan",
    "hallucination detection", "deteksi halusinasi", "framing keyword",
    "framing keywords", "kata kunci framing", "style match",
    "reduce hallucination", "mode ketat", "strict mode", "fitur", "feature",
    "features", "aplikasi", "this app", "the app", "sentra", "chatbot",
    "sistem ini", "cara kerja", "unsupported ratio", "toggle", "trust meter",
    "alignment",
]

def _phrase_pattern(phrase: str) -> str:
    """
    Word-boundary pattern that tolerates a plural 's'.

    Needed because the UI labels are plural ("Framing Keywords", "Analysis
    Results") while users type either form. A strict trailing \\b would reject
    "framing keywords" when the listed term is "framing keyword".
    """
    return rf"\b{re.escape(phrase)}s?\b"


_APP_SCOPE_RE = re.compile("|".join(_phrase_pattern(t) for t in APP_SCOPE_TERMS))


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def detect_language(text: str) -> str:
    """
    Return 'id' or 'en'.

    Defaults to English when there is no evidence either way, because the corpus
    and the analysis prompts are English - an unmarked query is more likely a
    keyword search ("prabowo cabinet") than a sentence in Indonesian.
    """
    words = _words(text)
    if not words:
        return ENGLISH

    id_hits = sum(1 for w in words if w in ID_MARKERS)
    en_hits = sum(1 for w in words if w in EN_MARKERS)

    # Greeting words are the only signal in messages like "halo" or "good morning",
    # which contain no function words at all.
    id_hits += sum(1 for w in words if w in ID_GREETINGS)
    en_hits += sum(1 for w in words if w in EN_GREETINGS)

    if id_hits > en_hits:
        return INDONESIAN
    if en_hits > id_hits:
        return ENGLISH

    # Tie (including 0-0): fall back to distinctive Indonesian affixes that have
    # no English equivalent, e.g. "menjelaskan", "pemilihan", "kebijakan".
    if any(re.search(r"\b(me[mnrl]?\w{4,}|pe[mnrl]?\w{4,}an|ke\w{3,}an)\b", w) for w in words):
        return INDONESIAN

    return ENGLISH


def detect_help_topic(text: str):
    """
    Return the help topic a message is asking about, or None.

    Requires an interface-scoped term to be present, so a question about how
    outlets framed an event never gets mistaken for a question about the
    "Framing Keywords" panel.
    """
    cleaned = text.strip().lower()
    if not cleaned:
        return None

    if not _APP_SCOPE_RE.search(cleaned):
        return None

    for phrases, topic in HELP_PATTERNS:
        if any(re.search(_phrase_pattern(phrase), cleaned) for phrase in phrases):
            return topic

    # Mentions the app but matches no specific panel - give the overview
    return "overview"


def classify_intent(text: str) -> str:
    """
    Classify a message as small talk, a question about the app, or a real
    research question.

    Only clearly conversational or interface-scoped messages are diverted;
    anything else goes through retrieval. Being conservative here matters - a
    misrouted research question would silently skip the whole RAG pipeline.
    """
    cleaned = text.strip().lower().rstrip("!?.,")
    if not cleaned:
        return INTENT_QUESTION

    if detect_help_topic(cleaned):
        return INTENT_HELP

    words = _words(cleaned)

    if any(phrase in cleaned for phrase in THANKS_PHRASES) and len(words) <= 5:
        return INTENT_THANKS

    if any(phrase in cleaned for phrase in FAREWELL_PHRASES) and len(words) <= 4:
        return INTENT_FAREWELL

    if any(phrase in cleaned for phrase in GREETING_PHRASES) and len(words) <= 6:
        return INTENT_GREETING

    # Bare greetings only: "halo", "hi there", "selamat pagi semua"
    if words and len(words) <= 3 and all(w in GREETINGS or len(w) <= 2 for w in words):
        return INTENT_GREETING

    return INTENT_QUESTION


# --- Canned conversational replies -----------------------------------------
# Kept here rather than in prompts.py: these never reach the LLM, so they must
# be deterministic and must not be confused with model output.

SMALL_TALK: Dict[str, Dict[str, str]] = {
    INTENT_GREETING: {
        ENGLISH: (
            "Hello! I'm Sentra, an assistant for analysing how Indonesian news outlets "
            "frame political events.\n\n"
            "I answer only from a corpus of articles from **ANTARANEWS**, **TEMPO**, and "
            "**ABC News** covering the 2024 Indonesian election and its aftermath. If the "
            "articles don't cover something, I'll say so instead of guessing.\n\n"
            "Try asking:\n"
            "- *Who won the 2024 Indonesian election?*\n"
            "- *How did the outlets differ in covering the cabinet formation?*\n"
            "- *What did the Constitutional Court decide?*"
        ),
        INDONESIAN: (
            "Halo! Saya Sentra, asisten untuk menganalisis bagaimana media Indonesia "
            "membingkai peristiwa politik.\n\n"
            "Saya hanya menjawab dari kumpulan artikel **ANTARANEWS**, **TEMPO**, dan "
            "**ABC News** seputar Pemilu 2024 dan dampaknya. Kalau artikelnya tidak "
            "memuat sesuatu, saya akan bilang tidak tahu, bukan mengarang.\n\n"
            "Coba tanyakan:\n"
            "- *Siapa yang memenangkan Pemilu 2024?*\n"
            "- *Bagaimana perbedaan media dalam meliput pembentukan kabinet?*\n"
            "- *Apa putusan Mahkamah Konstitusi soal pemilu?*"
        ),
    },
    INTENT_THANKS: {
        ENGLISH: "You're welcome. Ask me anything else about how the outlets covered the election.",
        INDONESIAN: "Sama-sama. Silakan tanya lagi soal bagaimana media meliput pemilu.",
    },
    INTENT_FAREWELL: {
        ENGLISH: "Goodbye. Come back whenever you want to compare how outlets framed a story.",
        INDONESIAN: "Sampai jumpa. Datang lagi kalau ingin membandingkan framing antar media.",
    },
}


# --- Help topics ------------------------------------------------------------
# Written for a first-time user, not for the developer: plain words, a concrete
# example, and no jargon that isn't immediately explained.

HELP_TOPICS: Dict[str, Dict[str, str]] = {
    "overview": {
        ENGLISH: (
            "**What Sentra does**\n\n"
            "I answer questions about the 2024 Indonesian election using a fixed set of "
            "news articles from **ANTARANEWS**, **TEMPO**, and **ABC News** — and then I "
            "check my own answer against those articles.\n\n"
            "**Main features**\n\n"
            "1. **Grounded answers** — I only use the stored articles. If they don't cover "
            "something, I say so instead of guessing.\n"
            "2. **Media framing comparison** — I show how each outlet covered the same event "
            "differently.\n"
            "3. **Self-fact-checking** — every sentence I write is compared back to the "
            "articles. Sentences I can't support get flagged.\n"
            "4. **Confidence score** — how good the supporting articles were for your question.\n"
            "5. **Two graders side by side** — a trained AI model and a simple rule-based "
            "one, so you can see whether the AI is actually better.\n"
            "6. **Strict mode** — the *Reduce Hallucination* switch makes me far more cautious.\n"
            "7. **Bilingual** — ask in Indonesian or English; I reply in the same language.\n\n"
            "Ask me *\"what is Analysis Results?\"* to learn about the panel on the right."
        ),
        INDONESIAN: (
            "**Apa yang Sentra lakukan**\n\n"
            "Saya menjawab pertanyaan seputar Pemilu Indonesia 2024 memakai kumpulan artikel "
            "tetap dari **ANTARANEWS**, **TEMPO**, dan **ABC News** — lalu saya memeriksa "
            "jawaban saya sendiri terhadap artikel itu.\n\n"
            "**Fitur utama**\n\n"
            "1. **Jawaban berbasis sumber** — saya hanya memakai artikel yang tersimpan. "
            "Kalau tidak ada, saya bilang tidak tahu, bukan mengarang.\n"
            "2. **Perbandingan framing media** — saya tunjukkan bagaimana tiap media meliput "
            "peristiwa yang sama secara berbeda.\n"
            "3. **Pemeriksaan fakta mandiri** — tiap kalimat yang saya tulis dicocokkan "
            "kembali ke artikel. Kalimat yang tidak terdukung akan ditandai.\n"
            "4. **Skor kepercayaan (Confidence)** — seberapa bagus artikel pendukung untuk "
            "pertanyaanmu.\n"
            "5. **Dua penilai berdampingan** — satu model AI terlatih, satu aturan sederhana, "
            "supaya kamu bisa melihat apakah AI-nya memang lebih baik.\n"
            "6. **Mode ketat** — sakelar *Reduce Hallucination* membuat saya jauh lebih hati-hati.\n"
            "7. **Dwibahasa** — tanya dalam bahasa Indonesia atau Inggris, saya jawab dengan "
            "bahasa yang sama.\n\n"
            "Coba tanya *\"apa itu Analysis Results?\"* untuk tahu soal panel di sebelah kanan."
        ),
    },
    "analysis_results": {
        ENGLISH: (
            "**Analysis Results — the panel on the right**\n\n"
            "Think of it as a report card for the answer I just gave you. Instead of asking "
            "you to trust me, it shows you the evidence.\n\n"
            "It has four parts:\n\n"
            "- **Confidence** (the two circles) — how good the supporting articles were. "
            "High means the articles matched your question closely.\n"
            "- **Hallucination Detection** — how many of my sentences could NOT be backed up "
            "by the articles. Shown as `1/9`, meaning 1 out of 9 sentences was unsupported. "
            "**Lower is better.**\n"
            "- **Framing Keywords** — the words each outlet leaned on most, which reveals "
            "what they chose to emphasise.\n"
            "- **Sources** — the actual article snippets I used, so you can read them yourself.\n\n"
            "Each part is measured twice, by **Model A** and **Model B**. Ask me about those "
            "next if you're curious."
        ),
        INDONESIAN: (
            "**Analysis Results — panel di sebelah kanan**\n\n"
            "Anggap saja itu rapor untuk jawaban yang baru saya berikan. Alih-alih meminta "
            "kamu percaya begitu saja, panel ini menunjukkan buktinya.\n\n"
            "Isinya empat bagian:\n\n"
            "- **Confidence** (dua lingkaran) — seberapa bagus artikel pendukungnya. Tinggi "
            "berarti artikelnya cocok dengan pertanyaanmu.\n"
            "- **Hallucination Detection** — berapa kalimat saya yang TIDAK bisa dibuktikan "
            "oleh artikel. Ditampilkan seperti `1/9`, artinya 1 dari 9 kalimat tidak "
            "terdukung. **Makin kecil makin baik.**\n"
            "- **Framing Keywords** — kata-kata yang paling menonjol di tiap media, yang "
            "memperlihatkan apa yang mereka tekankan.\n"
            "- **Sources** — potongan artikel yang benar-benar saya pakai, supaya bisa kamu "
            "baca sendiri.\n\n"
            "Tiap bagian diukur dua kali, oleh **Model A** dan **Model B**. Tanyakan itu "
            "kalau penasaran."
        ),
    },
    "model_ab": {
        ENGLISH: (
            "**Model A vs Model B**\n\n"
            "Two different graders scoring the *same* answer, shown side by side.\n\n"
            "- **Model A — the trained one.** Machine learning models trained for this job: "
            "Logistic Regression for fact-checking, Random Forest for confidence, and a "
            "fine-tuned DistilBERT for writing style.\n"
            "- **Model B — the simple one.** Plain rules with no learning: count matching "
            "words, count how many articles were found.\n\n"
            "**Why show both?** Because \"we used AI\" proves nothing on its own. Putting a "
            "trained model next to a dumb baseline shows whether the training was actually "
            "worth it. On our test set Model A wins clearly — 75% vs 56% accuracy at "
            "fact-checking, and 89% vs 44% at confidence scoring.\n\n"
            "It also keeps me honest: on one task, keyword-based framing analysis, the simple "
            "approach did *not* lose by much — and we report that too."
        ),
        INDONESIAN: (
            "**Model A vs Model B**\n\n"
            "Dua penilai berbeda yang menilai jawaban yang *sama*, ditampilkan berdampingan.\n\n"
            "- **Model A — yang dilatih.** Model machine learning yang dilatih khusus: "
            "Logistic Regression untuk cek fakta, Random Forest untuk skor kepercayaan, dan "
            "DistilBERT yang di-fine-tune untuk mengenali gaya penulisan.\n"
            "- **Model B — yang sederhana.** Aturan biasa tanpa pembelajaran: hitung kata "
            "yang cocok, hitung berapa artikel yang ketemu.\n\n"
            "**Kenapa ditampilkan berdua?** Karena klaim \"kami pakai AI\" tidak membuktikan "
            "apa-apa. Menaruh model terlatih di sebelah pembanding sederhana memperlihatkan "
            "apakah pelatihannya memang berguna. Di data uji kami, Model A menang jelas — "
            "akurasi 75% vs 56% untuk cek fakta, dan 89% vs 44% untuk skor kepercayaan.\n\n"
            "Ini juga menjaga kejujuran: pada satu tugas, analisis framing berbasis kata "
            "kunci, cara sederhana ternyata tidak kalah jauh — dan itu pun kami laporkan."
        ),
    },
    "hallucination": {
        ENGLISH: (
            "**Hallucination Detection — how it works**\n\n"
            "A \"hallucination\" is when an AI states something confidently that its sources "
            "never said. Here's how I catch my own:\n\n"
            "1. I split my answer into individual sentences.\n"
            "2. For each sentence, I search the retrieved articles for backing.\n"
            "3. Sentences with no backing get counted as unsupported.\n\n"
            "The result shows as `1/9` — **1 unsupported out of 9 sentences checked. Lower "
            "is better.** \"Strong Alignment\" means everything checked out; \"Weak "
            "Alignment\" means some claims need verifying.\n\n"
            "**The two methods differ in how they check:**\n\n"
            "- **Model A** compares *meaning*. It turns the sentence and the articles into "
            "numbers that capture meaning, then measures how close they are. It catches a "
            "correct fact even when worded completely differently.\n"
            "- **Model B** compares *words*. It just checks how many words overlap. Simpler, "
            "and easily fooled — a paraphrase looks like a hallucination to it.\n\n"
            "One honest caveat: interpretive sentences like \"this suggests a shift in tone\" "
            "often get flagged even when they're reasonable, because no article states them "
            "literally. I'd rather over-flag than let a real fabrication through."
        ),
        INDONESIAN: (
            "**Hallucination Detection — cara kerjanya**\n\n"
            "\"Halusinasi\" adalah saat AI menyatakan sesuatu dengan yakin padahal sumbernya "
            "tidak pernah mengatakan itu. Begini cara saya menangkap halusinasi saya sendiri:\n\n"
            "1. Jawaban saya dipecah menjadi kalimat-kalimat.\n"
            "2. Tiap kalimat dicari pendukungnya di artikel yang ditemukan.\n"
            "3. Kalimat tanpa pendukung dihitung sebagai tidak terdukung.\n\n"
            "Hasilnya tampil seperti `1/9` — **1 tidak terdukung dari 9 kalimat yang "
            "diperiksa. Makin kecil makin baik.** \"Strong Alignment\" berarti semuanya lolos; "
            "\"Weak Alignment\" berarti ada klaim yang perlu dicek ulang.\n\n"
            "**Dua metodenya berbeda cara memeriksa:**\n\n"
            "- **Model A** membandingkan *makna*. Kalimat dan artikel diubah jadi angka yang "
            "menangkap makna, lalu diukur kedekatannya. Fakta yang benar tetap terdeteksi "
            "walau susunan katanya beda total.\n"
            "- **Model B** membandingkan *kata*. Hanya menghitung berapa kata yang sama. "
            "Lebih sederhana, dan gampang tertipu — parafrase dianggapnya halusinasi.\n\n"
            "Satu catatan jujur: kalimat interpretatif seperti \"ini menunjukkan pergeseran "
            "nada\" sering ikut ditandai meski masuk akal, karena tidak ada artikel yang "
            "menyatakannya secara harfiah. Saya lebih memilih menandai berlebihan daripada "
            "meloloskan karangan sungguhan."
        ),
    },
    "framing": {
        ENGLISH: (
            "**Framing Keywords — what they tell you**\n\n"
            "\"Framing\" is the angle a news outlet chooses. Two outlets can report the exact "
            "same event and still leave you with different impressions, just by choosing "
            "which details to emphasise.\n\n"
            "This panel shows the words each outlet used most distinctively — not the most "
            "common words, but the ones that set that outlet apart from the others. If TEMPO "
            "leans on \"investigation\" while ANTARANEWS leans on \"development\", that "
            "difference is the framing.\n\n"
            "**Model A** predicts *which outlet a text sounds like*, using a neural network "
            "trained on their writing styles. It shows a percentage — \"Style Match: 77%\" "
            "means the text reads a lot like that outlet.\n\n"
            "**Model B** simply lists distinctive keywords using word statistics. Easier to "
            "read, but shallower.\n\n"
            "A caveat worth knowing: this measures *which outlet wrote it*, which is a stand-in "
            "for framing, not framing itself. True frame analysis would need human-labelled "
            "categories like \"conflict\" or \"economic\"."
        ),
        INDONESIAN: (
            "**Framing Keywords — apa artinya**\n\n"
            "\"Framing\" adalah sudut pandang yang dipilih sebuah media. Dua media bisa "
            "memberitakan peristiwa yang persis sama tapi meninggalkan kesan berbeda, hanya "
            "dengan memilih detail mana yang ditonjolkan.\n\n"
            "Panel ini menampilkan kata-kata yang paling khas dipakai tiap media — bukan kata "
            "yang paling sering, tapi kata yang membedakan media itu dari yang lain. Kalau "
            "TEMPO condong ke \"investigasi\" sementara ANTARANEWS condong ke \"pembangunan\", "
            "perbedaan itulah framing-nya.\n\n"
            "**Model A** memperkirakan *teks ini terdengar seperti media mana*, memakai neural "
            "network yang dilatih pada gaya tulisan mereka. Hasilnya persentase — \"Style "
            "Match: 77%\" berarti teksnya sangat mirip gaya media tersebut.\n\n"
            "**Model B** sekadar menampilkan kata kunci khas berdasarkan statistik kata. Lebih "
            "mudah dibaca, tapi lebih dangkal.\n\n"
            "Catatan yang perlu diketahui: ini mengukur *media mana yang menulis*, yang "
            "merupakan pengganti untuk framing, bukan framing itu sendiri. Analisis framing "
            "sesungguhnya butuh kategori berlabel manusia seperti \"konflik\" atau \"ekonomi\"."
        ),
    },
    "confidence": {
        ENGLISH: (
            "**Confidence Score — what the percentage means**\n\n"
            "It answers one question: *how good were the articles I found for your question?* "
            "It is **not** how sure I feel about being right.\n\n"
            "It goes up when the articles closely match your question and several outlets "
            "cover it; it goes down when the matches are weak or inconsistent.\n\n"
            "- **Model A (Random Forest)** was trained on retrieval quality signals and can "
            "give a *low* score even when many articles were found, if they're inconsistent.\n"
            "- **Model B (Heuristic)** mostly counts: more articles, more sources, higher "
            "score. It's often over-optimistic.\n\n"
            "So when Model A reads lower than Model B, that's usually Model A noticing "
            "something Model B missed. Treat a low score as a hint to read the Sources panel "
            "yourself."
        ),
        INDONESIAN: (
            "**Confidence Score — arti persentasenya**\n\n"
            "Ini menjawab satu pertanyaan: *seberapa bagus artikel yang saya temukan untuk "
            "pertanyaanmu?* Ini **bukan** seberapa yakin saya bahwa jawaban saya benar.\n\n"
            "Nilainya naik kalau artikelnya cocok dengan pertanyaanmu dan beberapa media "
            "membahasnya; turun kalau kecocokannya lemah atau tidak konsisten.\n\n"
            "- **Model A (Random Forest)** dilatih pada sinyal kualitas pencarian dan bisa "
            "memberi nilai *rendah* meski artikelnya banyak, kalau isinya tidak konsisten.\n"
            "- **Model B (Heuristic)** kebanyakan cuma menghitung: makin banyak artikel dan "
            "sumber, makin tinggi nilainya. Sering terlalu optimistis.\n\n"
            "Jadi kalau Model A lebih rendah dari Model B, biasanya Model A menangkap sesuatu "
            "yang dilewatkan Model B. Anggap nilai rendah sebagai isyarat untuk membaca "
            "sendiri panel Sources."
        ),
    },
    "modes": {
        ENGLISH: (
            "**Reduce Hallucination — the switch at the bottom**\n\n"
            "It makes me much more cautious. With it on:\n\n"
            "- I need stronger evidence before I'll answer at all, and refuse more often.\n"
            "- I use fewer article snippets, but only the closest-matching ones.\n"
            "- I stick to facts stated literally in the sources and avoid interpretation.\n"
            "- I need at least three words in your question, so I know what to search for.\n\n"
            "**When to use it:** when you need to quote something reliably.\n"
            "**When to leave it off:** when you want a broader comparison of how outlets "
            "framed a story — that discussion needs a bit of interpretation."
        ),
        INDONESIAN: (
            "**Reduce Hallucination — sakelar di bagian bawah**\n\n"
            "Sakelar ini membuat saya jauh lebih hati-hati. Kalau dinyalakan:\n\n"
            "- Saya butuh bukti lebih kuat sebelum mau menjawab, jadi lebih sering menolak.\n"
            "- Saya memakai lebih sedikit potongan artikel, tapi hanya yang paling cocok.\n"
            "- Saya berpegang pada fakta yang tertulis harfiah di sumber dan menghindari "
            "penafsiran.\n"
            "- Pertanyaanmu minimal tiga kata, supaya saya tahu apa yang harus dicari.\n\n"
            "**Kapan dipakai:** kalau kamu perlu mengutip sesuatu secara andal.\n"
            "**Kapan dimatikan:** kalau kamu ingin perbandingan yang lebih luas tentang cara "
            "media membingkai suatu cerita — pembahasan itu butuh sedikit penafsiran."
        ),
    },
    "sources": {
        ENGLISH: (
            "**Sources — the list at the bottom right**\n\n"
            "These are the actual article snippets I used to build the answer, with the outlet "
            "name and how closely each one matched your question.\n\n"
            "They're there so you never have to take my word for it: if a claim looks "
            "surprising, open the sources and check. When this panel says *(0)*, I found "
            "nothing relevant — and in that case I refuse to answer rather than guess.\n\n"
            "One caveat: the articles are stored snapshots, so the original web pages may have "
            "changed since."
        ),
        INDONESIAN: (
            "**Sources — daftar di kanan bawah**\n\n"
            "Ini potongan artikel yang benar-benar saya pakai untuk menyusun jawaban, lengkap "
            "dengan nama media dan seberapa cocok masing-masing dengan pertanyaanmu.\n\n"
            "Ada di situ supaya kamu tidak perlu percaya begitu saja: kalau ada klaim yang "
            "terasa mengejutkan, buka sumbernya dan periksa. Kalau panel ini menampilkan "
            "*(0)*, berarti saya tidak menemukan apa pun yang relevan — dan dalam kondisi itu "
            "saya memilih menolak menjawab daripada mengarang.\n\n"
            "Satu catatan: artikelnya adalah salinan tersimpan, jadi halaman aslinya mungkin "
            "sudah berubah."
        ),
    },
}


def help_reply(topic: str, language: str) -> str:
    return HELP_TOPICS[topic].get(language, HELP_TOPICS[topic][ENGLISH])


# --- Refusal messages -------------------------------------------------------
# The old strict-mode refusal was the bare string "Insufficient source coverage.",
# which tells the user nothing about what to do next.

REFUSALS: Dict[str, Dict[str, str]] = {
    "no_sources": {
        ENGLISH: (
            "I couldn't find anything about that in the news database.\n\n"
            "My corpus only covers the 2024 Indonesian election and its aftermath, from "
            "ANTARANEWS, TEMPO, and ABC News."
        ),
        INDONESIAN: (
            "Saya tidak menemukan apa pun tentang itu di basis data berita.\n\n"
            "Korpus saya hanya memuat Pemilu Indonesia 2024 dan dampaknya, dari "
            "ANTARANEWS, TEMPO, dan ABC News."
        ),
    },
    "low_similarity": {
        ENGLISH: (
            "I can't answer that from the available sources - the retrieved articles "
            "aren't relevant enough, and I'd rather refuse than guess.\n\n"
            "Try naming a specific event, person, or institution from the 2024 election "
            "(for example: the cabinet appointments, the Constitutional Court ruling, or "
            "voter turnout)."
        ),
        INDONESIAN: (
            "Saya tidak bisa menjawab itu dari sumber yang ada - artikel yang ditemukan "
            "kurang relevan, dan saya memilih menolak daripada mengarang.\n\n"
            "Coba sebutkan peristiwa, tokoh, atau lembaga yang spesifik dari Pemilu 2024 "
            "(misalnya: penunjukan kabinet, putusan Mahkamah Konstitusi, atau angka "
            "partisipasi pemilih)."
        ),
    },
    "low_similarity_strict": {
        ENGLISH: (
            "Strict mode: insufficient source coverage to answer this safely.\n\n"
            "Strict mode applies a higher evidence bar. Turn off *Reduce Hallucination* "
            "for a broader answer, or ask something more specific."
        ),
        INDONESIAN: (
            "Mode ketat: cakupan sumber tidak cukup untuk menjawab dengan aman.\n\n"
            "Mode ketat memakai ambang bukti yang lebih tinggi. Matikan *Reduce "
            "Hallucination* untuk jawaban yang lebih luas, atau ajukan pertanyaan yang "
            "lebih spesifik."
        ),
    },
    "too_short": {
        ENGLISH: (
            "That question is too short for strict mode. Please ask with at least three "
            "words so I can retrieve the right articles."
        ),
        INDONESIAN: (
            "Pertanyaan itu terlalu pendek untuk mode ketat. Tuliskan minimal tiga kata "
            "agar saya bisa mencari artikel yang tepat."
        ),
    },
    # Not a refusal to answer, but a failure to reach the model server. Worded so
    # the user can tell the difference: nothing is wrong with their question, and
    # the sources below the message were still retrieved successfully.
    "llm_unavailable": {
        ENGLISH: (
            "I found relevant articles, but the language model that writes the answer "
            "is not responding, so I cannot produce one right now.\n\n"
            "This is a server problem, not a problem with your question. The retrieved "
            "sources are listed in the panel; try again in a moment."
        ),
        INDONESIAN: (
            "Saya menemukan artikel yang relevan, tetapi model bahasa yang menyusun "
            "jawaban tidak merespons, jadi saya belum bisa membuatnya sekarang.\n\n"
            "Ini masalah server, bukan masalah pada pertanyaan Anda. Sumber yang "
            "ditemukan tetap tercantum di panel; silakan coba lagi sebentar."
        ),
    },
}


# --- Disclaimers appended to a generated answer -----------------------------
# These must match the answer's language, otherwise a reply in Indonesian ends
# with an English footer.

DISCLAIMERS: Dict[str, Dict[str, str]] = {
    "academic": {
        ENGLISH: (
            "\n\n---\n*This analysis focuses on media framing and emphasis. Some conclusions "
            "are interpretive statements and depend on the scope of retrieved articles.*"
        ),
        INDONESIAN: (
            "\n\n---\n*Analisis ini berfokus pada framing dan penekanan media. Sebagian "
            "kesimpulan bersifat interpretatif dan bergantung pada cakupan artikel yang "
            "ditemukan.*"
        ),
    },
    "moderate": {
        ENGLISH: "\n\n---\n*Note: Contains some Framing Inferences not explicitly in source text.*",
        INDONESIAN: (
            "\n\n---\n*Catatan: Memuat sebagian inferensi framing yang tidak dinyatakan "
            "eksplisit dalam teks sumber.*"
        ),
    },
    "claims_header": {
        ENGLISH: "\n\n---\n**Interpretive Claims with Limited Source Support:**\n",
        INDONESIAN: "\n\n---\n**Klaim Interpretatif dengan Dukungan Sumber Terbatas:**\n",
    },
    "claims_footer": {
        ENGLISH: (
            "\n*Interpretive statements (Framing Inferences) are evaluated conservatively "
            "and may be flagged despite being reasonable in context.*"
        ),
        INDONESIAN: (
            "\n*Pernyataan interpretatif (inferensi framing) dinilai secara konservatif "
            "dan bisa saja ditandai meskipun masuk akal dalam konteksnya.*"
        ),
    },
}

# Claim categories shown when verification is low. Indexed by a stable key so the
# detection logic stays language-independent.
CLAIM_CATEGORIES: Dict[str, Dict[str, str]] = {
    "jokowi_role": {
        ENGLISH: "Role of President Jokowi in the process",
        INDONESIAN: "Peran Presiden Jokowi dalam proses tersebut",
    },
    "opposition": {
        ENGLISH: "Acceptance by former opposition figures",
        INDONESIAN: "Penerimaan oleh tokoh oposisi sebelumnya",
    },
    "campaign": {
        ENGLISH: "Attribution of campaign success factors",
        INDONESIAN: "Atribusi faktor keberhasilan kampanye",
    },
    "cabinet": {
        ENGLISH: "Cabinet appointment decisions",
        INDONESIAN: "Keputusan penunjukan kabinet",
    },
    "coalition": {
        ENGLISH: "Political coalition dynamics",
        INDONESIAN: "Dinamika koalisi politik",
    },
    "economic": {
        ENGLISH: "Economic implications and outlook",
        INDONESIAN: "Implikasi dan prospek ekonomi",
    },
    "diplomatic": {
        ENGLISH: "Diplomatic and foreign relations",
        INDONESIAN: "Hubungan diplomatik dan luar negeri",
    },
    "military": {
        ENGLISH: "Military and security aspects",
        INDONESIAN: "Aspek militer dan keamanan",
    },
    "social": {
        ENGLISH: "Social program initiatives",
        INDONESIAN: "Inisiatif program sosial",
    },
    "legal": {
        ENGLISH: "Constitutional and legal matters",
        INDONESIAN: "Persoalan konstitusional dan hukum",
    },
    "generic": {
        ENGLISH: "General interpretive framing statements",
        INDONESIAN: "Pernyataan framing interpretatif umum",
    },
}


# --- Section headings used inside the prompt templates ----------------------
# Telling a 3B model to "write headings in Indonesian" does not work: it copies
# the literal headings it can see in the template. So the template itself is
# rendered with headings already in the target language.

HEADINGS: Dict[str, Dict[str, str]] = {
    "direct_answer": {ENGLISH: "Direct Answer", INDONESIAN: "Jawaban Langsung"},
    "summary": {ENGLISH: "Summary", INDONESIAN: "Ringkasan"},
    "media_coverage": {
        ENGLISH: "Media Coverage by Source",
        INDONESIAN: "Liputan Media per Sumber",
    },
    "key_themes": {
        ENGLISH: "Key themes reported",
        INDONESIAN: "Tema utama yang diberitakan",
    },
    "framing_emphasis": {
        ENGLISH: "Framing emphasis",
        INDONESIAN: "Penekanan framing",
    },
    "comparative": {
        ENGLISH: "Comparative Framing Analysis",
        INDONESIAN: "Analisis Framing Komparatif",
    },
    "factual_summary": {
        ENGLISH: "Factual Summary (Source-Aligned)",
        INDONESIAN: "Ringkasan Faktual (Sesuai Sumber)",
    },
    "interpretive": {
        ENGLISH: "Interpretive Analysis (Limited)",
        INDONESIAN: "Analisis Interpretatif (Terbatas)",
    },
    "see_below": {
        ENGLISH: "See analysis below.",
        INDONESIAN: "Lihat analisis di bawah.",
    },
    "no_explicit_framing": {
        ENGLISH: "No explicit framing analysis found in sources.",
        INDONESIAN: "Tidak ditemukan analisis framing eksplisit dalam sumber.",
    },
}


def headings(language: str) -> Dict[str, str]:
    """All section headings for one language, keyed for prompt formatting."""
    return {
        f"h_{key}": translations.get(language, translations[ENGLISH])
        for key, translations in HEADINGS.items()
    }


def small_talk_reply(intent: str, language: str) -> str:
    return SMALL_TALK[intent].get(language, SMALL_TALK[intent][ENGLISH])


def refusal(kind: str, language: str) -> str:
    return REFUSALS[kind].get(language, REFUSALS[kind][ENGLISH])


def disclaimer(kind: str, language: str) -> str:
    return DISCLAIMERS[kind].get(language, DISCLAIMERS[kind][ENGLISH])


def claim_category(key: str, language: str) -> str:
    return CLAIM_CATEGORIES[key].get(language, CLAIM_CATEGORIES[key][ENGLISH])
