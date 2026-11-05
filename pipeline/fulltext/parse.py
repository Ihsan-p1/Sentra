"""
Parse and clean saved Antara and Tempo article pages for the full-text corpus.

parse_article() pulls the title, date, byline and raw body paragraphs out of one
page; clean_paragraphs() drops datelines and boilerplate lines, so no model can
guess the outlet from them (spec section 7); to_ascii() normalises typographic
punctuation, which would otherwise split TF-IDF term counts.
"""
import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, NavigableString

OUTLETS = ("antaranews", "tempo")

# A dateline opens the first paragraph: "Surabaya, E Java (ANTARA) - ", "TEMPO.CO, Jakarta- ".
DATELINES = {
    "antaranews": re.compile(r"^[^()]{0,60}\(ANTARA\)\s*[-–—]\s*"),
    "tempo": re.compile(r"^TEMPO\.CO,[^-–—]{0,40}[-–—]\s*"),
}

# One rule per line; a paragraph matching any of them is dropped whole.
BOILERPLATE = [
    re.compile(r"^Click here to get the latest news updates from Tempo (on|in) Google News"),
    re.compile(r"^Editor['’]s Choice\s*:"),
    re.compile(r"^Read the Complete Story in Tempo English Magazine"),
    re.compile(r"^(Read also|Also read|Related news)\s*:", re.IGNORECASE),
    re.compile(r"^(Editor|Translator|Reporter)\s*:"),
    re.compile(r"^Reporter\s.+\d{4}\s+\d{1,2}:\d{2}\s*[ap]m$", re.IGNORECASE),
    re.compile(r"^Copyright\b.*ANTARA"),
]

# Tempo closes the body with the reporters' names ("ANTARA | DANIEL A. FAJRI", "Adil Al Hasan"):
# a short last paragraph with no closing punctuation, unlike a sentence.
TEMPO_CREDIT_MAX_WORDS = 8
SENTENCE_END = (".", "!", "?", '"', "'", "”", "’", ")")

ASCII_MAP = str.maketrans({
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-", "‑": "-",
    " ": " ",
})


def to_ascii(text: str) -> str:
    return text.translate(ASCII_MAP)


def _meta(soup: BeautifulSoup, prop: str) -> Optional[str]:
    tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
    return tag.get("content") if tag else None


def _iso_date(soup: BeautifulSoup) -> Optional[str]:
    value = _meta(soup, "article:published_time")
    match = re.match(r"\d{4}-\d{2}-\d{2}", value or "")
    return match.group(0) if match else None


def _title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else (_meta(soup, "og:title") or "")


def _br_paragraphs(container) -> List[str]:
    """Antara separates paragraphs with <br/>, not <p>: collapse source whitespace, split on <br>."""
    for text in container.find_all(string=True):
        text.replace_with(NavigableString(re.sub(r"\s+", " ", text)))
    for br in container.find_all("br"):
        br.replace_with("\n")
    return [line.strip() for line in container.get_text().split("\n") if line.strip()]


def _parse_antara(soup: BeautifulSoup) -> Dict:
    body = soup.find("div", class_="post-content")
    author = ""
    if body is None:
        return {"author": author, "paragraphs": []}
    credits = body.find("p", class_="text-muted")
    if credits:
        match = re.search(r"Translator:\s*([^\n]+?)\s*(Editor:|$)", credits.get_text("\n"))
        author = match.group(1).strip() if match else ""
    # Pull quotes repeat a body sentence; related-news boxes and credits are not article text.
    for tag in body.find_all(["blockquote", "script", "style"]) + body.find_all(
            "span", class_="baca-juga") + body.find_all("p", class_="text-muted"):
        tag.decompose()
    return {"author": author, "paragraphs": _br_paragraphs(body)}


def _inline_text(tag) -> str:
    # Join inline tags without a separator so "<b>TEMPO.CO</b>, Jakarta" stays "TEMPO.CO, Jakarta".
    return re.sub(r"\s+", " ", tag.get_text()).strip()


def _parse_tempo(soup: BeautifulSoup) -> Dict:
    # The page appends "next" articles (class "berikutnya") that carry their own dateline.
    main = next((a for a in soup.find_all("article", class_="detail-artikel")
                 if "berikutnya" not in a.get("class", [])), soup)
    author = ""
    for credit in main.select("div.block-avatar div.text-avatar"):
        label, name = credit.find("p", class_="margin-bottom-ss"), credit.find("p", class_="bold")
        if label and name and label.get_text(strip=True) == "Translator":
            author = name.get_text(" ", strip=True)
    # The body is the parent of the <p> that opens with the dateline (recon 2026-10-29).
    first = next((p for p in main.find_all("p") if _inline_text(p).startswith("TEMPO.CO")), None)
    if first is None:
        return {"author": author, "paragraphs": []}
    paragraphs = [_inline_text(p) for p in first.parent.find_all("p", recursive=False)]
    return {"author": author, "paragraphs": [p for p in paragraphs if p]}


def parse_article(html: str, outlet: str) -> Dict:
    if outlet not in OUTLETS:
        raise ValueError(f"unknown outlet {outlet!r}; expected one of {OUTLETS}")
    soup = BeautifulSoup(html, "html.parser")
    parsed = _parse_antara(soup) if outlet == "antaranews" else _parse_tempo(soup)
    return {
        "title": _title(soup),
        "author": parsed["author"],
        "published_date": _iso_date(soup),
        "paragraphs": parsed["paragraphs"],
    }


def clean_paragraphs(paragraphs: List[str], outlet: str) -> List[str]:
    if outlet not in OUTLETS:
        raise ValueError(f"unknown outlet {outlet!r}; expected one of {OUTLETS}")
    kept = [p.strip() for p in paragraphs
            if p.strip() and not any(rule.search(p.strip()) for rule in BOILERPLATE)]
    if (outlet == "tempo" and len(kept) > 1 and not kept[-1].endswith(SENTENCE_END)
            and len(kept[-1].split()) <= TEMPO_CREDIT_MAX_WORDS):
        kept.pop()
    if kept:
        kept[0] = DATELINES[outlet].sub("", kept[0], count=1)
    return [p for p in kept if p]
