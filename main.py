import os, sys, json, datetime as dt
import feedparser, requests, yaml
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
from rapidfuzz import fuzz, process
from email.utils import format_datetime
from pathlib import Path
from zoneinfo import ZoneInfo  # stdlib tz support

# ---------- Config ----------
ELEVEN_API_KEY  = os.getenv("ELEVEN_API_KEY", "").strip()
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o").strip()   # your secret overrides this
MAX_ITEMS       = int(os.getenv("MAX_ITEMS", "12"))

PUBLIC_DIR = Path("public")
EP_DIR     = PUBLIC_DIR / "episodes"
SH_NOTES   = PUBLIC_DIR / "shownotes"
DEBUG_DIR  = PUBLIC_DIR / "debug"
for d in (PUBLIC_DIR, EP_DIR, SH_NOTES, DEBUG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Load feeds ----------
with open("feeds.yml", "r", encoding="utf-8") as f:
    feeds_cfg = yaml.safe_load(f) or {}
SOURCES    = feeds_cfg.get("sources", [])
EXCLUDE    = set(str(k).lower() for k in feeds_cfg.get("exclude_keywords", []))
LIMIT_PER  = int(feeds_cfg.get("daily_limit_per_source", 6))

# ---------- Helpers ----------
def is_newsworthy(title: str) -> bool:
    t = (title or "").lower()
    return t and not any(k in t for k in EXCLUDE)

def fetch_items():
    items = []
    for src in SOURCES:
        name, rss = src.get("name","Unknown"), src.get("rss","")
        if not rss:
            continue
        try:
            fp = feedparser.parse(rss)
            count = 0
            for e in fp.entries:
                if count >= LIMIT_PER: break
                title = (e.get("title") or "").strip()
                link  = (e.get("link") or "").strip()
                if not title or not link: continue
                if not is_newsworthy(title): continue
                items.append({"source": name, "title": title, "link": link})
                count += 1
        except Exception as ex:
            print(f"[warn] feed error {name}: {ex}", file=sys.stderr)
    return items

def dedupe(items, threshold=90):
    kept, seen = [], []
    for it in items:
        title = it["title"]
        if not seen:
            kept.append(it); seen.append(title); continue
        match = process.extractOne(title, seen, scorer=fuzz.token_set_ratio)
        if not match or match[1] < threshold:
            kept.append(it); seen.append(title)
    return kept

def extract_text(url: str) -> str:
    # 1) trafilatura
    try:
        downloaded = trafilatura.fetch_url(url, timeout=20)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted and len(extracted.split()) > 40:
                return extracted
    except Exception:
        pass
    # 2) readability
    try:
        html = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"}).text
        doc = Document(html)
        cleaned = doc.summary()
        text = BeautifulSoup(cleaned, "html.parser").get_text("\n")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        lines = [l for l in lines if len(l.split()) > 4]
        return "\n".join(lines)
    except Exception:
        return ""

def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    for sep in [". ", " — ", " – ", " • "]:
        if sep in text:
            cand = text.split(sep)[0]
            if len(cand.split()) >= 8:
                return cand.strip(".•–— ")
    return text[:240].rsplit(" ",1)[0]

def build_notes(items):
    """Short factual notes with attribution + link. GPT will rewrite naturally."""
    notes = []
    used = 0
    for it in items:
        if used >= MAX_ITEMS: break
        txt = extract_text(it["link"])
        if not txt:
            continue
        sent = first_sentence(txt)
        if len(sent.split()) < 6:
            continue
        notes.append(f"{it['source']}: {sent}  (link: {it['link']})")
        used += 1
    return notes

def boston_now():
    """Return Boston local time and greeting bucket."""
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    hour = now.hour
    if 5 <= hour < 12:
        tod = "morning"
    elif 12 <= hour < 18:
        tod = "afternoon"
    else:
        tod = "evening"
    # ‘Monday, August 11, 2025’ format without %-d for Windows compatibility
    pretty_date = now.strftime("%A, %B ") + str(int(now.strftime("%d"))) + now.strftime(", %Y")
    return now, tod, pretty_date

# ---------- OpenAI ----------
try:
    from openai import OpenAI
    _client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception as e:
    print(f"[warn] openai import failed: {e}", file=sys.stderr)
    _client = None

def _responses_api(prompt_text: str, notes: list[str], model: str) -> str:
    """Use Responses API for gpt-5* models — NO temperature here to avoid 400s."""
    now, tod, pretty_date = boston_now()
    control = (
        "HARD CONSTRAINTS (do not violate):\n"
        f"- Time-of-day greeting MUST be: 'Good {tod}, it’s {pretty_date}.'\n"
        "- Lead with the most important news; do NOT lead with sports unless it is indisputably the top story.\n"
        "- Absolutely no editorializing, sympathy, or sentiment (no 'thoughts and prayers', 'we hope', etc.).\n"
        "- Integrate source names naturally in the sentence (e.g., 'The Globe reports…', 'Boston.com says…', 'B-Side notes…').\n"
        "- 5–8 items; smooth transitions; end with quick weather + notable events, then the disclosure.\n"
    )
    user_block = "STORIES (verbatim notes, may be messy):\n" + "\n\n".join(notes)
    full_input = f"{control}\n\nUSER PROMPT:\n{prompt_text.strip()}\n\n{user_block}"

    resp = _client.responses.create(
        model=model,
        input=full_input,
        # temperature intentionally omitted for compatibility
        max_output_tokens=2000,
    )
    return (getattr(resp, "output_text", None) or "").strip()

def _chat_api(prompt_text: str, notes: list[str], model: str) -> str:
    """Use Chat Completions for gpt-4* (supports temperature/max_tokens)."""
    now, tod, pretty_date = boston_now()
    control = (
        "HARD CONSTRAINTS (do not violate):\n"
        f"- Time-of-day greeting MUST be: 'Good {tod}, it’s {pretty_date}.'\n"
        "- Lead with the most important news; do NOT lead with sports unless it is indisputably the top story.\n"
        "- Absolutely no editorializing, sympathy, or sentiment (no 'thoughts and prayers', 'we hope', etc.).\n"
        "- Integrate source names naturally in the sentence (e.g., 'The Globe reports…', 'Boston.com says…', 'B-Side notes…').\n"
        "- 5–8 items; smooth transitions; end with quick weather + notable events, then the disclosure.\n"
    )
    user_block = "STORIES (verbatim notes, may be messy):\n" + "\n\n".join(notes)
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":control},
            {"role":"user","content":f"{prompt_text.strip()}\n\n{user_block}"},
        ],
        temperature=0.65,    # a little more life, still factual
        max_tokens=2000,     # give it headroom
        presence_penalty=0.15,
        frequency_penalty=0.35,
    )
    return resp.choices[0].message.content.strip()

def rewrite_with_openai(prompt_text: str, notes: list[str]) -> str | None:
    if not _client or not OPENAI_MODEL:
        return None
    try:
        if OPENAI_MODEL.lower().startswith("gpt-5"):
            return _responses_api(prompt_text, notes, OPENAI_MODEL)
        else:
            return _chat_api(prompt_text, notes, OPENAI_MODEL)
    except Exception as e:
        print(f"[warn] OpenAI error: {e}", file=sys.stderr)
        # Fallback to gpt-4o so runs still succeed
        try:
            return _chat_api(prompt_text, notes, "gpt-4o")
        except Exception as e2:
            print(f"[warn] OpenAI fallback failed: {e2}", file=sys.stderr)
            return None

# ---------- ElevenLabs (kept compatible) ----------
def tts_elevenlabs(text: str) -> bytes | None:
    if not ELEVEN_API_KEY or not ELEVEN_VOICE_ID or not text.strip():
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.55,          # smoother/steadier phrasing
            "similarity_boost": 0.85,   # keep timbre close without overconstraining
            "style": 0.40,              # light emphasis, avoids drawn-out syllables
            "use_speaker_boost": True
        },
        "voice_speed": 1.05,           # mild pace-up to avoid rushy catch-ups
        "model_id": "eleven_multilingual_v2"
    }
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json"
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    return r.content

# ---------- Output ----------
def write_shownotes(date_str, items):
    html = ["<html><head><meta charset='utf-8'><title>Boston Briefing – Sources</title></head><body>"]
    html.append(f"<h2>Boston Briefing – {date_str}</h2>")
    html.append("<ol>")
    take = 0
    for it in items:
        if take >= MAX_ITEMS: break
        html.append(f"<li><a href='{it['link']}' target='_blank' rel='noopener'>{it['title']}</a> – {it['source']}</li>")
        take += 1
    html.append("</ol></body></html>")
    (SH_NOTES / f"{date_str}.html").write_text("\n".join(html), encoding="utf-8")

def write_index():
    url = f"{PUBLIC_BASE_URL}/feed.xml" if PUBLIC_BASE_URL else "feed.xml"
    shownotes_base = (PUBLIC_BASE_URL or ".").rstrip("/")
    lines = [
        "<html><head><meta charset='utf-8'><title>Boston Briefing</title></head>",
        "<body>",
        "  <h1>Boston Briefing</h1>",
        f'  <p>Podcast RSS: <a href="{url}">{url}</a></p>',
        f'  <p>Shownotes: <a href="{shownotes_base}/shownotes/">Open folder</a></p>',
        "</body></html>",
        "",
    ]
    (PUBLIC_DIR / "index.html").write_text("\n".join(lines), encoding="utf-8")
