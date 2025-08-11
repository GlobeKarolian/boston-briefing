import os, sys, io, json, datetime as dt
from zoneinfo import ZoneInfo
import feedparser, requests, yaml
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
from rapidfuzz import fuzz, process
from email.utils import format_datetime

# ---------- Config via env ----------
ELEVEN_API_KEY  = os.getenv("ELEVEN_API_KEY", "").strip()
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
# Use whatever you set in repo secret OPENAI_MODEL (e.g., "gpt-4o" or your preferred model)
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o").strip()

MAX_ITEMS       = int(os.getenv("MAX_ITEMS", "12"))

# ---------- Paths ----------
PUBLIC_DIR = "public"
EP_DIR     = os.path.join(PUBLIC_DIR, "episodes")
os.makedirs(PUBLIC_DIR, exist_ok=True)
os.makedirs(EP_DIR, exist_ok=True)

# ---------- Load feeds config ----------
with open("feeds.yml", "r", encoding="utf-8") as f:
    feeds_cfg = yaml.safe_load(f)

SOURCES    = feeds_cfg.get("sources", [])
EXCLUDE    = set(k.lower() for k in feeds_cfg.get("exclude_keywords", []))
LIMIT_PER  = int(feeds_cfg.get("daily_limit_per_source", 6))

def is_newsworthy(title: str) -> bool:
    t = title.lower()
    return not any(k in t for k in EXCLUDE)

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
                if count >= LIMIT_PER:
                    break
                title = (e.get("title") or "").strip()
                link  = (e.get("link") or "").strip()
                if not title or not link:
                    continue
                if not is_newsworthy(title):
                    continue
                items.append({"source": name, "title": title, "link": link})
                count += 1
        except Exception as ex:
            print(f"[warn] feed error {name}: {ex}", file=sys.stderr)
    return items

def dedupe(items, threshold=90):
    kept, seen = [], []
    for it in items:
        if not seen:
            kept.append(it); seen.append(it["title"]); continue
        match = process.extractOne(it["title"], seen, scorer=fuzz.token_set_ratio)
        if not match or match[1] < threshold:
            kept.append(it); seen.append(it["title"])
    return kept

def extract_text(url: str) -> str:
    # Try trafilatura
    try:
        downloaded = trafilatura.fetch_url(url, timeout=15)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted and len(extracted.split()) > 40:
                return extracted
    except Exception:
        pass
    # Fallback: readability first paragraphs
    try:
        html = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"}).text
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
    # Try common sentence or clause breaks
    for sep in [". ", "? ", "! ", " — ", " – ", " • "]:
        if sep in text:
            cand = text.split(sep)[0]
            if len(cand.split()) >= 8:
                return cand.strip(".•–— ")
    # Fallback: truncate
    return text[:240].rsplit(" ",1)[0]

def load_prompt(base_prompt_path="prompt.txt", date_label="", tod_label=""):
    try:
        with open(base_prompt_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        # Basic guard if file missing
        raw = (
            "Write a concise script for a short factual Boston news audio briefing. "
            "No opinions. Attribute naturally to sources. Start with a natural greeting for {TOD}. "
            "Date: {DATE}. Use most-important-first ordering. End with a brief disclosure."
        )
    return raw.replace("{DATE}", date_label).replace("{TOD}", tod_label)

def notes_block(items):
    # Build a compact notes section with (source) title + first sentence from article
    lines = []
    used = 0
    for it in items:
        if used >= MAX_ITEMS:
            break
        body = extract_text(it["link"])
        if not body:
            continue
        lead = first_sentence(body)
        if len(lead.split()) < 6:
            continue
        # Keep it factual and compact; model will smooth and order
        lines.append(f"- {it['title']} — Source: {it['source']} — First line: {lead}")
        used += 1
    return "\n".join(lines), used

# -------------- OpenAI (Responses API) --------------
def draft_with_openai(prompt_text: str, notes: str) -> str:
    if not OPENAI_API_KEY:
        print("[warn] No OPENAI_API_KEY; skipping LLM script", file=sys.stderr)
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Compose a single user message (prompt + notes)
        user_input = f"{prompt_text}\n\n# Story notes\n{notes}\n"
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[{"role": "user", "content": user_input}],
            max_output_tokens=1200  # leave temperature out (not supported by some models)
        )
        return (resp.output_text or "").strip()
    except Exception as ex:
        print(f"[warn] OpenAI error: {ex}", file=sys.stderr)
        return ""

# -------------- ElevenLabs TTS --------------
def tts_elevenlabs(text: str, attempt_label="main") -> bytes | None:
    if not ELEVEN_API_KEY or not ELEVEN_VOICE_ID:
        print("[warn] ElevenLabs not configured", file=sys.stderr)
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    payload = {
        "text": text,
        "voice_settings": {
            # Tuned to reduce stretched vowels and rushed catch-ups:
            # slightly lower stability, moderate similarity, mild style,
            # and disable speaker boost which can exaggerate dynamics.
            "stability": 0.25,
            "similarity_boost": 0.7,
            "style": 0.25,
            "use_speaker_boost": False
        },
        "model_id": "eleven_multilingual_v2"
    }
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json"
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        return r.content
    except Exception as ex:
        print(f"[warn] ElevenLabs {attempt_label} error: {ex}", file=sys.stderr)
        return None

def save_bytes(path, data: bytes):
    with open(path, "wb") as f:
        f.write(data)

# -------------- Site output --------------
def write_shownotes(date_str, items):
    html = ["<html><head><meta charset='utf-8'><title>Boston Briefing – Sources</title></head><body>"]
    html.append(f"<h2>Boston Briefing – {date_str}</h2>")
    html.append("<ol>")
    for it in items[:MAX_ITEMS]:
        html.append(f"<li><a href='{it['link']}' target='_blank' rel='noopener'>{it['title']}</a> – {it['source']}</li>")
    html.append("</ol></body></html>")
    notes_dir = os.path.join(PUBLIC_DIR, "shownotes")
    os.makedirs(notes_dir, exist_ok=True)
    with open(os.path.join(notes_dir, f"{date_str}.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))

def write_index():
    html = f"""<html><head><meta charset='utf-8'><title>Boston Briefing</title></head>
<body>
  <h1>Boston Briefing</h1>
  <p><strong>Internal beta — do not share externally.</strong></p>
  <p>Podcast RSS: <a href="{PUBLIC_BASE_URL}/feed.xml">{PUBLIC_BASE_URL}/feed.xml</a></p>
  <p>Shownotes: <a href="{PUBLIC_BASE_URL}/shownotes/">Open folder</a></p>
</body></html>"""
    with open(os.path.join(PUBLIC_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

def build_feed(episode_url: str, pub_dt: dt.datetime, filesize: int):
    title = "Boston Briefing"
    desc = "A short, factual Boston news briefing."
    link = PUBLIC_BASE_URL
    last_build = format_datetime(pub_dt)

    item_title = pub_dt.strftime("Boston Briefing – %Y-%m-%d")
    guid = episode_url or item_title

    # Build enclosure separately to avoid nested f-string/escape issues
    enclosure = ""
    if episode_url:
        enclosure = f'<enclosure url="{episode_url}" length="{filesize}" type="audio/mpeg"/>'

    feed = (
f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>{title}</title>
    <link>{link}</link>
    <language>en-us</language>
    <description>{desc}</description>
    <itunes:author>Boston Briefing</itunes:author>
    <itunes:explicit>false</itunes:explicit>
    <lastBuildDate>{last_build}</lastBuildDate>

    <item>
      <title>{item_title}</title>
      <description>{desc}</description>
      <link>{episode_url or link}</link>
      <guid isPermaLink="false">{guid}</guid>
      <pubDate>{last_build}</pubDate>
      {enclosure}
    </item>
  </channel>
</rss>
"""
    )
    with open(os.path.join(PUBLIC_DIR, "feed.xml"), "w", encoding="utf-8") as f:
        f.write(feed)

# -------------- Main orchestration --------------
def main():
    # Always use Boston local time for filenames & greeting
    now_bos = dt.datetime.now(ZoneInfo("America/New_York"))
    date_str = now_bos.strftime("%Y-%m-%d")
    # Natural time-of-day label for prompt
    hour = now_bos.hour
    if 5 <= hour < 12:
        tod = "morning"
    elif 12 <= hour < 17:
        tod = "afternoon"
    elif 17 <= hour < 22:
        tod = "evening"
    else:
        tod = "late-night"

    # Fetch & prep items
    items = dedupe(fetch_items())
    notes, used = notes_block(items)
    if used == 0:
        print("[warn] No usable items found; proceeding with minimal script", file=sys.stderr)

    # Load prompt (user-editable file) and draft script with OpenAI
    prompt_text = load_prompt("prompt.txt", date_label=date_str, tod_label=tod)
    script = draft_with_openai(prompt_text, notes).strip()

    # Hard fallback: if model yielded nothing, keep it short & factual
    if not script:
        script = (
            f"Good {tod}, it’s {now_bos.strftime('%A, %B %-d, %Y')}. "
            f"This is the Boston Briefing. Our AI had trouble generating today’s script, "
            f"so we’ll be back with a full update soon. Thanks for listening."
        )

    # Site outputs (always)
    write_shownotes(date_str, items)
    write_index()

    # Try ElevenLabs TTS with one retry; if still fails, synthesize a short apology
    ep_name = f"boston-briefing-{date_str}.mp3"
    ep_path = os.path.join(EP_DIR, ep_name)
    mp3_bytes = tts_elevenlabs(script, attempt_label="main")
    if not mp3_bytes:
        mp3_bytes = tts_elevenlabs(script, attempt_label="retry")

    if not mp3_bytes:
        apology = (
            "Oops, something went wrong generating today’s briefing. "
            "Sorry about that. Please email Matt Karolian so we can fix it."
        )
        mp3_bytes = tts_elevenlabs(apology, attempt_label="apology")

    if mp3_bytes:
        save_bytes(ep_path, mp3_bytes)
        filesize = len(mp3_bytes)
        ep_url = f"{PUBLIC_BASE_URL}/episodes/{ep_name}"
        print(f"Saved MP3: {ep_path} ({filesize} bytes)")
        build_feed(ep_url, now_bos, filesize)
    else:
        print("[warn] No audio produced; publishing feed without enclosure", file=sys.stderr)
        build_feed("", now_bos, 0)

if __name__ == "__main__":
    main()
