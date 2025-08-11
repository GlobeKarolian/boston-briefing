import os, sys, io, json, datetime as dt
import feedparser, requests, yaml
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
from rapidfuzz import fuzz, process

# ---------- Config via env ----------
ELEVEN_API_KEY   = os.getenv("ELEVEN_API_KEY", "").strip()
ELEVEN_VOICE_ID  = os.getenv("ELEVEN_VOICE_ID", "").strip()
PUBLIC_BASE_URL  = os.getenv("PUBLIC_BASE_URL", "").strip()  # e.g. https://globekarolian.github.io/boston-briefing
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()  # set in repo secrets; switch later if you want
MAX_ITEMS        = int(os.getenv("MAX_ITEMS", "12"))

# ElevenLabs tuning (safe defaults; adjust in small steps)
EL_STABILITY         = float(os.getenv("EL_STABILITY", "0.65"))
EL_SIMILARITY_BOOST  = float(os.getenv("EL_SIMILARITY_BOOST", "0.7"))
EL_STYLE             = float(os.getenv("EL_STYLE", "0.25"))
EL_USE_SPEAKER_BOOST = bool(int(os.getenv("EL_SPEAKER_BOOST", "1")))
EL_SPEED             = float(os.getenv("EL_SPEED", "1.08"))  # slightly brisker for news

# ---------- Folders ----------
PUBLIC_DIR = "public"
EP_DIR     = os.path.join(PUBLIC_DIR, "episodes")
DEBUG_DIR  = os.path.join(PUBLIC_DIR, "debug")
os.makedirs(PUBLIC_DIR, exist_ok=True)
os.makedirs(EP_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# ---------- Feeds config ----------
with open("feeds.yml", "r", encoding="utf-8") as f:
    feeds_cfg = yaml.safe_load(f)

SOURCES    = feeds_cfg.get("sources", [])
EXCLUDE    = set(k.lower() for k in feeds_cfg.get("exclude_keywords", []))
LIMIT_PER  = int(feeds_cfg.get("daily_limit_per_source", 6))

# ---------- Helpers ----------
def is_newsworthy(title: str) -> bool:
    t = (title or "").lower()
    return bool(t) and not any(k in t for k in EXCLUDE)

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
                link = (e.get("link") or "").strip()
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
    # 1) trafilatura (fast & clean when it works)
    try:
        downloaded = trafilatura.fetch_url(url, timeout=15)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted and len(extracted.split()) > 40:
                return extracted
    except Exception:
        pass
    # 2) readability as fallback
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
    # Try sentence-ish breaks; prefer ≥8 words to avoid fragments
    for sep in [". ", "… ", " — ", " – ", " • ", "? ", "! "]:
        if sep in text:
            cand = text.split(sep)[0]
            if len(cand.split()) >= 8:
                return cand.strip(".•–—?! ")
    # Fallback truncate
    cut = text[:280].rsplit(" ", 1)[0]
    return cut

def greeting_and_dateline(now: dt.datetime):
    # Human greeting based on local time
    hour = now.hour
    if   5 <= hour < 12:  greet = "Good morning"
    elif 12 <= hour < 17: greet = "Good afternoon"
    elif 17 <= hour < 22: greet = "Good evening"
    else:                 greet = "Hello"
    # Dateline like: Monday, August 11, 2025
    # Use platform-independent day formatting (avoid %-d on Windows)
    day_num = str(now.day)
    date_line = f"{now.strftime('%A')}, {now.strftime('%B')} {day_num}, {now.year}"
    return greet, date_line

def load_prompt():
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

# ---------- OpenAI (Responses API) ----------
def generate_script_with_openai(outlines: str, date_line: str, greeting: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Guardrail system message (keeps tone + constraints tight)
    sys_msg = (
        "You are writing a short, spoken news brief for a Boston audience.\n"
        "Rules:\n"
        "- Lead with the most important hard news; avoid sports first unless it's city-scale significance.\n"
        "- Always be factual and neutral. Absolutely no opinions, condolences, or speculation.\n"
        "- Natural public-radio cadence; attributions should be smooth and conversational "
        "(e.g., 'The Boston Globe reports …', 'Boston.com notes …', 'B-Side highlights …').\n"
        "- Keep it tight (about 90–150 seconds when read).\n"
        "- If greeting/time feel mismatched (e.g., late evening), use a neutral greeting like 'Hello'.\n"
        "- End with: 'This is an internal beta; AI summarized and voiced this update.'"
    )

    user_msg = (
        f"DATE_LINE: {date_line}\n"
        f"GREETING: {greeting}\n\n"
        "NOTES (bullet points of facts; deduped):\n"
        f"{outlines}\n\n"
        "Task: Write the full script as one block of text (no bullets). "
        "Keep attributions natural and brief. Avoid filler. Do not add condolences. "
        "Do not claim certainty where the notes don't support it."
    )

    # Important: some models reject 'temperature'. We omit it.
    # Use max_completion_tokens (not max_tokens) on the Responses API.
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg}
        ],
        max_completion_tokens=800
    )

    txt = (getattr(resp, "output_text", None) or "").strip()
    return txt

# ---------- ElevenLabs ----------
def tts_elevenlabs(text: str) -> bytes | None:
    if not ELEVEN_API_KEY or not ELEVEN_VOICE_ID:
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": EL_STABILITY,
            "similarity_boost": EL_SIMILARITY_BOOST,
            "style": EL_STYLE,
            "use_speaker_boost": EL_USE_SPEAKER_BOOST
        },
        "generation_config": {
            # speed isn’t in voice_settings; it’s here for v2
            "chunk_length_schedule": [EL_SPEED]
        }
    }
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json"
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
    r.raise_for_status()
    return r.content

# ---------- Output helpers ----------
def write_shownotes(date_str, items):
    html = [
        "<html><head><meta charset='utf-8'><title>Boston Briefing – Sources</title></head><body>",
        f"<h2>Boston Briefing – {date_str}</h2>",
        "<ol>"
    ]
    for it in items[:MAX_ITEMS]:
        html.append(
            f"<li><a href='{it['link']}' target='_blank' rel='noopener'>{it['title']}</a> – {it['source']}</li>"
        )
    html.append("</ol></body></html>")
    notes_dir = os.path.join(PUBLIC_DIR, "shownotes")
    os.makedirs(notes_dir, exist_ok=True)
    with open(os.path.join(notes_dir, f"{date_str}.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))

def build_feed(episode_url: str, pub_dt: dt.datetime, filesize: int):
    title = "Boston Briefing"
    desc  = "A short, factual Boston news briefing."
    link  = PUBLIC_BASE_URL or ""
    last_build = pub_dt.astimezone().strftime("%a, %d %b %Y %H:%M:%S %z")
    item_title = pub_dt.strftime("Boston Briefing – %Y-%m-%d")
    guid = episode_url or item_title

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">')
    lines.append("  <channel>")
    lines.append(f"    <title>{title}</title>")
    lines.append(f"    <link>{link}</link>")
    lines.append("    <language>en-us</language>")
    lines.append(f"    <description>{desc}</description>")
    lines.append("    <itunes:author>Boston Briefing</itunes:author>")
    lines.append("    <itunes:explicit>false</itunes:explicit>")
    lines.append(f"    <lastBuildDate>{last_build}</lastBuildDate>")
    lines.append("    <item>")
    lines.append(f"      <title>{item_title}</title>")
    lines.append(f"      <description>{desc}</description>")
    lines.append(f"      <link>{episode_url}</link>")
    lines.append(f"      <guid isPermaLink=\"false\">{guid}</guid>")
    lines.append(f"      <pubDate>{last_build}</pubDate>")
    if episode_url and filesize > 0:
        lines.append(
            f"      <enclosure url=\"{episode_url}\" length=\"{filesize}\" type=\"audio/mpeg\"/>"
        )
    lines.append("    </item>")
    lines.append("  </channel>")
    lines.append("</rss>")
    with open(os.path.join(PUBLIC_DIR, "feed.xml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_index():
    # minimal root index so your public base stays handy
    url = f"{PUBLIC_BASE_URL}/feed.xml" if PUBLIC_BASE_URL else "feed.xml"
    shownotes = f"{PUBLIC_BASE_URL}/shownotes/" if PUBLIC_BASE_URL else "shownotes/"
    html = [
        "<html><head><meta charset='utf-8'><title>Boston Briefing</title></head><body>",
        "<h1>Boston Briefing</h1>",
        f"<p>Podcast RSS: <a href=\"{url}\">{url}</a></p>",
        f"<p>Shownotes: <a href=\"{shownotes}\">Open folder</a></p>",
        "</body></html>"
    ]
    with open(os.path.join(PUBLIC_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))

# ---------- Pipeline ----------
def build_script(items):
    now = dt.datetime.now().astimezone()
    greeting, date_line = greeting_and_dateline(now)

    # Outline bullets from first sentences (with source hint)
    bullets = []
    used = 0
    for it in items:
        if used >= MAX_ITEMS: break
        body = extract_text(it["link"])
        if not body:
            continue
        sent = first_sentence(body)
        if len(sent.split()) < 6:
            continue
        bullets.append(f"- {sent} (source: {it['source']})")
        used += 1
    outlines = "\n".join(bullets)

    # Load your tweakable prompt, then send to OpenAI
    base_prompt = load_prompt()
    try:
        text = generate_script_with_openai(
            outlines=outlines,
            date_line=date_line,
            greeting=greeting
        )
        # If you maintain an editor prompt, prepend it here (model reads both)
        if base_prompt:
            text = text  # already guided by system+user; keep prompt.txt as your authored guidance source file
    except Exception as ex:
        print(f"[warn] OpenAI error: {ex}", file=sys.stderr)
        text = ""

    # Always write latest script for debugging
    try:
        with open(os.path.join(DEBUG_DIR, "latest_script.txt"), "w", encoding="utf-8") as f:
            f.write(text or "")
    except Exception:
        pass

    # If nothing sensible came back, return empty (caller decides fallback)
    return text.strip(), outlines

def main():
    items = fetch_items()
    items = dedupe(items)

    now = dt.datetime.now().astimezone()
    date_str = now.strftime("%Y-%m-%d")

    script, outlines = build_script(items)

    # Public scaffolding (shownotes & index always)
    write_shownotes(date_str, items)
    write_index()

    # Determine what to speak
    if not script or len(script.split()) < 20:
        voice_text = (
            "Oops — something went wrong generating today's script. "
            "Sorry about that. Please email Matt Karolian so I can fix it."
        )
    else:
        voice_text = script

    # Try to synthesize
    mp3_bytes = None
    try:
        mp3_bytes = tts_elevenlabs(voice_text)
    except Exception as ex:
        print(f"[warn] ElevenLabs error: {ex}", file=sys.stderr)

    # Write episode & feed
    ep_name = f"boston-briefing-{date_str}.mp3"
    ep_path = os.path.join(EP_DIR, ep_name)

    if mp3_bytes:
        with open(ep_path, "wb") as f:
            f.write(mp3_bytes)
        filesize = len(mp3_bytes)
        ep_url = f"{PUBLIC_BASE_URL}/episodes/{ep_name}" if PUBLIC_BASE_URL else f"episodes/{ep_name}"
        build_feed(ep_url, now, filesize)
        print(f"Saved MP3: {ep_path} ({filesize} bytes)")
    else:
        build_feed("", now, 0)
        print("[warn] No MP3 produced; feed item created without enclosure")

if __name__ == "__main__":
    main()
