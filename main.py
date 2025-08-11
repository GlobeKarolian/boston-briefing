import os, sys, io, json, datetime as dt
import feedparser, requests, yaml
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
from rapidfuzz import fuzz, process
from email.utils import format_datetime

# ---------- ENV / CONSTANTS ----------
ELEVEN_API_KEY  = os.getenv("ELEVEN_API_KEY", "").strip()
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "").strip()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-5").strip()  # default to flagship

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
MAX_ITEMS       = int(os.getenv("MAX_ITEMS", "12"))

PUBLIC_DIR = "public"
EP_DIR     = os.path.join(PUBLIC_DIR, "episodes")
os.makedirs(PUBLIC_DIR, exist_ok=True)
os.makedirs(EP_DIR, exist_ok=True)

# ---------- FEEDS CONFIG ----------
with open("feeds.yml", "r", encoding="utf-8") as f:
    feeds_cfg = yaml.safe_load(f)

SOURCES    = feeds_cfg.get("sources", [])
EXCLUDE    = set(k.lower() for k in feeds_cfg.get("exclude_keywords", []))
LIMIT_PER  = int(feeds_cfg.get("daily_limit_per_source", 6))

# ---------- HELPERS ----------
def is_newsworthy(title: str) -> bool:
    t = (title or "").lower()
    return bool(title) and not any(k in t for k in EXCLUDE)

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
                link  = (e.get("link")  or "").strip()
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

def fetch_html(url: str) -> str:
    return requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"}).text

def extract_text(url: str) -> str:
    # try trafilatura
    try:
        downloaded = trafilatura.fetch_url(url, timeout=20)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted and len(extracted.split()) > 40:
                return extracted
    except Exception:
        pass
    # fallback readability
    try:
        html = fetch_html(url)
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
    # Find a substantial first sentence
    stops = [". ", "?” ", "!” ", "— ", " – ", " • "]
    for sep in stops:
        if sep in text:
            cand = text.split(sep)[0]
            if len(cand.split()) >= 8:
                return cand.strip(" .•–—”")
    # fallback clip
    return text[:220].rsplit(" ",1)[0]

def greeting_for_now():
    # Eastern Time greeting
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=-4)))  # naive ET approximation
    hr = now.hour
    if 5 <= hr < 12:
        return "Good morning"
    if 12 <= hr < 17:
        return "Good afternoon"
    if 17 <= hr < 22:
        return "Good evening"
    return "Hello"

def today_strings():
    now = dt.datetime.now().astimezone()
    iso = now.strftime("%Y-%m-%d")
    nice = now.strftime("%A, %B %-d, %Y") if "%" in "%-d" else now.strftime("%A, %B %d, %Y").replace(" 0"," ")
    return now, iso, nice

# ---------- FACT NOTES ----------
def build_fact_notes(items):
    notes = []
    used = 0
    for it in items:
        if used >= MAX_ITEMS:
            break
        txt = extract_text(it["link"])
        if not txt:
            continue
        sent = first_sentence(txt)
        if len(sent.split()) < 6:
            continue
        # a compact note with source tag
        notes.append(f"- {sent}  [source: {it['source']}, link: {it['link']}]")
        used += 1
    return notes

# ---------- GPT SCRIPT ----------
def load_prompt_template():
    # External prompt file so you can tweak without code changes
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def openai_chat(messages):
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": OPENAI_MODEL,
        "messages": messages,
        # Cadillac generation settings
        "max_completion_tokens": 2500,
        "temperature": 0.72,
        "top_p": 1.0,
        "presence_penalty": 0.15,
        "frequency_penalty": 0.35,
    }
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI error: {r.status_code} – {r.text}")
    data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()

def generate_script_with_openai(notes, date_nice):
    prompt_template = load_prompt_template()

    # System message: guardrails + voice
    system = (
        "You are a professional public radio host writing a tight, 2–3 minute Boston news briefing. "
        "Natural, conversational delivery like Kai Ryssdal. Don’t editorialize. No ‘thoughts and prayers’. "
        "Attribute sources conversationally, not rigidly. Lead with the most important local news first; "
        "sports goes later unless it’s truly top news. Include a quick weather line only if relevant. "
        "Avoid numbers soup; group stats and keep them digestible. Keep paragraphs short for read-aloud. "
        "End with a brief disclosure that this is an internal beta: AI-summarized + AI voice recreation of Matt Karolian."
    )

    greeting = greeting_for_now()

    # User content: template + runtime context
    joined_notes = "\n".join(notes)
    user_content = f"""
{prompt_template}

Context:
- Date: {date_nice}
- Greeting to use appropriately: "{greeting}, it’s {date_nice}."
- Audience: Greater Boston, general listeners.
- Keep it 2–3 minutes when read aloud (~220–370 words).
- Strictly factual from the notes below; if unsure, omit. Do NOT invent facts.
- Use paragraph breaks generously for natural pacing.

Story notes (each line is a fact you may rewrite but not embellish):
{joined_notes}

Deliverables:
1) A clean narration script only (no markdown, no labels). Paragraphs separated by blank lines.
2) Include light, conversational attributions (‘the Globe says…’, ‘Boston.com reports…’, ‘B-Side notes…’) without overdoing it.
3) Close with: “That’s the Boston Briefing. Email Matt@Boston.com with feedback. This internal beta was summarized by AI, and the voice is an AI recreation of Matt Karolian. Please keep this private.”
""".strip()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    return openai_chat(messages)

# ---------- ELEVENLABS TTS ----------
def tts_elevenlabs(text: str) -> bytes | None:
    if not ELEVEN_API_KEY or not ELEVEN_VOICE_ID:
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            # Cadillac-ish settings for broadcasty, natural delivery
            "stability": 0.40,
            "similarity_boost": 0.92,
            "style": 0.50,
            "use_speaker_boost": True
        }
    }
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json"
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    try:
        r.raise_for_status()
    except Exception:
        print(f"[warn] ElevenLabs error response: {r.text[:400]}", file=sys.stderr)
        raise
    return r.content

def tts_apology() -> bytes | None:
    apology = (
        "Oops, something went wrong. Sorry about that. "
        "Why don't you email Matt Karolian so I can fix it."
    )
    try:
        return tts_elevenlabs(apology)
    except Exception as ex:
        print(f"[warn] ElevenLabs fallback also failed: {ex}", file=sys.stderr)
        return None

# ---------- OUTPUT ----------
def write_shownotes(date_str, items):
    html = ["<html><head><meta charset='utf-8'><title>Boston Briefing – Sources</title></head><body>"]
    html.append(f"<h2>Boston Briefing – {date_str}</h2>")
    html.append("<ol>")
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

    enclosure = f'<enclosure url="{episode_url}" length="{filesize}" type="audio/mpeg"/>' if episode_url else ""

    feed = f"""<?xml version="1.0" encoding="UTF-8"?>
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
      <link>{episode_url}</link>
      <guid isPermaLink="false">{guid}</guid>
      <pubDate>{last_build}</pubDate>
      {enclosure}
    </item>
  </channel>
</rss>
"""
    with open(os.path.join(PUBLIC_DIR, "feed.xml"), "w", encoding="utf-8") as f:
        f.write(feed)

def write_index():
    base = PUBLIC_BASE_URL.rstrip("/") if PUBLIC_BASE_URL else ""
    html = f"""<html><head><meta charset='utf-8'><title>Boston Briefing</title></head>
<body>
  <h1>Boston Briefing</h1>
  <p>Podcast RSS: <a href="{base}/feed.xml">{base}/feed.xml</a></p>
  <p>Shownotes: <a href="{base}/shownotes/">Open folder</a></p>
</body></html>"""
    with open(os.path.join(PUBLIC_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

# ---------- MAIN ----------
def main():
    run_dt, date_iso, date_nice = today_strings()

    # 1) gather notes
    items = dedupe(fetch_items())
    notes = build_fact_notes(items)

    # 2) generate script with GPT (Cadillac settings)
    script_text = ""
    if OPENAI_API_KEY:
        try:
            script_text = generate_script_with_openai(notes, date_nice)
        except Exception as ex:
            print(f"[warn] OpenAI error: {ex}", file=sys.stderr)
    else:
        print("[warn] OPENAI_API_KEY is missing; skipping script generation.", file=sys.stderr)

    # ensure paragraphs for better prosody
    script_text = (script_text or "").strip()
    if script_text:
        # Prefix with greeting line for clarity
        greeting = greeting_for_now()
        opener = f"{greeting}, it’s {date_nice}. Here’s your Boston Briefing."
        if not script_text.lower().startswith(greeting.lower()):
            script_text = opener + "\n\n" + script_text

    # 3) write site scaffolding (shownotes & index)
    write_shownotes(date_iso, items)
    write_index()

    # 4) TTS (with apology fallback)
    mp3_bytes = None
    ep_url = ""
    try:
        if script_text:
            mp3_bytes = tts_elevenlabs(script_text)
        else:
            mp3_bytes = tts_apology()
    except Exception as ex:
        print(f"[warn] ElevenLabs TTS failed: {ex}", file=sys.stderr)
        mp3_bytes = tts_apology()

    # 5) publish podcast feed + file
    ep_name = f"boston-briefing-{date_iso}.mp3"
    ep_path = os.path.join(EP_DIR, ep_name)
    os.makedirs(EP_DIR, exist_ok=True)

    if mp3_bytes:
        with open(ep_path, "wb") as f:
            f.write(mp3_bytes)
        filesize = len(mp3_bytes)
        ep_url = f"{PUBLIC_BASE_URL.rstrip('/')}/episodes/{ep_name}" if PUBLIC_BASE_URL else ""
        build_feed(ep_url, run_dt, filesize)
        print(f"Saved MP3: {ep_path} ({filesize} bytes)")
    else:
        print("[warn] No audio produced; writing feed without enclosure.", file=sys.stderr)
        build_feed("", run_dt, 0)

if __name__ == "__main__":
    main()
