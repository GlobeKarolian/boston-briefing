import os, sys, json, datetime as dt, time
from zoneinfo import ZoneInfo
import feedparser, requests, yaml
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
from rapidfuzz import fuzz, process

# =========================
# ENV / CONSTANTS
# =========================
ELEVEN_API_KEY   = os.getenv("ELEVEN_API_KEY", "").strip()
ELEVEN_VOICE_ID  = os.getenv("ELEVEN_VOICE_ID", "").strip()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o").strip()   # set to a model you definitely have
PUBLIC_BASE_URL  = os.getenv("PUBLIC_BASE_URL", "").strip()
MAX_ITEMS        = int(os.getenv("MAX_ITEMS", "12"))

PUBLIC_DIR = "public"
EP_DIR     = os.path.join(PUBLIC_DIR, "episodes")
NOTES_DIR  = os.path.join(PUBLIC_DIR, "shownotes")
DEBUG_DIR  = os.path.join(PUBLIC_DIR, "debug")
for d in (PUBLIC_DIR, EP_DIR, NOTES_DIR, DEBUG_DIR):
    os.makedirs(d, exist_ok=True)

# =========================
# FEEDS CONFIG
# =========================
with open("feeds.yml", "r", encoding="utf-8") as f:
    feeds_cfg = yaml.safe_load(f) or {}

SOURCES    = feeds_cfg.get("sources", [])
EXCLUDE    = set(str(k).lower() for k in feeds_cfg.get("exclude_keywords", []))
LIMIT_PER  = int(feeds_cfg.get("daily_limit_per_source", 6))

# =========================
# HELPERS
# =========================
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
        doc = Document(html); cleaned = doc.summary()
        text = BeautifulSoup(cleaned, "html.parser").get_text("\n")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        lines = [l for l in lines if len(l.split()) > 4]
        return "\n".join(lines)
    except Exception:
        return ""

def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    for sep in [". ", "? ", "! ", " — ", " – ", " • "]:
        if sep in text:
            cand = text.split(sep)[0]
            if len(cand.split()) >= 8:
                return cand.strip(".•–—?! ")
    return text[:260].rsplit(" ",1)[0]

def boston_now():
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    return now

def greeting_and_date(now: dt.datetime):
    hr = now.hour
    if   5 <= hr < 12:  greet = "Good morning"
    elif 12 <= hr < 17: greet = "Good afternoon"
    elif 17 <= hr < 22: greet = "Good evening"
    else:               greet = "Hello"
    # Avoid %-d for Windows; remove leading zero manually
    day_num = str(int(now.strftime("%d")))
    date_line = f"{now.strftime('%A')}, {now.strftime('%B')} {day_num}, {now.year}"
    return greet, date_line

def build_notes(items):
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
        notes.append(f"- {sent}  (source: {it['source']}, link: {it['link']})")
        used += 1
    return notes

# =========================
# OPENAI (Chat Completions API) with retries
# =========================
def openai_chat(messages, max_tokens=2500, temperature=0.72, top_p=1.0, presence_penalty=0.15, frequency_penalty=0.35):
    if not OPENAI_API_KEY:
        print("[warn] No OPENAI_API_KEY", file=sys.stderr)
        return ""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    backoffs = [0, 2, 5]  # quick retries
    for attempt, delay in enumerate(backoffs, start=1):
        if delay: time.sleep(delay)
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[warn] OpenAI chat error (attempt {attempt}): {e}", file=sys.stderr)
    return ""

def generate_script_with_openai(notes, now: dt.datetime):
    greeting, date_line = greeting_and_date(now)
    prompt_txt = ""
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            prompt_txt = f.read().strip()
    except Exception:
        pass

    system_msg = (
        "You are a professional public radio host writing a tight, 2–3 minute Boston news briefing.\n"
        "Rules:\n"
        "- Lead with the most important hard news; sports only leads if it's indisputably #1.\n"
        "- Absolutely no editorializing, sympathy, or speculation (no 'thoughts and prayers').\n"
        "- Attribute sources naturally: 'The Boston Globe reports…', 'Boston.com notes…', 'B-Side highlights…'.\n"
        "- Keep paragraphs short for read-aloud pacing; vary sentence length for rhythm.\n"
        "- End with: “That’s the Boston Briefing. Email Matt@Boston.com with feedback. "
        "This internal beta was summarized by AI, and the voice is an AI recreation of Matt Karolian. Please keep this private.”"
    )

    user_msg = (
        f"{prompt_txt}\n\n"
        f"Context:\n"
        f"- Greeting line to use: \"{greeting}, it’s {date_line}.\"\n"
        f"- Audience: Greater Boston, smart but busy.\n"
        f"- Length target: ~220–370 words (2–3 minutes aloud).\n"
        f"- Use only the facts in NOTES. If unsure, omit.\n"
        f"- Use paragraph breaks (blank lines) between topics.\n\n"
        f"NOTES (deduped facts, may be rough):\n" + "\n".join(notes) + "\n\n"
        "Deliver the final narration script as one block of text (no headings/bullets/markdown)."
    )

    messages = [
        {"role":"system", "content": system_msg},
        {"role":"user",   "content": user_msg},
    ]
    return openai_chat(messages)

# =========================
# ELEVENLABS TTS with retry
# =========================
def tts_elevenlabs(text: str, settings=None) -> bytes | None:
    if not ELEVEN_API_KEY or not ELEVEN_VOICE_ID or not text.strip():
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    voice_settings = settings or {
        # Broadcasty + natural; adjust if needed
        "stability": 0.40,
        "similarity_boost": 0.92,
        "style": 0.50,
        "use_speaker_boost": True
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": voice_settings
    }
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json"
    }

    backoffs = [0, 3]  # one retry
    for attempt, delay in enumerate(backoffs, start=1):
        if delay: time.sleep(delay)
        try:
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as ex:
            print(f"[warn] ElevenLabs TTS error (attempt {attempt}): {ex}", file=sys.stderr)
    return None

def tts_apology() -> bytes | None:
    apology = ("Oops — something went wrong generating today’s script. "
               "Sorry about that. Please email Matt Karolian so we can fix it.")
    return tts_elevenlabs(apology)

# =========================
# OUTPUT / FEED
# =========================
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
    with open(os.path.join(NOTES_DIR, f"{date_str}.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))

def write_index():
    idx = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(idx):  # don't overwrite a custom site
        return
    feed = f"{PUBLIC_BASE_URL.rstrip('/')}/feed.xml" if PUBLIC_BASE_URL else "feed.xml"
    notes = f"{PUBLIC_BASE_URL.rstrip('/')}/shownotes/" if PUBLIC_BASE_URL else "shownotes/"
    html = f"""<html><head><meta charset='utf-8'><title>Boston Briefing</title></head>
<body>
  <h1>Boston Briefing</h1>
  <p><strong>Internal beta — do not share externally.</strong></p>
  <p>Podcast RSS: <a href="{feed}">{feed}</a></p>
  <p>Shownotes: <a href="{notes}">Open folder</a></p>
</body></html>"""
    with open(idx, "w", encoding="utf-8") as f:
        f.write(html)

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
      <link>{episode_url or link}</link>
      <guid isPermaLink="false">{guid}</guid>
      <pubDate>{last_build}</pubDate>
      {enclosure}
    </item>
  </channel>
</rss>
"""
    with open(os.path.join(PUBLIC_DIR, "feed.xml"), "w", encoding="utf-8") as f:
        f.write(feed)

# =========================
# MAIN
# =========================
def main():
    now = boston_now()
    date_str = now.strftime("%Y-%m-%d")

    # Gather news and notes
    items = dedupe(fetch_items())
    notes = build_notes(items)

    # Build script
    script = ""
    if notes:
        script = generate_script_with_openai(notes, now) or ""
    else:
        print("[warn] No usable notes; script will be minimal.", file=sys.stderr)

    # Always save latest script for debugging
    try:
        with open(os.path.join(DEBUG_DIR, "latest_script.txt"), "w", encoding="utf-8") as f:
            f.write(script)
    except Exception:
        pass

    # Write auxiliary pages
    write_shownotes(date_str, items)
    write_index()

    # TTS
    if script.strip():
        mp3 = tts_elevenlabs(script)
        if not mp3:
            print("[warn] ElevenLabs main TTS failed; trying apology.", file=sys.stderr)
            mp3 = tts_apology()
    else:
        print("[warn] Empty script; using apology audio.", file=sys.stderr)
        mp3 = tts_apology()

    # Save episode + feed
    ep_name = f"boston-briefing-{date_str}.mp3"
    ep_path = os.path.join(EP_DIR, ep_name)
    if mp3:
        with open(ep_path, "wb") as f:
            f.write(mp3)
        filesize = len(mp3)
        ep_url = f"{PUBLIC_BASE_URL.rstrip('/')}/episodes/{ep_name}" if PUBLIC_BASE_URL else f"episodes/{ep_name}"
        build_feed(ep_url, now, filesize)
        print(f"Saved MP3: {ep_path} ({filesize} bytes)")
    else:
        build_feed("", now, 0)
        print("[warn] No audio produced; feed item written without enclosure.", file=sys.stderr)

if __name__ == "__main__":
    main()
