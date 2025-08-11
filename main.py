import os, sys, io, json, datetime as dt
import feedparser, requests, yaml
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
from rapidfuzz import fuzz, process
from email.utils import format_datetime

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "").strip()
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "12"))

PUBLIC_DIR = "public"
EP_DIR = os.path.join(PUBLIC_DIR, "episodes")
os.makedirs(PUBLIC_DIR, exist_ok=True)
os.makedirs(EP_DIR, exist_ok=True)

with open("feeds.yml", "r", encoding="utf-8") as f:
    feeds_cfg = yaml.safe_load(f)

SOURCES = feeds_cfg.get("sources", [])
EXCLUDE = set(k.lower() for k in feeds_cfg.get("exclude_keywords", []))
LIMIT_PER = int(feeds_cfg.get("daily_limit_per_source", 6))

def is_newsworthy(title: str) -> bool:
    t = title.lower()
    return not any(k in t for k in EXCLUDE)

def fetch_items():
    items = []
    for src in SOURCES:
        name, rss = src.get("name","Unknown"), src.get("rss","")
        if not rss: continue
        try:
            fp = feedparser.parse(rss)
            count = 0
            for e in fp.entries:
                if count >= LIMIT_PER: break
                title = (e.get("title") or "").strip()
                link = (e.get("link") or "").strip()
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
        if not seen:
            kept.append(it); seen.append(it["title"]); continue
        match = process.extractOne(it["title"], seen, scorer=fuzz.token_set_ratio)
        if not match or match[1] < threshold:
            kept.append(it); seen.append(it["title"])
    return kept

def extract_text(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url, timeout=15)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted and len(extracted.split()) > 40:
                return extracted
    except Exception:
        pass
    try:
        html = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"}).text
        doc = Document(html); cleaned = doc.summary()
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

def build_script(items):
    today = dt.datetime.now().astimezone()
    intro = f"Good morning, it’s {today.strftime('%A, %B %-d, %Y')}. Here’s your Boston Briefing."
    lines = []
    used = 0
    for it in items:
        if used >= MAX_ITEMS: break
        txt = extract_text(it["link"])
        if not txt:
            continue
        sent = first_sentence(txt)
        if len(sent.split()) < 6:
            continue
        lines.append(f"According to {it['source']}: {sent}.")
        used += 1
    outro = "That’s the Boston Briefing. Links to all sources are on the website."
    full = intro + "\n\n" + "\n".join(lines) + "\n\n" + outro
    return full, lines

def tts_elevenlabs(text: str) -> bytes | None:
    if not ELEVEN_API_KEY or not ELEVEN_VOICE_ID:
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.35,
            "similarity_boost": 0.9,
            "style": 0.65,
            "use_speaker_boost": True
        },
        "model_id": "eleven_multilingual_v2",
        # If your account supports it, this speeds up delivery slightly:
        # "voice_speed": 1.12
    }
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json"
    }
    # Increased timeout from 60 -> 180 seconds
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
    r.raise_for_status()
    return r.content

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

def build_feed(episode_url: str, pub_dt: dt.datetime, filesize: int):
    title = "Boston Briefing"
    desc = "A short, 100% factual morning news briefing for Greater Boston."
    link = PUBLIC_BASE_URL
    last_build = format_datetime(pub_dt)
    item_title = pub_dt.strftime("Boston Briefing – %Y-%m-%d")
    guid = episode_url or item_title

    enclosure = ""
    if episode_url:
        enclosure = f'<enclosure url="{episode_url}" length="{filesize}" type="audio/mpeg"/>'

    feed = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">\n'
        '  <channel>\n'
        f'    <title>{title}</title>\n'
        f'    <link>{link}</link>\n'
        '    <language>en-us</language>\n'
        f'    <description>{desc}</description>\n'
        '    <itunes:author>Boston Briefing</itunes:author>\n'
        '    <itunes:explicit>false</itunes:explicit>\n'
        f'    <lastBuildDate>{last_build}</lastBuildDate>\n'
        '\n'
        '    <item>\n'
        f'      <title>{item_title}</title>\n'
        f'      <description>{desc}</description>\n'
        f'      <link>{episode_url}</link>\n'
        f'      <guid isPermaLink="false">{guid}</guid>\n'
        f'      <pubDate>{format_datetime(pub_dt)}</pubDate>\n'
        f'      {enclosure}\n'
        '    </item>\n'
        '  </channel>\n'
        '</rss>\n'
    )

    with open(os.path.join(PUBLIC_DIR, "feed.xml"), "w", encoding="utf-8") as f:
        f.write(feed)

def write_index():
    html = f"""<html><head><meta charset='utf-8'><title>Boston Briefing</title></head>
<body>
  <h1>Boston Briefing</h1>
  <p>Podcast RSS: <a href="{PUBLIC_BASE_URL}/feed.xml">{PUBLIC_BASE_URL}/feed.xml</a></p>
  <p>Shownotes: <a href="{PUBLIC_BASE_URL}/shownotes/">Open folder</a></p>
</body></html>"""
    with open(os.path.join(PUBLIC_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

def main():
    items = fetch_items()
    items = dedupe(items)
    script, _lines = build_script(items)
    today = dt.datetime.now().astimezone()
    date_str = today.strftime("%Y-%m-%d")

    # NEW: show and save the script so you can review it
    print("\n--- SCRIPT TO READ ---\n")
    print(script)
    print("\n--- END SCRIPT ---\n")
    with open(os.path.join(PUBLIC_DIR, f"script-{date_str}.txt"), "w", encoding="utf-8") as f:
        f.write(script)

    write_shownotes(date_str, items)
    write_index()

    mp3_bytes = None
    try:
        mp3_bytes = tts_elevenlabs(script)
    except Exception as ex:
        print(f"[warn] ElevenLabs error: {ex}", file=sys.stderr)

    ep_name = f"boston-briefing-{date_str}.mp3"
    ep_path = os.path.join(EP_DIR, ep_name)
    if mp3_bytes:
        with open(ep_path, "wb") as f:
            f.write(mp3_bytes)
        filesize = len(mp3_bytes)
        ep_url = f"{PUBLIC_BASE_URL}/episodes/{ep_name}"
        build_feed(ep_url, today, filesize)
    else:
        build_feed("", today, 0)

if __name__ == "__main__":
    main()
