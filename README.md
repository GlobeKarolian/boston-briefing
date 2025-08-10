# Boston Briefing (Starter)

A no-code-ish, scheduled pipeline that builds a **100% factual** Boston morning briefing podcast and publishes it via **GitHub Pages**. It fetches headlines from RSS feeds, extracts first-paragraph facts, writes a tight script with explicit attribution, converts it to audio using **ElevenLabs**, and deploys a public podcast feed.

## What you need (accounts)
- **GitHub** account (free)
- **ElevenLabs** account + API key (free tier works to start)

> No AWS required: the MP3 and podcast RSS are hosted by **GitHub Pages**.

---

## One-time setup (10–15 minutes)

1) **Create a new GitHub repo** (e.g., `boston-briefing`).
2) **Upload these starter files** (drag/drop or use "Add file" → "Upload files"). Keep the structure the same.
3) In your repo, open **Settings → Pages** and set:
   - *Build and deployment* → **Source: GitHub Actions**.
4) Add **Secrets** (Settings → Secrets and variables → Actions → New repository secret):
   - `ELEVEN_API_KEY` = your ElevenLabs API key
   - `ELEVEN_VOICE_ID` = your ElevenLabs Voice ID (copy from ElevenLabs voice page)
   - `PUBLIC_BASE_URL` = `https://<your-username>.github.io/<your-repo>`
5) Edit **feeds.yml** to tweak your source list (optional for day 1).

## Test it now
- Go to **Actions** → select “Build & Deploy” → **Run workflow**.
- After it finishes, visit:  
  `https://<your-username>.github.io/<your-repo>/feed.xml`  
  You should see a valid podcast RSS with today’s episode.
- Add that URL to your podcast player.

## Daily schedule
- By default, it runs every day at **6:05am Eastern Time** (10:05 UTC). You can change the time in `.github/workflows/build_and_deploy.yml`.

## How it stays 100% factual
- Only uses **facts present in the article** excerpts.
- Each item is prefixed with **“According to {Source} …”**
- No opinions, no speculation, and it automatically **skips** items it can’t extract cleanly.

---

## Customize (later)
- Add/remove feeds in `feeds.yml`.
- Tweak script length via `MAX_ITEMS` env (default 12).
- Switch voices in ElevenLabs by changing `ELEVEN_VOICE_ID`.
- Add background music (advanced).

## Troubleshooting
- If the feed says *empty*: your ElevenLabs key/voice is missing or feeds returned nothing extractable.
- If your player can’t subscribe, make sure the feed URL starts with `https://` and the site is live (Pages may take a minute after first deploy).
