# Boston Briefing – Static Test Site (Internal Beta)

This folder contains the small mobile‑friendly site for the Boston Briefing internal test.

## How to use
1) Upload the `site/` folder to the root of your repo (same level as `main.py`).
2) In your workflow `.github/workflows/build_and_deploy.yml`, add this step **after** `Run python main.py`:

```yaml
    - name: Copy static site over generated folder
      run: |
        mkdir -p public
        cp -r site/* public/
```

3) Re‑run **Build & Deploy** in GitHub Actions.
4) Open your Pages URL: `https://globekarolian.github.io/boston-briefing/`

This site clearly labels itself as **INTERNAL BETA** and links to your podcast feed and shownotes.
