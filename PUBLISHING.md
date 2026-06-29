# Publishing AlphaMoo to GitHub

> **STOP.** First, revoke the GitHub token you pasted in chat. Go to:
> https://github.com/settings/tokens → find the token → **Revoke**
>
> That token is now in your chat log. Anyone with chat access has it. Rotate first, then publish.

## Step 1: Create a fresh Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click **"Generate new token (classic)"** or use a fine-grained token
3. Select scope: `repo` (full control of private repositories)
4. Set expiration: 90 days (or whatever you prefer)
5. Copy the token to your password manager — **never paste it in chat**

## Step 2: Initialize the repo locally

The repo scaffolding is already built at `/home/z/my-project/alphamoo/`. To publish:

```bash
cd /home/z/my-project/alphamoo

# Initialize git
git init
git branch -M main

# Stage everything (respects .gitignore — .vtx files, .env, secrets all excluded)
git add .

# Verify nothing dangerous got staged
git status
# Should show: src/, tests/, scripts/, .github/, docs files, README.md, etc.
# Should NOT show: data/*.vtx, .env, *.pem, anything starting with ghp_/sk-/ak-

# First commit
git commit -m "Initial commit: AlphaMoo v0.4.1

- Phase 0.5 data mining pipeline (14,798 actions analyzed)
- Phase 1 perception module (CCL + cascade interpreter)
- Module 16: Agent State Tracker (94% detection rate)
- Vortex reader, frame renderer, replay viewer CLI
- Full test suite (49 tests, all passing)
- Project scaffolding: pyproject.toml, CI, issue templates"

# Add the remote (the repo https://github.com/0xKunalsharma/Alphamoo must exist)
# Create the empty repo on GitHub first via the web UI, THEN:
git remote add origin https://github.com/0xKunalsharma/Alphamoo.git

# Push
git push -u origin main
```

## Step 3: Publish the data as a release artifact

The 25 .vtx files (33MB compressed) shouldn't go in git but should be downloadable. Use GitHub Releases:

```bash
# Create a release with the data zip attached
gh release create data-v1 \
    /home/z/my-project/alphamoo/data/converted_vortex_data.zip \
    --repo 0xKunalsharma/Alphamoo \
    --title "Public Replay Data v1" \
    --notes "25 ARC-AGI-3 public demo game replays, 14,798 actions total, Vortex format (33MB compressed).

Source: played on arcprize.org playground, exported via the ARC-AGI-3 toolkit.
Use: download and unzip into the project's data/ directory."
```

## Step 4: Verify CI passes

After the first push, GitHub Actions will run the CI workflow at
`.github/workflows/ci.yml`. Check the Actions tab to confirm:
- Lint passes (`ruff check`)
- Unit tests pass (41 tests, no data required)
- Integration tests are skipped (data not available in CI yet)

If integration tests should run in CI, the workflow already downloads
`data-v1` from releases — just make sure you published the release in Step 3.

## What's NOT in the repo (by design)

Per `.gitignore`:

- `data/*.vtx` — too large for git, distributed via releases
- `download/` — generated PNGs and mining reports (regenerable)
- `.env`, `*.pem`, `ghp_*`, `sk-*` — secrets, never committed
- `*.safetensors`, `*.gguf`, `*.bin` — model weights (separate releases)
- `__pycache__/`, `.pytest_cache/`, `htmlcov/` — Python junk

## Repo structure

```
Alphamoo/
├── .github/
│   ├── ISSUE_TEMPLATE/       Bug report + feature request templates
│   └── workflows/ci.yml      Lint + test on push/PR
├── src/alphamoo/             Main package (8 modules)
│   ├── schemas.py            Data contract
│   ├── vtx_reader.py         Vortex .vtx parser
│   ├── frame_renderer.py     Grid → PNG
│   ├── perception.py         Module 1: CCL + objects + relations
│   ├── cascade_interpreter.py Module 14: cascade diff
│   ├── agent_tracker.py      Module 16: agent state tracking
│   ├── data_mining.py        Phase 0.5 pipeline
│   └── replay_viewer.py      CLI inspection tool
├── tests/                    49 tests (41 unit + 8 integration)
├── scripts/                  Test scripts for visual inspection
├── data/                     (gitignored .vtx files go here)
├── .gitignore                Blocks secrets, large files, junk
├── pyproject.toml            Package config, deps, lint, pytest
├── LICENSE                   MIT
├── CONTRIBUTING.md           How to contribute
├── CHANGELOG.md              Version history
└── README.md                 Project overview
```

## After publishing

- Star your own repo so it shows up on your profile
- Add a topic: `arc-agi`, `arc-prize-2026`, `reinforcement-learning`, `scientific-discovery`
- Optional: enable GitHub Discussions for Q&A
- Optional: add a codecov.io integration for coverage history
