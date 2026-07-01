#!/usr/bin/env python3
"""
Download the public replay data from the GitHub release.

The .vtx files are too large for git (33MB total) and live in a release
tagged "Data" on GitHub. This script downloads and extracts them into
the project's data/ directory.

Usage:
    python scripts/download_data.py
"""
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "0xKunalsharma/Alphamoo"
RELEASE_TAG = "Data"
DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    print(f"Downloading data from GitHub release '{RELEASE_TAG}'...")
    print(f"  Repo: {REPO}")
    print(f"  Target: {DATA_DIR}")
    print()

    # Get release info
    api_url = f"https://api.github.com/repos/{REPO}/releases/tags/{RELEASE_TAG}"
    print(f"Fetching release metadata: {api_url}")
    try:
        req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            release = json.loads(resp.read())
    except Exception as e:
        print(f"Failed to fetch release info: {e}")
        sys.exit(1)

    assets = release.get("assets", [])
    if not assets:
        print("Release has no assets. Did you upload the data?")
        sys.exit(1)

    print(f"Found {len(assets)} assets in release '{release.get('name', RELEASE_TAG)}':")
    total_size = 0
    for asset in assets:
        size_mb = asset["size"] / 1024 / 1024
        total_size += asset["size"]
        print(f"  {asset['name']:<60} {size_mb:>6.2f} MB")
    print(f"\nTotal to download: {total_size / 1024 / 1024:.1f} MB")
    print()

    # Check if any are zip files (preferred) — if so, just download the zip
    zip_assets = [a for a in assets if a["name"].endswith(".zip")]
    vtx_assets = [a for a in assets if a["name"].endswith(".vtx")]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if zip_assets:
        # Download the zip and extract
        for asset in zip_assets:
            print(f"Downloading {asset['name']}...")
            download_to(asset, DATA_DIR / asset["name"])
            print("Extracting...")
            with zipfile.ZipFile(DATA_DIR / asset["name"]) as zf:
                zf.extractall(DATA_DIR)
            print(f"  Extracted to {DATA_DIR}")
    elif vtx_assets:
        # Download individual .vtx files
        print(f"No zip found; downloading {len(vtx_assets)} individual .vtx files...")
        for i, asset in enumerate(vtx_assets, 1):
            print(f"  [{i}/{len(vtx_assets)}] {asset['name']}...")
            download_to(asset, DATA_DIR / asset["name"])
    else:
        print("No .vtx or .zip files found in release!")
        sys.exit(1)

    # Verify
    vtx_files = list(DATA_DIR.glob("*.vtx"))
    print(f"\n✓ Downloaded {len(vtx_files)} .vtx files to {DATA_DIR}")
    if len(vtx_files) == 25:
        print("✓ All 25 public demo games present")
    else:
        print(f"⚠ Expected 25, got {len(vtx_files)}")


def download_to(asset: dict, target: Path):
    """Download an asset to a file with progress."""
    url = asset["browser_download_url"]
    expected_size = asset["size"]

    req = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(target, "wb") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)

    actual_size = target.stat().st_size
    if actual_size != expected_size:
        print(f"  ⚠ Size mismatch: expected {expected_size}, got {actual_size}")


if __name__ == "__main__":
    main()
