#!/usr/bin/env python3
"""
Download KOI (Kepler cumulative), TOI, and K2 planets & candidates full tables
from the NASA Exoplanet Archive using the TAP sync endpoint (preferred).
Saves CSVs into ./data/.

Works with:
  - KOI cumulative: table name = cumulative
  - TOI: table name = toi
  - K2 planets & candidates: table name = k2pandc

References: Exoplanet Archive TAP/API documentation.
"""
import os
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

BASE_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
# Older API fallback (nph-nstedAPI)
BASE_OLD_API = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"

# Map friendly names -> TAP table names
TABLES = {
    "koi": "cumulative",    # KOI cumulative delivery
    "toi": "toi",            # TESS Objects of Interest
    "k2": "k2pandc"         # K2 Planets & Candidates
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_tap_table(table_name: str, out_path: Path, timeout: int = 300, max_retries: int = 3) -> bool:
    """
    Download a full table using the TAP sync interface.
    Returns True on success, False on failure.
    """
    params = {
        "query": f"select * from {table_name}",
        "format": "csv"
    }
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            print(f"[TAP] Requesting table '{table_name}' (attempt {attempt})...")
            with requests.get(BASE_TAP, params=params, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    print(f"  -> TAP returned status {r.status_code}. Content: {r.text[:500]}")
                    break
                total = r.headers.get("Content-Length")
                if total:
                    try:
                        total = int(total)
                    except Exception:
                        total = None
                # stream to file
                with open(out_path, "wb") as f:
                    downloaded = 0
                    chunk_size = 1024 * 16
                    start = time.time()
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                    elapsed = time.time() - start
                    size_mb = downloaded / (1024 * 1024)
                    print(f"  -> Saved {out_path} ({size_mb:.2f} MB) in {elapsed:.1f}s")
                return True
        except requests.exceptions.RequestException as e:
            print(f"  -> TAP request error: {e}. Retrying in 3s...")
            time.sleep(3)
    return False


def download_old_api(table_name: str, out_path: Path, timeout: int = 300) -> bool:
    """
    Fallback to older API endpoint which uses 'table' and 'format' params.
    """
    params = {"table": table_name, "format": "csv"}
    try:
        print(f"[OLD API] Attempting old API for table '{table_name}'...")
        with requests.get(BASE_OLD_API, params=params, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                print(f"  -> old API returned status {r.status_code}")
                return False
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 16):
                    if chunk:
                        f.write(chunk)
            print(f"  -> Saved {out_path} via old API")
            return True
    except requests.exceptions.RequestException as e:
        print(f"  -> old API request error: {e}")
        return False


def detect_label_col(df: pd.DataFrame) -> Optional[str]:
    """Return a likely disposition/label column name if present."""
    candidates = [
        "koi_disposition", "koi_disposition_using_kepler_data", "koi_pdisposition", "disposition",
        "tfopwg_disp", "tfopwg_disposition", "TFOWPG_Disposition", "toi_disposition",
        "archive_disposition", "archive_disposition"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: common variants
    for c in df.columns:
        if "disposition" in c.lower() or "disp" in c.lower():
            return c
    return None


def small_preview(csv_path: Path, nrows: int = 5):
    """Load a small preview and print shape + head + label counts (if any)."""
    print(f"\nPreviewing {csv_path} (first {nrows} rows)...")
    try:
        df_head = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
        print(df_head.head(nrows))
        # now load just column names and a tiny sample of full file to detect total rows (fast method)
        # Use pandas to get total rows (may read entire file); if file is large this will still usually be OK.
        df_full = pd.read_csv(csv_path, nrows=0)  # get columns only
        label_col = detect_label_col(df_full)
        # For row count, use a fast count by iterating lines
        with open(csv_path, "rb") as fh:
            row_count = sum(1 for _ in fh) - 1  # subtract header
        print(f"Columns: {len(df_full.columns)}; Approx rows: {row_count}")
        if label_col:
            # safe partial read for value counts
            vc = pd.read_csv(csv_path, usecols=[label_col], low_memory=False)[label_col].value_counts(dropna=False)
            print(f"Detected label column: '{label_col}'")
            print(vc)
        else:
            print("No obvious label/disposition column detected in this table preview.")
    except Exception as e:
        print("  -> Preview/read error:", e)


def main():
    print("Starting download of exoplanet archive tables (KOI, TOI, K2)...")
    for shortname, table in TABLES.items():
        filename = {
            "koi": f"koi_{table}.csv",
            "toi": "toi.csv",
            "k2": f"k2_{table}.csv"
        }.get(shortname, f"{shortname}_{table}.csv")
        out_path = DATA_DIR / filename
        # skip if already exists
        if out_path.exists():
            print(f"\nFile {out_path} already exists — skipping download.")
            small_preview(out_path, nrows=3)
            continue
        success = download_tap_table(table, out_path)
        if not success:
            print("  -> TAP download failed, trying older API fallback...")
            success = download_old_api(table, out_path)
        if not success:
            print(f"ERROR: Failed to download table '{table}'. See docs or try again later.")
        else:
            # quick preview (first rows + counts)
            small_preview(out_path, nrows=5)

    print("\nAll done. Check the ./data/ folder for CSV files.")


if __name__ == "__main__":
    main()
