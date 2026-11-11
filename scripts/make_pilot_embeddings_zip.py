"""make_pilot_embeddings_zip.py
Creates the pilot embeddings Kaggle Dataset zip with POSIX forward-slash paths.
Run from repo root:  python scripts\make_pilot_embeddings_zip.py
"""
import os
import zipfile
from pathlib import Path

SRC_DIR    = Path("E:/btp_oct/pilot_output/loglm_embeddings")
OUTPUT_ZIP = Path("E:/btp_oct/LogLDR-pp/pilot_embeddings.zip")

if not SRC_DIR.exists():
    raise FileNotFoundError(f"Source directory not found: {SRC_DIR}")

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(SRC_DIR.iterdir()):
        if f.is_file():
            arcname = f"loglm_embeddings/{f.name}"
            zf.write(f, arcname)

size_mb = OUTPUT_ZIP.stat().st_size / 1024 / 1024
print(f"Done! {OUTPUT_ZIP} ({size_mb:.1f} MB)")
print("Upload this to Kaggle as Dataset: logdlr-pilot-embeddings")
