"""make_kaggle_data_zip.py
Creates the Kaggle Dataset zip with POSIX forward-slash paths.
Run from repo root:  python scripts\make_kaggle_data_zip.py
"""
import os
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SOURCE_DIR = Path("E:/btp_oct/data/raw")
OUTPUT_ZIP = Path("E:/btp_oct/LogLDR-pp/kaggle_upload_logdlr_data.zip")
NROWS = 1_000_000   # lines per dataset

FILES = {
    "Thunderbird": SOURCE_DIR / "Thunderbird10M.log",
    "BGL":         SOURCE_DIR / "BGL.log",
    "Spirit":      SOURCE_DIR / "Spirit1G.log",
}
# ---------------------------------------------------------------------------

def trim_and_zip():
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, src in FILES.items():
            if not src.exists():
                raise FileNotFoundError(f"Missing source file: {src}")

            arcname = f"LogDLR/dataset/{name}/{name}.log"
            print(f"Processing {name} from {src} ...")

            lines_written = 0
            with open(src, "r", encoding="utf-8", errors="ignore") as rf:
                for i, line in enumerate(rf):
                    if i >= NROWS:
                        break
                    # Write line-by-line into zip using writestr
                    # For the first line, we start fresh; for subsequent lines we append
                    # zipfile.writestr doesn't support append, so we collect in memory
                    # Actually for 1M lines (~150MB) it's fine to buffer in a list
                    pass  # placeholder - we'll refactor

    # Better approach: use a temp directory, write files, then zip with arcnames
    # Actually, let's write to a temp file first then add to zip

if __name__ == "__main__":
    WORK_DIR = Path("E:/btp_oct/LogLDR-pp/_kaggle_zip_work")
    WORK_DIR.mkdir(exist_ok=True)

    for name, src in FILES.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")

        dest = WORK_DIR / "LogDLR" / "dataset" / name / f"{name}.log"
        dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"Trimming {name} -> {dest}")
        with open(src, "r", encoding="utf-8", errors="ignore") as rf, \
             open(dest, "w", encoding="utf-8", errors="ignore") as wf:
            for i in range(NROWS):
                line = rf.readline()
                if not line:
                    break
                wf.write(line)
        lines_written = i + 1 if i < NROWS - 1 else NROWS
        print(f"  Wrote {lines_written} lines ({dest.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\nCreating zip: {OUTPUT_ZIP}")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(WORK_DIR):
            for f in files:
                abs_path = Path(root) / f
                arcname = str(abs_path.relative_to(WORK_DIR)).replace(os.sep, "/")
                zf.write(abs_path, arcname)

    # Cleanup
    import shutil
    shutil.rmtree(WORK_DIR, ignore_errors=True)

    size_mb = OUTPUT_ZIP.stat().st_size / 1024 / 1024
    print(f"Done! {OUTPUT_ZIP} ({size_mb:.1f} MB)")
    print("Upload this to Kaggle as Dataset: logdlr-data")
