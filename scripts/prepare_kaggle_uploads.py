"""
prepare_kaggle_uploads.py
=========================
Run this script from your Windows machine to create the two zip files
you need to upload to Kaggle as Datasets.

Usage:
    cd E:\btp_oct\LogLDR-pp
    python scripts\prepare_kaggle_uploads.py

It will create two files in the project root:
  1. kaggle_upload_logdlr_data.zip   -> raw .log files
  2. kaggle_upload_logdlr_code.zip   -> the code + fixed kaggle_e1_e5.py

If your .log files live outside the repo, the script will ask for the path.
"""
import os
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def ask_for_dataset_folder():
    """Prompt user for the folder that contains Thunderbird/, BGL/, Spirit/ subfolders."""
    print("\nCould not find .log files inside the repo.")
    print("Please enter the full path to the folder that contains:")
    print("   Thunderbird/Thunderbird.log")
    print("   BGL/BGL.log")
    print("   Spirit/Spirit.log")
    user_path = input("Path: ").strip().strip('"').strip("'")
    p = Path(user_path)
    if not p.exists():
        print(f"ERROR: Path does not exist: {p}")
        sys.exit(1)
    return p


def find_dataset_root():
    """Try common locations for the datasets inside / near the repo."""
    candidates = [
        REPO_ROOT / "LogDLR" / "dataset",
        REPO_ROOT / "dataset",
        REPO_ROOT.parent / "dataset",
        REPO_ROOT / "data",
    ]
    for c in candidates:
        if (c / "BGL" / "BGL.log").exists():
            return c
    return None


def zip_data(dataset_root: Path, out_path: Path):
    """Create kaggle_upload_logdlr_data.zip with the exact folder structure Kaggle expects."""
    files_to_add = [
        (dataset_root / "Thunderbird" / "Thunderbird.log", "LogDLR/dataset/Thunderbird/Thunderbird.log"),
        (dataset_root / "BGL" / "BGL.log", "LogDLR/dataset/BGL/BGL.log"),
        (dataset_root / "Spirit" / "Spirit.log", "LogDLR/dataset/Spirit/Spirit.log"),
    ]
    print(f"\nCreating data archive: {out_path}")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files_to_add:
            if not src.exists():
                print(f"  WARNING: Missing {src} — skipping.")
                continue
            zf.write(src, arcname)
            print(f"  Added {arcname}")
    print(f"Done. Size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


def zip_code(out_path: Path):
    """Create kaggle_upload_logdlr_code.zip containing the repo code."""
    print(f"\nCreating code archive: {out_path}")
    ignore = {".git", "__pycache__", ".pytest_cache", ".idea", ".vscode", "*.pyc", ".ipynb_checkpoints"}
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(REPO_ROOT):
            # prune ignored dirs
            dirs[:] = [d for d in dirs if d not in ignore]
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                full = Path(root) / f
                arcname = str(full.relative_to(REPO_ROOT.parent))
                zf.write(full, arcname)
    print(f"Done. Size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    dataset_root = find_dataset_root()
    if dataset_root is None:
        dataset_root = ask_for_dataset_folder()
    else:
        print(f"Found datasets at: {dataset_root}")

    data_zip = REPO_ROOT / "kaggle_upload_logdlr_data.zip"
    code_zip = REPO_ROOT / "kaggle_upload_logdlr_code.zip"

    zip_data(dataset_root, data_zip)
    zip_code(code_zip)

    print("\n" + "=" * 60)
    print("UPLOAD INSTRUCTIONS")
    print("=" * 60)
    print(f"1. Go to https://www.kaggle.com/datasets")
    print(f"2. Click  'New Dataset'  (or 'Create New Dataset')")
    print(f"3. For the first dataset:")
    print(f"   - Title:  logdlr-data")
    print(f"   - Upload: {data_zip}")
    print(f"   - Visibility: Private (recommended)")
    print(f"   - Click  'Create'")
    print(f"4. Repeat for the second dataset:")
    print(f"   - Title:  logdlr-code")
    print(f"   - Upload: {code_zip}")
    print(f"   - Click  'Create'")
    print(f"5. Open a Kaggle Notebook and add both datasets as Inputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
