"""validate_kaggle_imports.py
Run this locally (after 'pip install -r requirements.txt') to catch import
namespace errors before pushing to Kaggle.
Usage:
    python scripts\validate_kaggle_imports.py
"""
import sys, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

errors = []

def check(name, import_stmt):
    try:
        exec(import_stmt)
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        errors.append((name, e))

print("Checking standard deps...")
check("torch", "import torch")
check("numpy", "import numpy as np")
check("pandas", "import pandas as pd")
check("sklearn", "from sklearn.model_selection import train_test_split; from sklearn import metrics")
check("tqdm", "from tqdm import tqdm")
check("transformers", "from transformers import AutoTokenizer, AutoModel; from transformers.optimization import get_cosine_schedule_with_warmup")
check("bitsandbytes", "import bitsandbytes")
check("accelerate", "import accelerate")
check("sentencepiece", "import sentencepiece")

print("\nChecking Lightning API (the usual Kaggle gotcha)...")
check("lightning.Trainer", "from lightning.pytorch import Trainer")
check("lightning.seed_everything", "from lightning.pytorch import seed_everything")
check("lightning.LightningModule", "from lightning.pytorch import LightningModule")
check("lightning.callbacks", "from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback")
check("lightning.loggers", "from lightning.pytorch.loggers import CSVLogger")

print("\nChecking LogDLR modules...")
check("LogDLR.transformer", "from LogDLR.model.modules.transformer import TransformerEncoder, TransformerDecoder")
check("LogDLR.grl", "from LogDLR.model.modules.grl import WarmStartGradientReverseLayer")
check("LogDLR.logdlr", "from LogDLR.model.logdlr.logdlr import LogDLR")
check("LogDLR.ae", "from LogDLR.model.logdlr.ae import AE")
check("LogDLR.data_utils", "from LogDLR.utils.data import CycleZip, Chain")

print("\n" + ("=" * 50))
if errors:
    print(f"FAILED: {len(errors)} import(s) broken. Fix before Kaggle run.")
    for name, e in errors:
        print(f"  - {name}: {e}")
    sys.exit(1)
else:
    print("ALL IMPORTS OK. Safe to push and run on Kaggle.")
    sys.exit(0)
