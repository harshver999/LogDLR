# =============================================================================
# LogDLR++ E1–E5 — Kaggle GPU Notebook
# =============================================================================
# **Purpose:** Run experiments E1–E5 for a single source→target pair per session.
# **Workflow on Kaggle:**
#   1. Create a new Notebook → turn GPU ON, Internet ON.
#   2. Clone your GitHub repo:  !git clone https://github.com/<user>/LogLDR-pp.git
#   3. (Optional) If your LogLM model is gated, add HF_TOKEN via Kaggle Secrets.
#   4. Set env vars (EXPERIMENT, SOURCE, TARGET, NROWS, etc.) in a cell.
#   5. Run:  !python /kaggle/working/LogLDR-pp/kaggle_e1_e5.py
#
# **Datasets:** The script auto-downloads Thunderbird / BGL / Spirit from Zenodo
# (LogHub) if they are not already present.  This avoids uploading multi-GB files.
#
# **Time limit:** 12 h/session. LogLM embedding extraction is checkpointed so you
# can resume across sessions by downloading the chunk files and re-uploading them
# as a Kaggle Dataset.
# =============================================================================
import os

# Experiment identifier: "E1", "E2", "E3", "E4", "E5"
# Default to the smallest pair (BGL <-> Spirit) to maximize chance of finishing
# within the 12-hour Kaggle limit.Override via env vars if needed.
EXPERIMENT = os.environ.get("EXPERIMENT", "E1")
SOURCE     = os.environ.get("SOURCE", "BGL")
TARGET     = os.environ.get("TARGET", "Spirit")

# Override these via notebook env vars if you don't want to edit the file.
NROWS          = int(os.environ.get("NROWS", "10000000"))      # Reduce to 1_000_000 for pilot
MAX_EPOCHS     = int(os.environ.get("MAX_EPOCHS", "50"))
LOGLM_MODEL_NAME = os.environ.get("LOGLM_MODEL_NAME", "meta-llama/Llama-2-7b-hf")
USE_8BIT       = os.environ.get("USE_8BIT", "1") == "1"

# Paths: derive PROJECT_ROOT from where this script lives (repo root).
# If you ran: !git clone ... /kaggle/working/LogDLR
# then __file__ == /kaggle/working/LogDLR/kaggle_e1_e5.py
_PROJECT_ROOT_FROM_FILE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", _PROJECT_ROOT_FROM_FILE)

# Data directory is independent of the repo name to avoid path confusion.
# It lives directly in /kaggle/working so it is persisted as output.
WORKING_DIR  = "/kaggle/working"
DATA_ROOT    = os.environ.get("DATA_ROOT", os.path.join(WORKING_DIR, "datasets"))

# Dataset log paths
LOG_FILE_S = os.path.join(DATA_ROOT, SOURCE, f"{SOURCE}.log")
LOG_FILE_T = os.path.join(DATA_ROOT, TARGET, f"{TARGET}.log")

# =============================================================================
# IMPORTANT — LogLM vs LLaMA-2 fallback
# =============================================================================
# The official LogLM repository (https://github.com/lunyiliu/LogLM) provides
# reproduction instructions but does NOT release the fine-tuned model weights.
# Because fine-tuning a 7B model is impossible on the free Kaggle tier, we fall
# back to the frozen *base* LLaMA-2-7B checkpoint.
#
# IMPACT:
#   - E1 will measure the effect of switching from SBERT to a general 7B LLM,
#     NOT the effect of the specialized LogLM checkpoint.
#   - E2-E5 improvements (structure prep, prototypes, cosine+huber, conformal)
#     are still valid and comparable across experiments.
#   - If LogLM weights ever become publicly available, simply change
#     LOGLM_MODEL_NAME below and re-run.
# =============================================================================

# Model configuration
MAX_LENGTH        = 256                          # LogLM truncation length
LOGLM_HIDDEN_SIZE = 4096                         # 4096 for LLaMA-2-7B base
EMBED_DIM         = 384                          # Autoencoder internal dim (same as baseline)
LATENT_DIM        = 512
NHEAD             = 16
NUM_LAYERS        = 4
WINDOW_SIZE       = 20
STEP_SIZE         = 4
BATCH_SIZE        = 32
EVAL_BATCH_SIZE   = 256
LR                = 1e-4
ALPHA             = 0.1   # L2 regularization weight
BETA              = 1.0   # Domain loss weight
NUM_WARMUP_STEPS  = 4000
RANDOM_SEED       = 8

TRAIN_SIZE_S = 100_000
TRAIN_SIZE_T = 10_000
NUM_FIELDS   = 10
MAX_TEST_SIZE = 100_000

# E3/E4/E5 specific
USE_PROTOTYPE     = EXPERIMENT in ("E3", "E4", "E5")
N_PROTOTYPES      = 4
PROTOTYPE_GAMMA   = 0.1
TAU               = 0.2
USE_COSINE_HUBER  = EXPERIMENT in ("E4", "E5")
LAMBDA_COS        = 0.5
HUBER_DELTA       = 1.0
USE_CONFORMAL     = EXPERIMENT == "E5"
CONFORMAL_ALPHA   = 0.05
N_CONFORMAL_CAL   = 10_000   # Normal samples used for conformal calibration

# Preprocessing
USE_STRUCTURE_PREPROCESS = EXPERIMENT in ("E2", "E3", "E4", "E5")

# Embedding cache
EMBED_CACHE_DIR = os.path.join(WORKING_DIR, "loglm_embeddings")

# Training checkpointing (Lightning saves automatically; we also copy best to working)
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints", f"{EXPERIMENT}_{SOURCE}_{TARGET}")

print(f"Running {EXPERIMENT}: {SOURCE} -> {TARGET}")
print(f"Model: {LOGLM_MODEL_NAME}  (frozen, 8-bit={USE_8BIT})")
print(f"Prototype: {USE_PROTOTYPE}, Cosine+Huber: {USE_COSINE_HUBER}, Conformal: {USE_CONFORMAL}")

# %% [code]
# =============================================================================
# OPTIONAL: Install & import dependencies
# =============================================================================
import subprocess, sys

_MISSING_DEPS = ["bitsandbytes", "accelerate", "sentencepiece", "lightning", "sentence-transformers", "gensim"]
_install_cmd = [sys.executable, "-m", "pip", "install", "-q"] + _MISSING_DEPS

for pkg in _MISSING_DEPS:
    try:
        __import__(pkg.replace("-", "_"))  # package names use underscores
    except ImportError:
        print(f"Installing missing dependency: {pkg} ...")
        subprocess.check_call(_install_cmd)
        break  # one install call covers all

sys.path.append(PROJECT_ROOT)

# Explicit Lightning imports (guaranteed namespace in v2.x)
from lightning.pytorch import Trainer, seed_everything, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import CSVLogger

import re
import time
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_cosine_schedule_with_warmup

# Existing LogDLR components
from LogDLR.model.modules.transformer import TransformerEncoder, TransformerDecoder
from LogDLR.model.modules.grl import WarmStartGradientReverseLayer
from LogDLR.utils.data import CycleZip, Chain

# Kaggle / PyTorch speed-ups for T4
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    device = 'cuda'
else:
    device = 'cpu'
print("Device:", device, "CUDA:", torch.cuda.is_available())

# =============================================================================
# STEP 1 — Verify datasets exist (NO auto-download; upload data as Kaggle Dataset)
# =============================================================================
print("Checking datasets...")
if not os.path.exists(LOG_FILE_S):
    raise FileNotFoundError(
        f"Source log not found: {LOG_FILE_S}\n"
        f"Please upload the datasets as a Kaggle Dataset and set DATA_ROOT.\n"
        f"Expected structure: <DATA_ROOT>/{SOURCE}/{SOURCE}.log"
    )
if not os.path.exists(LOG_FILE_T):
    raise FileNotFoundError(
        f"Target log not found: {LOG_FILE_T}\n"
        f"Please upload the datasets as a Kaggle Dataset and set DATA_ROOT.\n"
        f"Expected structure: <DATA_ROOT>/{TARGET}/{TARGET}.log"
    )
print("Datasets verified.")

# %% [code]
# =============================================================================
# STEP 2 — Preprocessing
# =============================================================================

def original_preprocess(text: str) -> str:
    """LogDLR baseline preprocessing (simple NLP cleaning)."""
    from gensim.utils import simple_preprocess
    return ' '.join(simple_preprocess(text, max_len=100))


def structure_preserving_preprocess(text: str) -> str:
    """
    E2–E5 preprocessing.
    Retains parameter identity via typed placeholders and preserves more text.
    """
    text = text.lower()
    # IP addresses -> placeholder
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<ip>', text)
    # Hex addresses -> placeholder
    text = re.sub(r'\b0x[0-9a-fA-F]+\b', '<hex>', text)
    # Very long numbers (IDs, ports, timestamps) -> placeholder
    text = re.sub(r'\b\d{5,}\b', '<num>', text)
    # Retry / port / count patterns
    text = re.sub(r'retry\s*=\s*\d+', 'retry=<cnt>', text)
    text = re.sub(r'port\s*=\s*\d+', 'port=<port>', text)
    # Clean and tokenize while keeping placeholders
    from gensim.utils import simple_preprocess
    tokens = simple_preprocess(text, max_len=100)
    return ' '.join(tokens)


preprocess_fn = structure_preserving_preprocess if USE_STRUCTURE_PREPROCESS else original_preprocess

# %% [code]
# =============================================================================
# STEP 3 — LogLM Encoder with chunked resume (CRITICAL for 12h Kaggle limit)
# =============================================================================

class LogLMEncoder:
    """Loads a frozen LLM and extracts mean-pooled last-layer hidden states."""

    def __init__(self, model_name: str, use_8bit: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_auth_token=os.environ.get("HF_TOKEN", True),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        token = os.environ.get("HF_TOKEN", None)
        load_kwargs = {
            "trust_remote_code": True,
            "token": token,
            "output_hidden_states": True,
            "torch_dtype": torch.float16,
        }
        if use_8bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = {"": device}

        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=8):
        embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="LogLM encode"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            # device_map="auto" handles placement, but let's be safe
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]                     # [batch, seq, hidden]
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # [batch, seq, 1]
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)     # [batch, hidden]
            embs.append(emb.cpu())
        return torch.cat(embs, dim=0)


def extract_and_cache_embeddings(unique_logs, dataset_name, encoder, chunk_size=2000):
    """
    Resume-friendly embedding extraction.
    Saves chunks to /kaggle/working/loglm_embeddings/ so you can download
    them and resume in a later session if the VM dies.
    """
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    done_path = os.path.join(EMBED_CACHE_DIR, f"{dataset_name}_done.pkl")

    done = set()
    if os.path.exists(done_path):
        with open(done_path, "rb") as f:
            done = pickle.load(f)
        print(f"Resuming {dataset_name}: {len(done)}/{len(unique_logs)} already cached.")

    remaining = [log for log in unique_logs if log not in done]
    if not remaining:
        print(f"All embeddings cached for {dataset_name}.")
        return

    for i in range(0, len(remaining), chunk_size):
        chunk = remaining[i:i + chunk_size]
        embs = encoder.encode(chunk, batch_size=4 if USE_8BIT else 8)

        chunk_file = os.path.join(EMBED_CACHE_DIR, f"{dataset_name}_chunk_{len(done)}.pt")
        buffer = {}
        for log, emb in zip(chunk, embs):
            buffer[log] = emb
            done.add(log)
        torch.save(buffer, chunk_file)

        with open(done_path, "wb") as f:
            pickle.dump(done, f)
        print(f"Saved {chunk_file} | total done: {len(done)}")

    print(f"Extraction complete for {dataset_name}.")


def load_cached_embeddings(dataset_name):
    """Loads all cached chunks for a dataset into a single dict {text: tensor}."""
    emb_dict = {}
    pattern = os.path.join(EMBED_CACHE_DIR, f"{dataset_name}_chunk_*.pt")
    import glob
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split("_chunk_")[-1].replace(".pt", "")))
    if not files:
        raise FileNotFoundError(f"No cached embeddings found for {dataset_name} in {EMBED_CACHE_DIR}")
    for f in files:
        chunk = torch.load(f, map_location="cpu")
        emb_dict.update(chunk)
    print(f"Loaded {len(emb_dict)} cached embeddings for {dataset_name}.")
    return emb_dict


# %% [code]
# =============================================================================
# STEP 4 — Data module ( consumes cached embeddings )
# =============================================================================

class LogSeqDataset(Dataset):
    def __init__(self, indexes, labels, logs, window_size, embedding):
        super().__init__()
        self.indexes = indexes
        self.labels = labels
        self.logs = logs
        self.window_size = window_size
        self.embedding = embedding

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, i):
        seq = [self.embedding[log] for log in self.logs[self.indexes[i]:self.indexes[i] + self.window_size]]
        return torch.stack(seq), self.labels[i]


class LogDataModule:
    def __init__(
        self,
        log_file: str,
        embedding: dict,          # text -> tensor dict
        preprocess_fn,
        num_fields: int = 10,
        nrows: int = 10**7,
        window_size: int = 20,
        step_size: int = 4,
        batch_size: int = 32,
        eval_batch_size: int = 256,
        train_size: int = 100000,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_test_size: int = 0,
        random_seed: int = 8,
    ):
        self.log_file = log_file
        self.embedding = embedding
        self.preprocess_fn = preprocess_fn
        self.num_fields = num_fields
        self.nrows = nrows
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.train_size = train_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_test_size = max_test_size
        self.random_seed = random_seed
        self.prepare_data()

    def prepare_data(self):
        self.logs = []
        labels = []
        with open(self.log_file, "r", encoding="utf-8", errors="ignore") as f:
            for i, row in enumerate(f):
                if i >= self.nrows:
                    break
                fields = row.strip().split(maxsplit=self.num_fields - 1)
                if len(fields) == self.num_fields:
                    labels.append(fields[0] != "-")
                    self.logs.append(self.preprocess_fn(fields[-1]))

        # Safety check: every unique log must exist in embedding cache
        missing = set(self.logs) - set(self.embedding.keys())
        if missing:
            raise ValueError(f"{len(missing)} logs missing from embedding cache! Example: {next(iter(missing))}")

        normal, anomaly = [], []
        for i in range(0, len(self.logs) - self.window_size + 1, self.step_size):
            if any(labels[i:i + self.window_size]):
                anomaly.append(i)
            else:
                normal.append(i)

        self.train_x, normal = train_test_split(
            normal, train_size=self.train_size, random_state=self.random_seed
        )
        self.eval_x, normal = train_test_split(
            normal, train_size=int(self.train_size * 0.1), random_state=self.random_seed
        )
        self.train_y = [0] * len(self.train_x)
        self.eval_y = [0] * len(self.eval_x)

        val_test_y = [0] * len(normal) + [1] * len(anomaly)
        self.val_x, self.test_x, self.val_y, self.test_y = train_test_split(
            normal + anomaly,
            val_test_y,
            train_size=int(self.train_size * 0.1),
            stratify=val_test_y,
            random_state=self.random_seed,
        )
        if self.max_test_size > 0 and len(self.test_x) > self.max_test_size:
            self.test_x, _, self.test_y, _ = train_test_split(
                self.test_x,
                self.test_y,
                train_size=self.max_test_size,
                stratify=self.test_y,
                random_state=self.random_seed,
            )

        self.data_dict = {
            "train": (self.train_x, self.train_y),
            "eval": (self.eval_x, self.eval_y),
            "val": (self.val_x, self.val_y),
            "test": (self.test_x, self.test_y),
        }

    def data_loader(self, mode):
        bs = self.batch_size if mode == "train" else self.eval_batch_size
        return DataLoader(
            LogSeqDataset(*self.data_dict[mode], self.logs, self.window_size, self.embedding),
            batch_size=bs,
            shuffle=(mode == "train"),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_normal_train_sequences(self):
        """Returns all normal training sequences as a stacked tensor for prototype K-Means."""
        idxs, _ = self.data_dict["train"]
        seqs = []
        for idx in idxs:
            seq = torch.stack([self.embedding[log] for log in self.logs[idx:idx + self.window_size]])
            seqs.append(seq)
        return torch.stack(seqs)


# %% [code]
# =============================================================================
# STEP 5 — Losses & Calibration
# =============================================================================

class ReconstructionLoss(nn.Module):
    """MSE (E1–E3) or Cosine + Huber (E4–E5)."""

    def __init__(self, loss_type="mse", lambda_cos=0.5, delta=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_cos = lambda_cos
        self.delta = delta

    def forward(self, recon_x, x):
        # x, recon_x: [batch, window, embed_dim]
        if self.loss_type == "mse":
            return F.mse_loss(recon_x, x, reduction="none").sum(dim=(1, 2))

        # Cosine per token
        cos_sim = F.cosine_similarity(recon_x, x, dim=-1)          # [batch, window]
        l_cos = (1.0 - cos_sim).sum(dim=1)                         # [batch]

        # Huber per element
        err = x - recon_x
        abs_err = err.abs()
        quad = 0.5 * err ** 2
        lin = self.delta * (abs_err - 0.5 * self.delta)
        l_huber = torch.where(abs_err <= self.delta, quad, lin).sum(dim=(1, 2))

        if self.loss_type == "cosine_huber":
            return self.lambda_cos * l_cos + (1.0 - self.lambda_cos) * l_huber
        elif self.loss_type == "cosine":
            return l_cos
        elif self.loss_type == "huber":
            return l_huber
        else:
            raise ValueError(self.loss_type)


class PrototypeContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss over K normal prototypes."""

    def __init__(self, tau=0.2):
        super().__init__()
        self.tau = tau

    def forward(self, z, prototypes):
        # z: [batch, latent_dim], prototypes: [K, latent_dim]
        z = F.normalize(z, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        sim = torch.mm(z, prototypes.t()) / self.tau   # [batch, K]
        # Positive = nearest prototype (highest similarity)
        pos = sim.argmax(dim=1)
        return F.cross_entropy(sim, pos)


class ConformalCalibrator:
    """Split-conformal p-value anomaly calibration using normal calibration scores."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.calibration_scores = None

    def fit(self, scores):
        # scores: anomaly/non-conformity scores from a normal-only calibration set
        self.calibration_scores = np.array(scores)

    def predict(self, scores):
        scores = np.array(scores)
        n = len(self.calibration_scores)
        # p(x) = (1 + #{cal_scores >= score(x)}) / (n+1)
        # Higher score -> lower p-value -> more anomalous
        p_values = np.array([ (1 + np.sum(self.calibration_scores >= s)) / (n + 1) for s in scores ])
        preds = (p_values < self.alpha).astype(int)
        return preds, p_values


# %% [code]
# =============================================================================
# STEP 6 — Lightning Model (LogDLR++)
# =============================================================================

class LogDLRpp(LightningModule):
    def __init__(self, num_training_steps, embed_dim=384, latent_dim=512, nhead=16,
                 num_layers=4, window_size=20, lr=1e-4, weight_decay=0.0,
                 alpha=0.1, beta=1.0, num_warmup_steps=4000,
                 use_cosine_huber=False, lambda_cos=0.5, huber_delta=1.0,
                 use_prototype=False, n_prototypes=4, prototype_gamma=0.1, tau=0.2,
                 use_conformal=False, conformal_alpha=0.05,
                 loglm_hidden_size=4096,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["num_training_steps"])
        self.num_training_steps = num_training_steps

        # Project raw LogLM vectors down to the autoencoder dimension
        self.input_proj = nn.Linear(loglm_hidden_size, embed_dim)

        self.encoder = TransformerEncoder(embed_dim, latent_dim, nhead, num_layers, window_size)
        self.decoder = TransformerDecoder(embed_dim, latent_dim, nhead, num_layers, window_size)

        self.grl = WarmStartGradientReverseLayer()
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim // 2, 2),
        )

        loss_type = "cosine_huber" if use_cosine_huber else "mse"
        self.recon_loss_fn = ReconstructionLoss(loss_type=loss_type, lambda_cos=lambda_cos, delta=huber_delta)
        self.proto_loss_fn = PrototypeContrastiveLoss(tau=tau) if use_prototype else None

        self.prototypes = None
        self.use_prototype = use_prototype
        self.prototype_gamma = prototype_gamma
        self.n_prototypes = n_prototypes

        self.use_conformal = use_conformal
        self.conformal_alpha = conformal_alpha
        self.conformal_calibrator = None

    def forward(self, x_raw):
        x = self.input_proj(x_raw)          # [batch, window, embed_dim]
        z = self.encoder(x)
        recon_x = self.decoder(z)
        pred = self.discriminator(self.grl(z))
        return {"z": z, "recon_x": recon_x, "pred": pred, "x_proj": x}

    def recon_loss(self, x, output):
        # Reconstruct the projected input, not the raw LogLM vector
        return self.recon_loss_fn(output["recon_x"], output["x_proj"])

    @staticmethod
    def embedding_loss(z):
        return 0.5 * torch.linalg.norm(z, dim=-1) ** 2

    @staticmethod
    def domain_loss(pred, label):
        return F.cross_entropy(pred, label.long().to(pred.device))

    def loss_function(self, x, output, domain_label):
        rec = self.recon_loss(x, output).mean()
        emb = self.embedding_loss(output["z"]).mean()
        ae = rec + self.hparams.alpha * emb
        dom = self.domain_loss(output["pred"], domain_label)
        loss = ae + self.hparams.beta * dom
        if self.use_prototype and self.prototypes is not None:
            proto = self.proto_loss_fn(output["z"], self.prototypes.to(output["z"].device))
            loss = loss + self.prototype_gamma * proto
            self.log("train_proto", proto)
        self.log("train_rec", rec)
        return loss

    def training_step(self, batch, batch_idx):
        (x_S, _), (x_T, _) = batch
        x = torch.cat((x_S, x_T))
        output = self.forward(x)
        domain_label = torch.cat((torch.zeros(x_S.shape[0]), torch.ones(x_T.shape[0]))).long().to(x.device)
        loss = self.loss_function(x, output, domain_label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, _), i = batch
        output = self.forward(x)
        domain_label = torch.full((x.shape[0],), i, dtype=torch.long, device=x.device)
        rec = self.recon_loss(x, output).mean()
        emb = self.embedding_loss(output["z"]).mean()
        ae = rec + self.hparams.alpha * emb
        dom = self.domain_loss(output["pred"], domain_label)
        loss = ae + self.hparams.beta * dom
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        return opt

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        self.lr_scheduler.step()

    # ---------------- Inference helpers ----------------
    def on_predict_start(self):
        self.recon_losses = []
        self.labels = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, label = batch
        output = self.forward(x)
        self.recon_losses.extend(self.recon_loss(x, output).tolist())
        self.labels.extend(label.tolist())

    def on_predict_end(self):
        if not self.use_conformal:
            self.th = self._threshold()

    def on_test_start(self):
        self.on_predict_start()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.predict_step(batch, batch_idx, dataloader_idx)

    def on_test_end(self):
        if self.use_conformal and self.conformal_calibrator is not None:
            preds, p_vals = self.conformal_calibrator.predict(self.recon_losses)
            p, r, f1, _ = metrics.precision_recall_fscore_support(self.labels, preds, average="binary")
            print(f"Conformal  Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")
        else:
            preds = [0 if s <= self.th else 1 for s in self.recon_losses]
            p, r, f1, _ = metrics.precision_recall_fscore_support(self.labels, preds, average="binary")
            print(f"Threshold  Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")

    def _threshold(self, n=100):
        mean_N = np.mean([s for s, y in zip(self.recon_losses, self.labels) if y == 0])
        mean_A = np.mean([s for s, y in zip(self.recon_losses, self.labels) if y == 1])
        step = (mean_A - mean_N) / n
        best_th, best_f1 = mean_N, -1
        th = mean_N
        for _ in range(n):
            preds = [0 if s <= th else 1 for s in self.recon_losses]
            f1 = metrics.f1_score(self.labels, preds, average="binary")
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
            th += step
        return best_th


# %% [code]
# =============================================================================
# STEP 7 — Prototype update callback
# =============================================================================

class PrototypeUpdateCallback(Callback):
    """Re-computes K-Means prototypes from normal source training latents every N epochs."""

    def __init__(self, source_dm, every_n_epochs=1):
        super().__init__()
        self.source_dm = source_dm
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if not pl_module.use_prototype:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        dl = self.source_dm.data_loader("train")
        zs = []
        pl_module.eval()
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(pl_module.device)
                z = pl_module.encoder(x)
                zs.append(z.cpu())
        pl_module.train()

        z_all = torch.cat(zs, dim=0).numpy()
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=pl_module.n_prototypes, random_state=RANDOM_SEED, n_init=10)
        kmeans.fit(z_all)
        pl_module.prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        print(f"[PrototypeCallback] Epoch {trainer.current_epoch + 1}: prototypes updated from {z_all.shape[0]} normal vectors.")


# %% [code]
# =============================================================================
# STEP 8 — Conformal calibration helper
# =============================================================================

def build_conformal_calibrator(model, datamodule, n_max=10000):
    """Builds calibrator using the first n_max normal samples from the val set."""
    val_x, val_y = datamodule.val_x, datamodule.val_y
    normal_idxs = [idx for idx, y in zip(val_x, val_y) if y == 0]
    normal_idxs = normal_idxs[:n_max]

    cal_dataset = LogSeqDataset(normal_idxs, [0]*len(normal_idxs),
                                datamodule.logs, datamodule.window_size, datamodule.embedding)
    cal_loader = DataLoader(cal_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    model.eval()
    scores = []
    with torch.no_grad():
        for x, _ in cal_loader:
            x = x.to(model.device)
            out = model(x)
            scores.extend(model.recon_loss(x, out).cpu().tolist())
    model.train()

    cal = ConformalCalibrator(alpha=model.conformal_alpha)
    cal.fit(scores)
    print(f"[Conformal] Calibrated on {len(scores)} normal validation samples.")
    return cal


# %% [code]
# =============================================================================
# STEP 9 — Main execution
# =============================================================================

def main():
    seed_everything(RANDOM_SEED)

    # ---------------------------------------------------------------
    # 9A. Extract / load LogLM embeddings (ONE TIME PER DATASET)
    # ---------------------------------------------------------------
    # If you already extracted embeddings in a previous Kaggle session,
    # upload them as a Kaggle Dataset and set EMBED_CACHE_DIR.
    #
    # NOTE: LogLM extraction on 10M rows with a 7B model on T4 is VERY slow.
    # If you have the embeddings cached, this loop skips automatically.
    encoder = None
    for ds_name, log_file in [(SOURCE, LOG_FILE_S), (TARGET, LOG_FILE_T)]:
        # Fast dedup scan to know how many unique logs we need
        uniq = set()
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for i, row in enumerate(f):
                if i >= NROWS:
                    break
                fields = row.strip().split(maxsplit=NUM_FIELDS - 1)
                if len(fields) == NUM_FIELDS:
                    uniq.add(preprocess_fn(fields[-1]))
        print(f"{ds_name}: {len(uniq)} unique preprocessed logs to embed.")

        # Load encoder lazily (only if something is missing)
        if encoder is None:
            encoder = LogLMEncoder(LOGLM_MODEL_NAME, use_8bit=USE_8BIT)

        # Run extraction with resume
        extract_and_cache_embeddings(list(uniq), ds_name, encoder, chunk_size=2000)

    # ---------------------------------------------------------------
    # 9B. Load caches and build data modules
    # ---------------------------------------------------------------
    emb_S = load_cached_embeddings(SOURCE)
    emb_T = load_cached_embeddings(TARGET)

    data_S = LogDataModule(
        log_file=LOG_FILE_S,
        embedding=emb_S,
        preprocess_fn=preprocess_fn,
        num_fields=NUM_FIELDS,
        nrows=NROWS,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        train_size=TRAIN_SIZE_S,
        max_test_size=MAX_TEST_SIZE,
        random_seed=RANDOM_SEED,
    )
    data_T = LogDataModule(
        log_file=LOG_FILE_T,
        embedding=emb_T,
        preprocess_fn=preprocess_fn,
        num_fields=NUM_FIELDS,
        nrows=NROWS,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        train_size=TRAIN_SIZE_T,
        max_test_size=MAX_TEST_SIZE,
        random_seed=RANDOM_SEED,
    )

    # ---------------------------------------------------------------
    # 9C. Model
    # ---------------------------------------------------------------
    steps_per_epoch = TRAIN_SIZE_S / BATCH_SIZE
    total_steps = int(MAX_EPOCHS * steps_per_epoch)

    model = LogDLRpp(
        num_training_steps=total_steps,
        embed_dim=EMBED_DIM,
        latent_dim=LATENT_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        window_size=WINDOW_SIZE,
        lr=LR,
        weight_decay=0.0,
        alpha=ALPHA,
        beta=BETA,
        num_warmup_steps=NUM_WARMUP_STEPS,
        use_cosine_huber=USE_COSINE_HUBER,
        lambda_cos=LAMBDA_COS,
        huber_delta=HUBER_DELTA,
        use_prototype=USE_PROTOTYPE,
        n_prototypes=N_PROTOTYPES,
        prototype_gamma=PROTOTYPE_GAMMA,
        tau=TAU,
        use_conformal=USE_CONFORMAL,
        conformal_alpha=CONFORMAL_ALPHA,
        loglm_hidden_size=LOGLM_HIDDEN_SIZE,
    )

    # ---------------------------------------------------------------
    # 9D. Callbacks
    # ---------------------------------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=4, verbose=True)
    proto_cb = PrototypeUpdateCallback(data_S, every_n_epochs=1) if USE_PROTOTYPE else None

    cbs = [checkpoint_cb, early_stop_cb]
    if proto_cb:
        cbs.append(proto_cb)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=cbs,
        logger=CSVLogger(save_dir=CHECKPOINT_DIR),
        enable_progress_bar=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    # ---------------------------------------------------------------
    # 9E. Train
    # ---------------------------------------------------------------
    trainer.fit(
        model=model,
        train_dataloaders=CycleZip(data_S.data_loader("train"), data_T.data_loader("train")),
        val_dataloaders=Chain(data_S.data_loader("eval"), data_T.data_loader("eval")),
    )

    best_path = checkpoint_cb.best_model_path
    print(f"Best checkpoint: {best_path}")
    # Load best for test
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    # ---------------------------------------------------------------
    # 9F. Optional conformal calibration before test
    # ---------------------------------------------------------------
    if USE_CONFORMAL:
        model.conformal_calibrator = build_conformal_calibrator(model, data_T, n_max=N_CONFORMAL_CAL)

    # ---------------------------------------------------------------
    # 9G. Evaluate on validation (threshold selection / calibration gap)
    # ---------------------------------------------------------------
    print(f"\n=== Validation ({TARGET}) ===")
    trainer.predict(model, data_T.data_loader("val"), return_predictions=False)

    # ---------------------------------------------------------------
    # 9H. Evaluate on test
    # ---------------------------------------------------------------
    print(f"\n=== Test ({TARGET}) ===")
    trainer.test(model, data_T.data_loader("test"))

    # ---------------------------------------------------------------
    # 9I. Persist results
    # ---------------------------------------------------------------
    result = {
        "experiment": EXPERIMENT,
        "source": SOURCE,
        "target": TARGET,
        "seed": RANDOM_SEED,
        "best_checkpoint": best_path,
        "config": {
            "experiment": EXPERIMENT,
            "use_structure_preprocess": USE_STRUCTURE_PREPROCESS,
            "use_prototype": USE_PROTOTYPE,
            "use_cosine_huber": USE_COSINE_HUBER,
            "use_conformal": USE_CONFORMAL,
            "lambda_cos": LAMBDA_COS,
            "prototype_gamma": PROTOTYPE_GAMMA,
            "tau": TAU,
            "conformal_alpha": CONFORMAL_ALPHA,
        },
    }
    result_path = os.path.join(WORKING_DIR, f"result_{EXPERIMENT}_{SOURCE}_{TARGET}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result JSON saved to {result_path}")

    # ----------------- Output packaging helper -----------------
    # Zips embeddings + checkpoints + results into a single file
    # so you only download one artifact from Kaggle after the run.
    zip_name = os.path.join(WORKING_DIR, f"{EXPERIMENT}_{SOURCE}_{TARGET}_outputs.zip")
    print(f"\nPackaging outputs into {zip_name} ...")
    subprocess.run(
        ["zip", "-rq", zip_name,
         os.path.relpath(EMBED_CACHE_DIR, WORKING_DIR),
         os.path.relpath(CHECKPOINT_DIR, WORKING_DIR),
         os.path.relpath(result_path, WORKING_DIR)],
        cwd=WORKING_DIR,
        check=False,
    )
    print(f"Packaging complete. You can download this single file from the Kaggle Output tab.")
    print("\n>>> DONE. Download /kaggle/working/ contents before the VM shuts down! <<<")


if __name__ == "__main__":
    main()
