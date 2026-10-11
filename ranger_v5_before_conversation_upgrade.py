"""
Ranger v5 — locally trained language model for Techno
====================================================

This file trains a small Transformer language model from scratch.
It does NOT download or use pretrained model weights.

Required:
    python -m pip install torch tokenizers

Expected files beside this script:
    training.txt                       general/literature text
    ranger_conversation_training.txt   dialogue examples
    ranger_conversation_extra.txt      optional extra dialogue examples
    ranger_eval.txt                    optional human-evaluation prompts (NOT training data)

Outputs:
    ranger_tokenizer.json
    ranger_v5_latest.pt
    ranger_v5_best.pt
    ranger_v5_training_log.csv

Run:
    python ranger_v5.py

Important:
- Keep ranger_v4.py and all v4 files unchanged.
- First run trains a new tokenizer and model.
- Later runs resume the latest checkpoint by default.
- To deliberately start a fresh v5 experiment, change FRESH_START to True.
  This creates a new run and renames existing v5 outputs with a backup suffix.
"""

from __future__ import annotations

import csv
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Iterable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit(
        "\nRanger v5 needs PyTorch.\n"
        "In this folder, run:\n"
        "    python -m pip install torch\n"
    )

try:
    from tokenizers import Tokenizer, AddedToken
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
except ImportError:
    raise SystemExit(
        "\nRanger v5 needs the local tokenizer library.\n"
        "It trains a tokenizer from your own text; it is NOT a pretrained LLM.\n"
        "Run:\n"
        "    python -m pip install tokenizers\n"
        "Then run ranger_v5.py again.\n"
    )


# ============================================================
# 1. SETTINGS — safe starting values for a modest local machine
# ============================================================

ROOT = Path(__file__).resolve().parent

GENERAL_FILE = ROOT / "training.txt"
CONVERSATION_FILES = [
    ROOT / "ranger_conversation_training.txt",
    ROOT / "ranger_conversation_extra.txt",
]
EVAL_FILE = ROOT / "ranger_eval.txt"  # Reference prompts only; never trained on.

TOKENIZER_FILE = ROOT / "ranger_tokenizer.json"
LATEST_CHECKPOINT = ROOT / "ranger_v5_latest.pt"
BEST_CHECKPOINT = ROOT / "ranger_v5_best.pt"
LOG_FILE = ROOT / "ranger_v5_training_log.csv"

# Model
VOCAB_TARGET = 2048
BLOCK_SIZE = 192
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.10

# Training
BATCH_SIZE = 12
TRAINING_STEPS = 6500
EVAL_INTERVAL = 250
EVAL_BATCHES = 12
SAVE_INTERVAL = 500
LEARNING_RATE = 3e-5
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.10
GRAD_CLIP = 1.0
CONVERSATION_BATCH_PROBABILITY = 0.50
VALIDATION_FRACTION = 0.10
RANDOM_SEED = 1337

# Generation
TEMPERATURE = 0.35
TOP_K = 20
TOP_P = 0.85
REPETITION_PENALTY = 1.05
MAX_NEW_TOKENS = 180
MAX_HISTORY_TURNS = 4

# True only if you intentionally want to discard the current v5 run and
# start a fresh experiment. v4 is never touched.
FRESH_START = False

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|doc|>",
    "<|user|>",
    "<|ranger|>",
    "<|end|>",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 2. FILE AND CORPUS PREPARATION
# ============================================================

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def split_documents(text: str, max_chars: int = 2500) -> list[str]:
    """Split a long text into independent, reasonably sized documents."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_parts = re.split(r"\n\s*\n+", text)
    documents: list[str] = []

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        if len(part) <= max_chars:
            documents.append(part)
            continue

        # For very long paragraphs, prefer splitting at sentence/line boundaries.
        pieces = re.split(r"(?<=[.!?])\s+|\n+", part)
        current = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if current and len(current) + len(piece) + 1 > max_chars:
                documents.append(current)
                current = piece
            elif not current:
                current = piece
            else:
                current += " " + piece

            # Avoid pathological single sentences that are extremely long.
            while len(current) > max_chars * 2:
                documents.append(current[:max_chars])
                current = current[max_chars:]

        if current.strip():
            documents.append(current.strip())

    return documents


LABEL_RE = re.compile(
    r"^\s*(Human|User|You|Ranger|Assistant)\s*:\s*(.*)$",
    flags=re.IGNORECASE,
)


def parse_dialogue(text: str) -> list[tuple[str, str]]:
    """
    Read common Human:/User:/You: and Ranger:/Assistant: transcripts.
    Multi-line turns are supported. Unlabelled lines continue the prior turn.
    """
    turns: list[tuple[str, str]] = []
    current_role: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_role, current_lines
        if current_role and current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                turns.append((current_role, body))
        current_role = None
        current_lines = []

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = LABEL_RE.match(line)
        if match:
            flush()
            label = match.group(1).lower()
            current_role = "user" if label in {"human", "user", "you"} else "ranger"
            current_lines = [match.group(2).strip()] if match.group(2).strip() else []
        elif current_role is not None:
            current_lines.append(line)

    flush()

    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for role, body in turns:
        if role == "user":
            pending_user = body
        elif role == "ranger" and pending_user:
            pairs.append((pending_user, body))
            pending_user = None
    return pairs


def remove_labeled_dialogue(text: str) -> str:
    """
    Remove explicitly labelled Human/User/You/Ranger/Assistant turns
    from the general corpus after extracting them as dialogue examples.
    Assumes these labelled examples are separated by blank lines.
    """
    kept_lines = []
    skipping_turn = False

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if LABEL_RE.match(line):
            skipping_turn = True
            continue

        if skipping_turn:
            if not line.strip():
                skipping_turn = False
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines)



def format_dialogue_pair(user_text: str, ranger_text: str) -> str:
    return (
        f"<|user|>\n{user_text.strip()}\n"
        f"<|ranger|>\n{ranger_text.strip()}\n<|end|>"
    )



def parse_embedded_dialogue(text: str) -> list[tuple[str, str]]:
    """
    Extract single-line Human/Ranger pairs embedded in a large text file.
    Ignore all unlabelled lines so book text cannot become a dialogue turn.
    """
    pairs = []
    pending_user = None

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = LABEL_RE.match(line)
        if not match:
            continue

        label = match.group(1).lower()
        body = match.group(2).strip()
        if not body:
            continue

        if label in {"human", "user", "you"}:
            pending_user = body
        elif pending_user is not None:
            pairs.append((pending_user, body))
            pending_user = None

    return pairs



def load_corpora() -> tuple[list[str], list[tuple[str, str]]]:
    if not GENERAL_FILE.exists():
        raise SystemExit(
            f"\nCould not find {GENERAL_FILE.name} in:\n{ROOT}\n"
            "Keep ranger_v5.py beside training.txt."
        )

    # Extract dialogue embedded in training.txt before splitting the
    # remaining literature/general text into documents.
    general_text = read_text(GENERAL_FILE)
    embedded_pairs = parse_embedded_dialogue(general_text)
    general_text = remove_labeled_dialogue(general_text)
    general_docs = split_documents(general_text)

    dialogue_pairs = list(embedded_pairs)
    print(
        f"Dialogue examples extracted from {GENERAL_FILE.name}: "
        f"{len(embedded_pairs):,}"
    )

    found_files = []
    for path in CONVERSATION_FILES:
        text = read_text(path)
        if not text.strip():
            print(f"Conversation file missing or empty: {path.name}")
            continue

        found_files.append(path.name)
        pairs = parse_dialogue(text)
        dialogue_pairs.extend(pairs)
        print(f"Dialogue pairs parsed from {path.name}: {len(pairs):,}")

    raw_count = len(dialogue_pairs)

    # Remove exact duplicate pairs while preserving order.
    dialogue_pairs = list(dict.fromkeys(
        (u.strip(), r.strip())
        for u, r in dialogue_pairs
        if u.strip() and r.strip()
    ))

    if len(dialogue_pairs) < 10:
        raise SystemExit(
            "\nToo few usable dialogue pairs were found. "
            "Check the Human:/Ranger: labels in your files."
        )

    print(f"General/literature documents: {len(general_docs):,}")
    print(f"Conversation files loaded: {', '.join(found_files) or 'none'}")
    print(f"Dialogue pairs before deduplication: {raw_count:,}")
    print(f"Unique usable dialogue pairs: {len(dialogue_pairs):,}")

    if EVAL_FILE.exists():
        print(f"Evaluation prompts found: {EVAL_FILE.name} (not used for training)")

    return general_docs, dialogue_pairs


def split_train_validation(items: list, fraction: float):
    if len(items) < 10:
        raise SystemExit(
            "Not enough documents/examples to make a useful train/validation split."
        )

    # Shuffle a copy using a fixed seed. This is reproducible without
    # changing the random sequence used later for model training.
    shuffled = list(items)
    local_rng = random.Random(RANDOM_SEED + len(items))
    local_rng.shuffle(shuffled)

    validation_count = max(1, round(len(shuffled) * fraction))
    validation_count = min(validation_count, len(shuffled) - 1)

    validation_items = shuffled[:validation_count]
    training_items = shuffled[validation_count:]

    return training_items, validation_items



def make_training_texts(
    general_docs: list[str],
    dialogue_pairs: list[tuple[str, str]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    # Stable split: validation documents/examples are held out from model training.
    g_train, g_val = split_train_validation(general_docs, VALIDATION_FRACTION)
    d_train, d_val = split_train_validation(dialogue_pairs, VALIDATION_FRACTION)

    general_train_texts = [f"<|doc|>\n{doc}\n<|end|>" for doc in g_train]
    general_val_texts = [f"<|doc|>\n{doc}\n<|end|>" for doc in g_val]
    dialogue_train_texts = [format_dialogue_pair(u, r) for u, r in d_train]
    dialogue_val_texts = [format_dialogue_pair(u, r) for u, r in d_val]

    return (
        general_train_texts,
        general_val_texts,
        dialogue_train_texts,
        dialogue_val_texts,
    )


# ============================================================
# 3. TRAIN A LOCAL BPE TOKENIZER — NO PRETRAINED MODEL
# ============================================================

def train_or_load_tokenizer(training_texts: list[str]) -> Tokenizer:
    if TOKENIZER_FILE.exists() and not FRESH_START:
        try:
            tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
            print(f"Loaded existing local tokenizer: {TOKENIZER_FILE.name}")
            return tokenizer
        except Exception as exc:
            print(f"Existing tokenizer could not be loaded ({exc}); training a new one.")

    print("\nTraining a Byte-Pair Encoding tokenizer from local training text...")
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_TARGET,
        min_frequency=2,
        special_tokens=[
            AddedToken(token, special=True, normalized=False)
            for token in SPECIAL_TOKENS
        ],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter(training_texts), trainer=trainer)
    tokenizer.save(str(TOKENIZER_FILE))
    print(f"Tokenizer saved: {TOKENIZER_FILE.name}")
    print(f"Tokenizer vocabulary: {tokenizer.get_vocab_size():,} tokens")
    return tokenizer


def encode_texts(tokenizer: Tokenizer, texts: list[str]) -> torch.Tensor:
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text).ids)
    return torch.tensor(ids, dtype=torch.long)


# ============================================================
# 4. THE RANGER TRANSFORMER
# ============================================================

class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        batch, time_steps, channels = x.shape
        key = self.key(x)
        query = self.query(x)
        weights = query @ key.transpose(-2, -1) * (key.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:time_steps, :time_steps] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        value = self.value(x)
        return weights @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        combined = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(combined))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class Ranger(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.final_norm = nn.LayerNorm(N_EMBD)
        self.language_head = nn.Linear(N_EMBD, vocab_size, bias=False)

        # Weight tying shares token embeddings and output weights.
        self.language_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, time_steps = idx.shape
        token_emb = self.token_embedding(idx)
        positions = torch.arange(time_steps, device=idx.device)
        position_emb = self.position_embedding(positions)
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.language_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * time_steps, -1),
                targets.reshape(batch * time_steps),
            )
        return logits, loss

    @torch.no_grad()
    def generate_one(self, idx, max_new_tokens: int, end_id: int,
                     user_id: int, doc_id: int, temperature: float,
                     top_k: int, top_p: float, repetition_penalty: float):
        self.eval()
        generated = []
        for _ in range(max_new_tokens):
            context = idx[:, -BLOCK_SIZE:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :].clone()

            # Penalize tokens recently used in the prompt/output to reduce loops.
            if repetition_penalty > 1.0:
                recent = set(idx[0, -min(64, idx.shape[1]):].tolist())
                for token_id in recent:
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty

            next_logits = next_logits / max(temperature, 1e-4)

            if top_k > 0 and top_k < next_logits.shape[-1]:
                cutoff = torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[next_logits < cutoff] = float("-inf")

            probabilities = F.softmax(next_logits, dim=-1)

            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                remove = cumulative - sorted_probs > top_p
                sorted_probs[remove] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                probabilities = torch.zeros_like(probabilities).scatter(
                    -1, sorted_indices, sorted_probs
                )

            next_token = torch.multinomial(probabilities, num_samples=1)
            token_id = int(next_token.item())

            # Stop if the model starts a new user/document turn unexpectedly.
            if token_id in {end_id, user_id, doc_id}:
                break

            generated.append(token_id)
            idx = torch.cat((idx, next_token), dim=1)

        self.train()
        return generated


# ============================================================
# 5. TRAINING, VALIDATION, CHECKPOINTS, AND LOGGING
# ============================================================


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    generator=None,
):
    if data.numel() <= BLOCK_SIZE + 1:
        raise ValueError(
            f"Encoded dataset has only {data.numel()} tokens; "
            f"need more than {BLOCK_SIZE + 1}."
        )

    # The upper bound is exclusive. This includes every valid starting
    # position while ensuring the final target stays inside the data.
    starts = torch.randint(
        0,
        data.numel() - BLOCK_SIZE,
        (batch_size,),
        generator=generator,
    )

    x = torch.stack([data[s:s + BLOCK_SIZE] for s in starts])
    y = torch.stack([data[s + 1:s + BLOCK_SIZE + 1] for s in starts])

    return x.to(DEVICE), y.to(DEVICE)




@torch.no_grad()
def estimate_loss(
    model: Ranger,
    general_data: torch.Tensor,
    dialogue_data: torch.Tensor,
    batches: int = EVAL_BATCHES,
):
    model.eval()
    results = {}

    for label, data in (
        ("general", general_data),
        ("dialogue", dialogue_data),
    ):
        losses = []

        # Reset the seed on every evaluation. The same validation windows
        # are therefore sampled at each checkpoint.
        seed_offset = 0 if label == "general" else 100_000
        generator = torch.Generator(device="cpu")
        generator.manual_seed(RANDOM_SEED + seed_offset)

        for _ in range(batches):
            x, y = get_batch(
                data,
                min(BATCH_SIZE, 8),
                generator=generator,
            )
            _, loss = model(x, y)
            losses.append(float(loss.item()))

        results[label] = sum(losses) / len(losses)

    model.train()

    # Keep dialogue quality as the primary goal, while retaining a
    # meaningful signal for general-language performance.
    results["combined"] = (
        0.25 * results["general"]
        + 0.75 * results["dialogue"]
    )

    return results



def learning_rate_at(step: int) -> float:
    if TRAINING_STEPS <= 1:
        return LEARNING_RATE
    progress = min(max(step / TRAINING_STEPS, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LEARNING_RATE + cosine * (LEARNING_RATE - MIN_LEARNING_RATE)


def save_checkpoint(path: Path, model: Ranger, optimizer, step: int,
                    best_validation: float, vocab_size: int, metrics: dict):
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "best_validation": best_validation,
        "vocab_size": vocab_size,
        "metrics": metrics,
        "config": {
            "block_size": BLOCK_SIZE,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "dropout": DROPOUT,
            "tokenizer_file": TOKENIZER_FILE.name,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def backup_v5_files():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for path in (TOKENIZER_FILE, LATEST_CHECKPOINT, BEST_CHECKPOINT, LOG_FILE):
        if path.exists():
            backup = path.with_name(f"{path.stem}_backup_{stamp}{path.suffix}")
            shutil.move(str(path), str(backup))
            print(f"Backed up {path.name} -> {backup.name}")


def load_checkpoint(path: Path, model: Ranger, optimizer=None):
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def train_model(tokenizer: Tokenizer,
                general_train_texts: list[str],
                general_val_texts: list[str],
                dialogue_train_texts: list[str],
                dialogue_val_texts: list[str]) -> Ranger:
    print("\nEncoding training and validation data...")
    general_train = encode_texts(tokenizer, general_train_texts)
    general_val = encode_texts(tokenizer, general_val_texts)
    dialogue_train = encode_texts(tokenizer, dialogue_train_texts)
    dialogue_val = encode_texts(tokenizer, dialogue_val_texts)

    for label, data in (
        ("General training", general_train),
        ("General validation", general_val),
        ("Dialogue training", dialogue_train),
        ("Dialogue validation", dialogue_val),
    ):
        print(f"{label} tokens: {data.numel():,}")

    vocab_size = tokenizer.get_vocab_size()
    model = Ranger(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = None  # Deliberately no mixed precision: stable CPU/GPU compatibility.
    start_step = 0
    best_validation = float("inf")
    last_metrics = {"general": float("inf"), "dialogue": float("inf"), "combined": float("inf")}

    if LATEST_CHECKPOINT.exists() and not FRESH_START:
        try:
            checkpoint = load_checkpoint(LATEST_CHECKPOINT, model, optimizer)
            # Never resume a checkpoint built with a different architecture/vocabulary.
            expected = checkpoint.get("config", {})
            if (
                checkpoint.get("vocab_size") != vocab_size
                or expected.get("block_size") != BLOCK_SIZE
                or expected.get("n_embd") != N_EMBD
                or expected.get("n_head") != N_HEAD
                or expected.get("n_layer") != N_LAYER
            ):
                raise ValueError("checkpoint architecture or vocabulary does not match")
            start_step = int(checkpoint.get("step", 0))
            best_validation = float(checkpoint.get("best_validation", float("inf")))
            last_metrics = checkpoint.get("metrics", last_metrics)
            print(f"Resuming v5 checkpoint at step {start_step:,}.")
        except Exception as exc:
            print(f"Could not resume old v5 checkpoint ({exc}). Starting a fresh model.")
            model = Ranger(vocab_size).to(DEVICE)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            start_step = 0
            best_validation = float("inf")

    parameter_count = sum(p.numel() for p in model.parameters())
    print("\nRanger v5 created.")
    print(f"Device: {DEVICE}")
    print(f"Vocabulary: {vocab_size:,}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Block size: {BLOCK_SIZE} tokens")
    print(f"Training target: step {TRAINING_STEPS:,}")
    print(f"Dialogue batch probability: {CONVERSATION_BATCH_PROBABILITY:.0%}")

    if not LOG_FILE.exists() or FRESH_START:
        with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "general_validation",
                             "dialogue_validation", "combined_validation", "learning_rate"])

    started = time.time()
    for step in range(start_step, TRAINING_STEPS):
        lr = learning_rate_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        use_dialogue = random.random() < CONVERSATION_BATCH_PROBABILITY
        batch_data = dialogue_train if use_dialogue else general_train
        x, y = get_batch(batch_data, BATCH_SIZE)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        current_step = step + 1
        if current_step == 1 or current_step % 100 == 0:
            print(f"Step {current_step:>5,} | training loss: {loss.item():.4f} | lr: {lr:.6g}")

        if current_step % EVAL_INTERVAL == 0 or current_step == TRAINING_STEPS:
            val = estimate_loss(model, general_val, dialogue_val)
            last_metrics = val
            elapsed = (time.time() - started) / 60
            print(
                f"  VALIDATION step {current_step:,} | "
                f"general {val['general']:.4f} | "
                f"dialogue {val['dialogue']:.4f} | "
                f"combined {val['combined']:.4f} | "
                f"{elapsed:.1f} min elapsed"
            )

            with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    current_step, f"{loss.item():.6f}",
                    f"{val['general']:.6f}", f"{val['dialogue']:.6f}",
                    f"{val['combined']:.6f}", f"{lr:.8g}",
                ])

            if val["combined"] < best_validation:
                best_validation = val["combined"]
                save_checkpoint(
                    BEST_CHECKPOINT, model, optimizer, current_step,
                    best_validation, vocab_size, val
                )
                print(f"  New best checkpoint saved: {BEST_CHECKPOINT.name}")

        if current_step % SAVE_INTERVAL == 0 or current_step == TRAINING_STEPS:
            save_checkpoint(
                LATEST_CHECKPOINT, model, optimizer, current_step,
                best_validation, vocab_size, last_metrics
            )
            print(f"  Recovery checkpoint saved: {LATEST_CHECKPOINT.name}")

    if BEST_CHECKPOINT.exists():
        try:
            best = load_checkpoint(BEST_CHECKPOINT, model, optimizer=None)
            print(
                f"\nLoaded best v5 weights from step {best.get('step', '?')} "
                f"(combined validation {best.get('best_validation', float('nan')):.4f})."
            )
        except Exception as exc:
            print(f"Could not load best checkpoint for chat: {exc}")

    print(f"Training finished. Results log: {LOG_FILE.name}")
    return model


# ============================================================
# 6. CHAT INTERFACE
# ============================================================

def special_id(tokenizer: Tokenizer, token: str) -> int:
    value = tokenizer.token_to_id(token)
    if value is None:
        raise RuntimeError(f"Tokenizer is missing required token: {token}")
    return value



BASELINE_TESTS = [
    (
        "1. Greeting and identity",
        "Hello. What is your name?",
    ),
    (
        "2. Basic explanation",
        "What is a language model? Explain it in two sentences.",
    ),
    (
        "3. Clear communication",
        "Explain how a language model learns in simple, everyday language.",
    ),
    (
        "4. Sailing knowledge",
        "What are two things a sailor should observe when assessing sailing conditions?",
    ),
    (
        "5. Honesty and uncertainty",
        "If you do not know the answer to a question, what should you do?",
    ),
]




def chat(model: Ranger, tokenizer: Tokenizer):
    end_id = special_id(tokenizer, "<|end|>")
    user_id = special_id(tokenizer, "<|user|>")
    ranger_id = special_id(tokenizer, "<|ranger|>")
    doc_id = special_id(tokenizer, "<|doc|>")

    history: list[tuple[str, str]] = []
    print("\n" + "=" * 60)
    print("Ranger v5 is ready.")
    print("Type /exit to quit, /reset to clear chat history, or /help for commands.")
    print("=" * 60)

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nRanger session ended.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            print("Ranger: Until next time.")
            break
        if user_text.lower() == "/reset":
            history.clear()
            print("Ranger: Conversation history cleared.")
            continue
        
        if user_text.lower() == "/help":
            print("Commands: /help, /test, /reset, /exit")
            continue

        if user_text.lower() == "/test":
            print("\n" + "=" * 60)
            print("RANGER BASELINE TEST")
            print("Use these same five prompts after each model update.")
            print("=" * 60)

            for index, (label, test_prompt) in enumerate(BASELINE_TESTS, start=1):
                prompt = (
                    f"<|user|>\n{test_prompt}\n"
                    f"<|ranger|>\n"
                )
                prompt_ids = tokenizer.encode(prompt).ids
                prompt_ids = prompt_ids[-(BLOCK_SIZE - 4):]

                context = torch.tensor(
                    [prompt_ids],
                    dtype=torch.long,
                    device=DEVICE,
                )

                # Make each test repeatable so model versions can be compared.
                torch.manual_seed(RANDOM_SEED + index)

                output_ids = model.generate_one(
                    context,
                    min(MAX_NEW_TOKENS, 100),
                    end_id,
                    user_id,
                    doc_id,
                    TEMPERATURE,
                    TOP_K,
                    TOP_P,
                    REPETITION_PENALTY,
                )

                answer = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                ).strip()

                if not answer:
                    answer = "[No response generated]"

                print(f"\n{label}")
                print(f"You: {test_prompt}")
                print(f"Ranger: {answer}")

            print("\nBaseline test complete.")
            continue


        messages = history[-MAX_HISTORY_TURNS:]
        prompt = ""
        for old_user, old_ranger in messages:
            prompt += f"<|user|>\n{old_user}\n<|ranger|>\n{old_ranger}\n<|end|>\n"
        prompt += f"<|user|>\n{user_text}\n<|ranger|>\n"

        prompt_ids = tokenizer.encode(prompt).ids
        # Keep the latest context within the model's context window.
        prompt_ids = prompt_ids[-(BLOCK_SIZE - 4):]
        context = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

        output_ids = model.generate_one(
            context, MAX_NEW_TOKENS, end_id, user_id, doc_id,
            TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY
        )
        answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

        # Guard against empty outputs without inventing a replacement answer.
        if not answer:
            answer = "I couldn't form a response to that yet."

        print(f"\nRanger: {answer}")
        history.append((user_text, answer))
        if len(history) > MAX_HISTORY_TURNS:
            history = history[-MAX_HISTORY_TURNS:]


# ============================================================
# 7. START
# ============================================================

def main():
    print("=" * 60)
    print("RANGER v5 — local language model for Techno")
    print("=" * 60)
    print(f"Folder: {ROOT}")
    print(f"Device selected: {DEVICE}")
    print("No pretrained language-model weights will be downloaded or used.")

    if FRESH_START:
        backup_v5_files()

    general_docs, dialogue_pairs = load_corpora()
    (
        general_train_texts,
        general_val_texts,
        dialogue_train_texts,
        dialogue_val_texts,
    ) = make_training_texts(general_docs, dialogue_pairs)

    tokenizer_training_texts = general_train_texts + dialogue_train_texts
    tokenizer = train_or_load_tokenizer(tokenizer_training_texts)

    model = train_model(
        tokenizer,
        general_train_texts,
        general_val_texts,
        dialogue_train_texts,
        dialogue_val_texts,
    )
    chat(model, tokenizer)


if __name__ == "__main__":
    main()
