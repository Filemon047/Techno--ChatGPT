
from pathlib import Path
import hashlib
import torch
from tokenizers import Tokenizer

FILES = [
    "ranger_tokenizer.json",
    "ranger_tokenizer_backup_20261010_131346.json",
    "ranger_tokenizer_backup_20261010_170214.json",
    "ranger_tokenizer_before_conversation_upgrade.json",
]

print("=== TOKENIZER COMPARISON ===")

loaded = {}

for name in FILES:
    path = Path(name)
    if not path.exists():
        continue

    raw = path.read_bytes()
    tokenizer = Tokenizer.from_file(str(path))
    vocab = tokenizer.get_vocab()
    loaded[name] = vocab

    print(f"\n{name}")
    print("SHA-256:", hashlib.sha256(raw).hexdigest())
    print("Vocabulary size:", tokenizer.get_vocab_size())

    for token in [
        "<|pad|>", "<|user|>", "<|ranger|>",
        "<|end|>", "<|doc|>"
    ]:
        print(f"  {token}: {tokenizer.token_to_id(token)}")

current_name = "ranger_tokenizer.json"
if current_name in loaded:
    current = loaded[current_name]

    print("\n=== DIFFERENCES FROM CURRENT TOKENIZER ===")
    for name, vocab in loaded.items():
        if name == current_name:
            continue

        changed = sum(
            current.get(token) != vocab.get(token)
            for token in set(current) | set(vocab)
        )
        print(f"{name}: {changed} token-ID differences")

print("\n=== BEST CHECKPOINT METADATA ===")

path = Path("ranger_v5_best.pt")
if path.exists():
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False
    )

    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", list(checkpoint.keys()))

        for key, value in checkpoint.items():
            if not isinstance(value, (dict, list, tuple)):
                print(f"{key}: {str(value)[:250]}")
    else:
        print("Checkpoint type:", type(checkpoint).__name__)
else:
    print("ranger_v5_best.pt not found")

print("\nCheck complete. No files changed.")
