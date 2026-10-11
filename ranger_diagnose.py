
from pathlib import Path

import torch
from tokenizers import Tokenizer

import ranger_v5 as r


def main():
    print("=" * 60)
    print("RANGER V5 DIAGNOSTIC — NO TRAINING")
    print("=" * 60)

    if not r.TOKENIZER_FILE.exists():
        print("ERROR: Tokenizer file not found.")
        return

    if not r.BEST_CHECKPOINT.exists():
        print("ERROR: Best checkpoint not found.")
        return

    tokenizer = Tokenizer.from_file(str(r.TOKENIZER_FILE))
    general_docs, pairs = r.load_corpora()

    (
        general_train,
        general_val,
        dialogue_train,
        dialogue_val,
    ) = r.make_training_texts(general_docs, pairs)

    encoded = r.encode_dialogue_examples(
        tokenizer, dialogue_train
    )

    print(f"\nTokenizer vocabulary: {tokenizer.get_vocab_size():,}")
    print(f"Dialogue pairs found: {len(pairs):,}")
    print(f"Encoded training examples: {len(encoded):,}")

    print("\n--- ANSWER-BOUNDARY CHECK ---")

    for i, (ids, response_start) in enumerate(encoded[:5], 1):
        marker = r.special_id(tokenizer, "<|ranger|>")

        try:
            marker_pos = ids.index(marker)
        except ValueError:
            print(f"Example {i}: ERROR — Ranger marker missing")
            continue

        print(f"\nExample {i}")
        print("Tokens around marker:")
        print(repr(tokenizer.decode(
            ids[max(0, marker_pos - 2):marker_pos + 8]
        )))
        print("Expected answer start index:", response_start)
        print("Decoded answer portion:")
        print(repr(tokenizer.decode(ids[response_start:])))

    print("\n--- CHECKPOINT CHECK ---")

    model = r.Ranger(tokenizer.get_vocab_size()).to(r.DEVICE)
    checkpoint = r.load_checkpoint(r.BEST_CHECKPOINT, model)

    if checkpoint.get("vocab_size") != tokenizer.get_vocab_size():
        print("ERROR: Checkpoint and tokenizer vocabulary mismatch.")
        return

    model.eval()
    print("Checkpoint step:", checkpoint.get("step", "unknown"))
    print("Device:", r.DEVICE)

    end_id = r.special_id(tokenizer, "<|end|>")
    user_id = r.special_id(tokenizer, "<|user|>")
    doc_id = r.special_id(tokenizer, "<|doc|>")

    print("\n--- SAMPLE RESPONSES ---")

    for user_text, expected in pairs[:5]:
        prompt = f"<|user|>\n{user_text}\n<|ranger|>\n"
        ids = tokenizer.encode(prompt).ids[-(r.BLOCK_SIZE - 4):]
        context = torch.tensor(
            [ids], dtype=torch.long, device=r.DEVICE
        )

        with torch.no_grad():
            output = model.generate_one(
                context,
                60,
                end_id,
                user_id,
                doc_id,
                0.0 + 0.2,
                1,
                1.0,
                1.0,
            )

        answer = tokenizer.decode(
            output, skip_special_tokens=True
        ).strip()

        print(f"\nUser: {user_text}")
        print(f"Expected: {expected[:180]}")
        print(f"Generated: {answer[:180]}")

    print("\nDiagnostic complete. No training was performed.")


if __name__ == "__main__":
    main()
