
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "training data"

SOURCE = DATA_DIR / "tiny_stories.txt"
OUTPUT = DATA_DIR / "tiny_stories_sample.txt"

MARKER = "<|endoftext|>"
CHUNK_SIZE = 1024 * 1024
MAX_SAMPLE_BYTES = 50 * 1024 * 1024
MAX_STORIES = 50_000


def main():
    if not SOURCE.is_file():
        raise SystemExit(f"ERROR: TinyStories file not found:\n{SOURCE}")

    print(f"Source: {SOURCE}")
    print(f"Source size: {SOURCE.stat().st_size:,} bytes")
    print("Checking the beginning of the file...")

    with SOURCE.open("r", encoding="utf-8", errors="replace") as f:
        beginning = f.read(8 * 1024 * 1024)

    if MARKER not in beginning:
        print("\nThe expected story separator was not found in the first 8 MB.")
        print("No sample was created. Do not train yet.")
        print("The file may use a different format; we need to inspect that format.")
        print("\nFirst 1,000 characters:")
        print(repr(beginning[:1000]))
        return

    stories_saved = 0
    sample_bytes = 0
    first_story = None
    buffer = ""
    finished = False

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with SOURCE.open("r", encoding="utf-8", errors="replace") as source, \
         OUTPUT.open("w", encoding="utf-8", newline="\n") as output:

        while not finished:
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break

            parts = (buffer + chunk).split(MARKER)
            buffer = parts.pop()

            for story in parts:
                story = story.strip()
                if not story:
                    continue

                encoded_size = len((story + "\n\n").encode("utf-8"))

                if sample_bytes + encoded_size > MAX_SAMPLE_BYTES:
                    finished = True
                    break

                if first_story is None:
                    first_story = story[:500]

                output.write(story + "\n\n")
                stories_saved += 1
                sample_bytes += encoded_size

                if (
                    stories_saved >= MAX_STORIES
                    or sample_bytes >= MAX_SAMPLE_BYTES
                ):
                    finished = True
                    break

        # Include the final unfinished story only if there is room.
        if not finished and buffer.strip():
            story = buffer.strip()
            encoded_size = len((story + "\n\n").encode("utf-8"))

            if (
                stories_saved < MAX_STORIES
                and sample_bytes + encoded_size <= MAX_SAMPLE_BYTES
            ):
                if first_story is None:
                    first_story = story[:500]
                output.write(story + "\n\n")
                stories_saved += 1
                sample_bytes += encoded_size

    if stories_saved == 0:
        OUTPUT.unlink(missing_ok=True)
        raise SystemExit(
            "ERROR: No complete stories were extracted. Do not train yet."
        )

    print("\nSUCCESS: TinyStories was read by Python.")
    print(f"Stories copied: {stories_saved:,}")
    print(f"Sample size: {sample_bytes:,} bytes")
    print(f"Sample saved to: {OUTPUT}")
    print("\nFirst story preview:")
    print(first_story)


if __name__ == "__main__":
    main()
