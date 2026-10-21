"""Verify an existing local checkpoint without downloading or loading it."""
import argparse
import hashlib
from pathlib import Path

EXPECTED_SIZE = 1050862408
EXPECTED_SHA256 = "995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    path = args.model_dir / "model.safetensors"
    if path.stat().st_size != EXPECTED_SIZE:
        raise SystemExit("Checkpoint size mismatch")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit("Checkpoint SHA256 mismatch")
    if not (args.model_dir / "config.json").is_file():
        raise SystemExit("Missing config.json")
    print("Checkpoint size and SHA256 verified; config.json exists.")


if __name__ == "__main__":
    main()
