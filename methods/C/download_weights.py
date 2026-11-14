"""Download and verify the official MedSAM ViT-B checkpoint from Zenodo."""

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

from medsam_model import MEDSAM_VIT_B_MD5

DEFAULT_URL = (
    "https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1"
)


def md5(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="weights/medsam_vit_b.pth", help="checkpoint destination"
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and md5(output) == MEDSAM_VIT_B_MD5:
        print(f"Checkpoint already verified: {output}")
        return

    temporary = output.with_suffix(output.suffix + ".part")
    print(f"Downloading official checkpoint to {output}")

    def report(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            print(f"\r{done / 1024**2:.1f}/{total / 1024**2:.1f} MiB", end="")

    try:
        urllib.request.urlretrieve(args.url, temporary, reporthook=report)
        print()
        actual = md5(temporary)
        if actual != MEDSAM_VIT_B_MD5:
            raise RuntimeError(
                f"downloaded checkpoint MD5 mismatch: expected {MEDSAM_VIT_B_MD5}, got {actual}"
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Checkpoint verified: {output}")


if __name__ == "__main__":
    main()
