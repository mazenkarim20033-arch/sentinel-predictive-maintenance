"""Downloads and verifies the public datasets used by SENTINEL (NASA PCoE). Files are not redistributed in this repository.

    python scripts/fetch_data.py                 # NASA C-MAPSS (about 12 MB)
    python scripts/fetch_data.py --battery       # also the NASA battery archive (about 210 MB)
    python scripts/fetch_data.py --zip PATH      # use an already downloaded CMAPSSData.zip instead of downloading

Each archive is checked against the size and SHA-256 recorded when the project was built; a mismatch stops the script."""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SOURCES = {
    "cmapss": {
        "url": "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
        "size": 12_425_978,
        "sha256": "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f",
        "zip": RAW / "CMAPSSData.zip",
        "extract": RAW / "extracted",
    },
    "battery": {
        "url": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
        "size": 209_708_670,
        "sha256": "82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e",
        "zip": RAW / "battery" / "zip" / "battery.zip",
        "extract": RAW / "battery" / "extracted",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:6.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def verify(name: str, path: Path) -> None:
    spec = SOURCES[name]
    if path.stat().st_size != spec["size"]:
        raise SystemExit(f"{path}: size {path.stat().st_size} does not match the expected {spec['size']}")
    if sha256(path) != spec["sha256"]:
        raise SystemExit(f"{path}: SHA-256 does not match the recorded value; the file may be corrupt or changed upstream")


def extract(name: str, path: Path) -> None:
    out = SOURCES[name]["extract"]
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(out)
    if name == "battery":  # the archive holds one zip per cell group
        base = out / "5. Battery Data Set"
        for inner in sorted(base.glob("*.zip")):
            target = base / inner.stem
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(inner) as z:
                z.extractall(target)


def fetch(name: str, local_zip: Path | None = None) -> None:
    spec = SOURCES[name]
    path = local_zip or spec["zip"]
    if not path.exists():
        print(f"{name}: downloading {spec['url']}")
        download(spec["url"], path, spec["size"])
    print(f"{name}: verifying {path}")
    verify(name, path)
    print(f"{name}: extracting to {spec['extract']}")
    extract(name, path)
    print(f"{name}: ready")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--battery", action="store_true", help="also fetch the NASA battery archive")
    p.add_argument("--zip", type=Path, help="use this local CMAPSSData.zip instead of downloading")
    args = p.parse_args()
    fetch("cmapss", args.zip)
    if args.battery:
        fetch("battery")


if __name__ == "__main__":
    sys.exit(main())
