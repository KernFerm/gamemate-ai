"""Download model assets used by the assistant.

This script supports two download paths:
- Use `ultralytics.YOLO` to ensure ultralytics models are available (by name).
- Directly download weights from URL into the local `models/` folder.

Usage:
  python scripts/download_models.py --all
  python scripts/download_models.py --model yolov8m
  python scripts/download_models.py --list
"""

import argparse
import sys
from pathlib import Path
import requests

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

MODELS = {
    # ultralytics model shorthand -> ultralytics will resolve these
    'yolov8n': {'type': 'ultralytics', 'id': 'yolov8n'},
    'yolov8s': {'type': 'ultralytics', 'id': 'yolov8s'},
    'yolov8m': {'type': 'ultralytics', 'id': 'yolov8m'},
    'yolov8l': {'type': 'ultralytics', 'id': 'yolov8l'},
    'yolov8x': {'type': 'ultralytics', 'id': 'yolov8x'},
    'yolov8n-seg': {'type': 'ultralytics', 'id': 'yolov8n-seg'},
    # Example direct URL (replace with real URLs if you host weights)
    # 'custom-model': {'type': 'url', 'url': 'https://example.com/path/to/model.pt'}
}

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'


def ensure_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_from_url(url: str, dest: Path):
    print(f"Downloading URL: {url} -> {dest}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"✓ Saved to {dest}")


def ensure_ultralytics_model(model_id: str):
    if YOLO is None:
        print("Warning: `ultralytics` package not available. Install it to auto-download models.")
        return False

    try:
        print(f"Ensuring ultralytics model available: {model_id}")
        # Construct YOLO object which will auto-download cache weights if needed
        _ = YOLO(model_id)
        print(f"✓ {model_id} is available via ultralytics")
        return True
    except Exception as e:
        print(f"⚠️ Failed to obtain {model_id} via ultralytics: {e}")
        return False


def list_models():
    print("Available model keys:")
    for k in sorted(MODELS.keys()):
        print(f" - {k}")


def main():
    parser = argparse.ArgumentParser(description='Download assistant model assets')
    parser.add_argument('--all', action='store_true', help='Download all configured models')
    parser.add_argument('--model', type=str, help='Download a single model by key')
    parser.add_argument('--list', action='store_true', help='List available model keys')

    args = parser.parse_args()

    if args.list:
        list_models()
        sys.exit(0)

    ensure_models_dir()

    targets = []
    if args.all:
        targets = list(MODELS.keys())
    elif args.model:
        if args.model not in MODELS:
            print(f"Unknown model key: {args.model}")
            list_models()
            sys.exit(1)
        targets = [args.model]
    else:
        print("No action specified. Use --all or --model <key> or --list")
        sys.exit(0)

    for key in targets:
        meta = MODELS[key]
        if meta['type'] == 'ultralytics':
            ok = ensure_ultralytics_model(meta['id'])
            if not ok:
                print(f"Failed to ensure model {key}")
        elif meta['type'] == 'url':
            url = meta['url']
            filename = Path(url).name
            dest = MODELS_DIR / filename
            download_from_url(url, dest)

    print('\n✓ Model download tasks complete')


if __name__ == '__main__':
    main()
