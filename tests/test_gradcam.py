"""Quick test: run Grad-CAM on one sample chest X-ray and save the overlay."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.xai.gradcam import run_gradcam

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "gradcam_output.png"


def find_sample_image() -> Path:
    if not SAMPLE_DIR.exists():
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    for ext in ("*.png", "*.jpg", "*.jpeg"):
        matches = list(SAMPLE_DIR.glob(ext))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No image found in {SAMPLE_DIR}. "
        "Place one chest X-ray PNG/JPG there and re-run."
    )


def main():
    image_path = find_sample_image()
    print(f"Input: {image_path}")

    pred_class, confidence, overlay = run_gradcam(str(image_path))

    print(f"Prediction : {pred_class}")
    print(f"Confidence : {confidence:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(OUTPUT_PATH)
    print(f"Saved to   : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
