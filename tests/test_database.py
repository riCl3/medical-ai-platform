import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.database import get_all_predictions, get_prediction_by_id, init_db, save_prediction


def main():
    init_db()
    print("Database initialized.\n")

    # Save a sample prediction
    pred_id = save_prediction(
        image_filename="sample.jpeg",
        predicted_class="PNEUMONIA",
        confidence=0.95,
        gradcam_image_path="outputs/gradcam_sample.png",
        llm_report="Findings: Opacities present. Impression: Pneumonia likely.",
    )
    print(f"Saved prediction with id={pred_id}")

    # Save another
    pred_id2 = save_prediction(
        image_filename="normal.jpeg",
        predicted_class="NORMAL",
        confidence=0.88,
    )
    print(f"Saved prediction with id={pred_id2}\n")

    # Get by id
    p = get_prediction_by_id(pred_id)
    print(f"get_prediction_by_id({pred_id}):")
    print(f"  class={p.predicted_class}, confidence={p.confidence}, timestamp={p.timestamp}\n")

    # Get all
    all_preds = get_all_predictions()
    print(f"Total predictions: {len(all_preds)}")
    for p in all_preds:
        print(f"  [{p.id}] {p.predicted_class} ({p.confidence:.2%}) - {p.image_filename}")


if __name__ == "__main__":
    main()
