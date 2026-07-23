import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.report_generator import generate_report


def main():
    predicted_class = "PNEUMONIA"
    confidence = 0.93

    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.2%}")
    print("=" * 60)

    report = generate_report(predicted_class, confidence)
    print(report)


if __name__ == "__main__":
    main()
