from pathlib import Path
from typing import List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Prediction

DB_PATH = Path(__file__).resolve().parent.parent.parent / "predictions.db"

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def save_prediction(
    image_filename: str,
    predicted_class: str,
    confidence: float,
    gradcam_image_path: str = None,
    llm_report: str = None,
) -> int:
    session = SessionLocal()
    try:
        pred = Prediction(
            image_filename=image_filename,
            predicted_class=predicted_class,
            confidence=confidence,
            gradcam_image_path=gradcam_image_path,
            llm_report=llm_report,
        )
        session.add(pred)
        session.commit()
        return pred.id
    finally:
        session.close()


def get_all_predictions() -> List[Prediction]:
    session = SessionLocal()
    try:
        return session.query(Prediction).order_by(Prediction.timestamp.desc()).all()
    finally:
        session.close()


def get_prediction_by_id(pred_id: int) -> Optional[Prediction]:
    session = SessionLocal()
    try:
        return session.query(Prediction).filter(Prediction.id == pred_id).first()
    finally:
        session.close()
