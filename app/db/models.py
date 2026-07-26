from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    image_filename = Column(String(512), nullable=False)
    predicted_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    gradcam_image_path = Column(String(1024))
    gradcam_blob = Column(Text)
    llm_report = Column(Text)

    def __repr__(self):
        return f"<Prediction(id={self.id}, predicted_class={self.predicted_class})>"
