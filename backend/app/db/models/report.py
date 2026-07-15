from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.db.session import Base


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    inference_result_id = Column(
        Integer,
        ForeignKey("inference_results.id"),
        nullable=False,
        index=True,
    )
    report_type = Column(String(64), nullable=False, default="pdf")
    report_path = Column(String(512), nullable=False)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())