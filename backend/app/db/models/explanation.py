from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from app.db.session import Base


class Explanation(Base):
    __tablename__ = "explanations"

    id = Column(Integer, primary_key=True, index=True)
    inference_result_id = Column(
        Integer,
        ForeignKey("inference_results.id"),
        nullable=False,
        index=True,
    )

    # Stores full explanation payload as JSON string:
    # {
    #   "top_channels": [...],
    #   "temporal_attention": [...],
    #   "gat_edges": [...],
    #   "seizure_ranges": [...],
    #   "probability_timeline": [...],
    #   "summary_text": "..."
    # }
    top_channels_json = Column(Text, nullable=True)

    saliency_path = Column(String(512), nullable=True)
    temporal_attention_path = Column(String(512), nullable=True)
    gat_attention_path = Column(String(512), nullable=True)

    summary_text = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())