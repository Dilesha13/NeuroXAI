from pydantic import BaseModel
from typing import List, Optional

class TimelineItem(BaseModel):
    window_index: int
    start_sec: float
    end_sec: float
    probability: float
    predicted_label: int

class ModelInfo(BaseModel):
    name: str
    threshold: float
    checkpoint: str

class InferenceSummary(BaseModel):
    num_windows: int
    num_seizure_windows: int
    max_probability: float
    mean_probability: float
    overall_decision: str
    overall_summary: str

class ExplainabilityPayload(BaseModel):
    top_channels: List[str]
    saliency_path: Optional[str] = None
    temporal_attention_path: Optional[str] = None
    gat_attention_path: Optional[str] = None
    summary_text: str

class InferenceResponse(BaseModel):
    inference_id: Optional[int] = None
    record_id: Optional[int] = None
    patient_id: Optional[int] = None
    model: ModelInfo
    summary: InferenceSummary
    timeline: List[TimelineItem]
    explainability: ExplainabilityPayload
