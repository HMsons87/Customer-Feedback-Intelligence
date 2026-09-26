from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class WordAttribution(BaseModel):
    word: str
    score: float
    direction: str  # "Negative", "Neutral", "Positive"

class ReviewRequest(BaseModel):
    review_text: str = Field(..., example="The color was pretty but size was extremely small and tight.")
    age: Optional[int] = Field(default=35, ge=18, le=100)
    department_name: Optional[str] = Field(default="Dresses")

class ComplaintClusterResponse(BaseModel):
    cluster_id: int
    category_name: str
    top_keywords: List[str]

class FeedbackAnalysisResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    is_complaint: bool
    complaint_cluster: Optional[ComplaintClusterResponse] = None
    key_drivers: List[WordAttribution] = []