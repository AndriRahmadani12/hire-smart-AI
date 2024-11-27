from pydantic import BaseModel
from typing import List

# Input Schema untuk CVAnalysis
class CVAnalysisInput(BaseModel):
    job_description: str
    tags: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "job_description": "We are looking for a Machine Learning Engineer to develop, deploy, and optimize machine learning models...",
                "tags": ["machine learning", "python", "tensorflow", "pytorch", "cloud computing", "mlops"]
            }
        }

# Response Schemas
class CVAnalysisResponse(BaseModel):
    summary: str
    tag_found: List[str]
    tag_matched: List[str]
    similarity_score: float
    tag_matched_score: float
    total_score: float

    class Config:
        json_schema_extra = {
            "example": {
                "summary": "The candidate has 5 years of experience...",
                "tag_found": ["python", "javascript", "docker"],
                "tag_matched": ["python", "docker"],
                "similarity_score": 0.85,
                "tag_matched_score": 0.5,
                "total_score": 0.7
            }
        }

class ErrorResponse(BaseModel):
    detail: str
    status_code: int

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Error processing CV file",
                "status_code": 500
            }
        }
