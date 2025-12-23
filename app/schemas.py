import pydantic
from typing import List, Optional, Dict, Any

class PatientData(pydantic.BaseModel):

    age: float = pydantic.Field(..., gt=0, le=120)
    sex: int = pydantic.Field(..., ge=0, le=1) 
    cp: int = pydantic.Field(..., ge=0, le=3)  
    trestbps: float = pydantic.Field(..., gt=0, le=250)
    chol: float = pydantic.Field(..., gt=0, le=600)
    fbs: int = pydantic.Field(..., ge=0, le=1)
    restecg: int = pydantic.Field(..., ge=0, le=2)
    thalach: float = pydantic.Field(..., gt=0, le=250)
    exang: int = pydantic.Field(..., ge=0, le=1)
    oldpeak: float = pydantic.Field(..., ge=0, le=10) 
    slope: int = pydantic.Field(..., ge=1, le=3)
    ca: int = pydantic.Field(..., ge=0, le=4)  
    thal: int = pydantic.Field(..., ge=1, le=7) 

    class Config:
        feature_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]

class ShapFactor(pydantic.BaseModel):
    factor: str
    value: float
    impact: float
    interpretation: str

class RiskResponse(pydantic.BaseModel):

    risk_score: float
    risk_level: str
    risk_emoji: str
    top_factors: List[ShapFactor]
    shap_available: bool

class AdviceResponse(pydantic.BaseModel):

    ai_advice: str
    llm_available: bool
    rag_available: bool
    rag_used: bool