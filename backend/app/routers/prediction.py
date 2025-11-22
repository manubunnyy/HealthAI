from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/prediction", tags=["prediction"])
service = PredictionService()

class PredictionRequest(BaseModel):
    prediction_type: str
    weight: float
    height: float
    glucose_fasting: Optional[float] = None
    glucose_post_meal: Optional[float] = None
    heart_rate: Optional[float] = None
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None

class PredictionResponse(BaseModel):
    bmi: float
    bmi_category: str
    insights: List[str]
    recommendations: List[str]

@router.post("/analyze", response_model=PredictionResponse)
async def analyze_health(request: PredictionRequest):
    """Analyze health metrics"""
    try:
        # Calculate BMI
        bmi = service.calculate_bmi(request.weight, request.height)
        
        # Prepare metrics dictionary
        metrics = request.dict(exclude_none=True)
        metrics['bmi'] = bmi
        
        # Get insights
        insights = service.get_health_insights(metrics, request.prediction_type)
        recommendations = service.get_recommendations()
        bmi_category = service.get_bmi_category(bmi)
        
        return PredictionResponse(
            bmi=bmi,
            bmi_category=bmi_category,
            insights=insights,
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_config():
    """Get prediction configuration"""
    return service.config
