from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.image_service import ImageAnalysisService

router = APIRouter(prefix="/image", tags=["image"])
service = ImageAnalysisService()


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze medical image"""
    try:
        content = await file.read()
        analysis = service.analyze_medical_image(content)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
