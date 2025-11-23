from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional
from app.services.report_service import HealthReportAnalyzer, AgentResponse
from PyPDF2 import PdfReader
import io

router = APIRouter(prefix="/report", tags=["report"])
analyzer = HealthReportAnalyzer()

class ReportAnalysisResponse(BaseModel):
    results: Dict[str, Any]

class DietPlanResponse(BaseModel):
    diet_plan: str

class DietPlanRequest(BaseModel):
    report_text: str

@router.post("/analyze", response_model=ReportAnalysisResponse)
async def analyze_report(file: UploadFile = File(...)):
    """Analyze health report"""
    try:

        # Use file.file directly to avoid loading entire file into RAM
        # UploadFile spools to disk for large files, so this is memory efficient
        
        if file.content_type == "application/pdf":
            try:
                # Reset file pointer to beginning
                await file.seek(0)
                # Pass the file-like object directly to the service for streaming processing
                results = await analyzer.analyze_report(file_object=file.file)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
        else:
            # For non-PDFs, we still need to read, but they are usually smaller text files
            content = await file.read()
            text = content.decode()
            results = await analyzer.analyze_report(text_content=text)
            
        # Explicitly clear large objects
        import gc
        gc.collect()
        
        # Convert AgentResponse objects to dicts
        formatted_results = {}
        for key, value in results.items():
            if isinstance(value, AgentResponse):
                formatted_results[key] = {
                    "content": value.content,
                    "confidence": value.confidence,
                    "processing_time": value.processing_time
                }
            else:
                formatted_results[key] = value
                
        return {"results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-diet-plan", response_model=DietPlanResponse)
async def generate_diet_plan(request: DietPlanRequest):
    """Generate diet plan from report text"""
    try:
        diet_plan = await analyzer.generate_diet_plan(request.report_text)
        return {"diet_plan": diet_plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
