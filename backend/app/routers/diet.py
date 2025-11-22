from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from app.services.diet_service import DietService
import base64

router = APIRouter(prefix="/diet", tags=["diet"])
service = DietService()

class DietPlanRequest(BaseModel):
    user_input: str

@router.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)):
    """Analyze food image"""
    try:
        content = await file.read()
        base64_image = base64.b64encode(content).decode('utf-8')
        
        input_prompt = """You are a creative and engaging nutritionist. Analyze this food image and provide:
        1. A list of all visible food items
        2. Estimated calories for each item
        3. Macronutrient breakdown (protein, carbs, fats)
        4. One fun nutrition fact about the main ingredient
        Format your response with emoji icons and clear headings."""
        
        response = service.get_image_response(input_prompt, base64_image)
        return {"analysis": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-plan")
async def generate_plan(request: DietPlanRequest):
    """Generate diet plan"""
    try:
        response = service.get_chatbot_response(request.user_input)
        return {"plan": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fun-fact")
async def get_fun_fact():
    """Get a random nutrition fun fact"""
    return {"fact": service.generate_fun_fact()}
