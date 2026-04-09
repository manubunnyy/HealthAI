from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.chat_service import AgentResponse, DietPlan, HealthcareAgent

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize agent (singleton for simplicity in this scope)
# In a production app, you might want dependency injection or per-request initialization depending on state
agent = HealthcareAgent()


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    synthesis: str
    agent_responses: Dict[str, Any]
    diet_plan: Optional[Dict[str, Any]] = None


class ChatFollowupRequest(BaseModel):
    query: str
    feature_type: str
    original_result: str
    history: List[Dict[str, str]]


class ChatFollowupResponse(BaseModel):
    response: str


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    try:
        file_data = []
        for file in files:
            content = await file.read()
            file_data.append((file.filename, content, file.content_type))

        success = await agent.process_documents(file_data)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to process documents")

        return {"message": "Documents processed successfully", "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=ChatResponse)
async def process_query(request: ChatRequest):
    """Process a chat query"""
    try:
        responses = await agent.process_query(request.query)

        # Extract synthesis content
        synthesis_content = responses["synthesis_agent"].content

        # Format diet plan if present
        diet_plan_data = None
        if "diet_plan" in responses and responses["diet_plan"]:
            diet_plan_data = responses["diet_plan"]
            # Convert DietPlan object to dict if it's inside the response
            if "diet_plan" in diet_plan_data and isinstance(
                diet_plan_data["diet_plan"], DietPlan
            ):
                diet_plan_data["diet_plan"] = diet_plan_data["diet_plan"].__dict__

        # Format other agent responses
        formatted_responses = {}
        for key, value in responses.items():
            if isinstance(value, AgentResponse):
                formatted_responses[key] = {
                    "content": value.content,
                    "confidence": value.confidence,
                    "metadata": value.metadata,
                }

        return ChatResponse(
            synthesis=synthesis_content,
            agent_responses=formatted_responses,
            diet_plan=diet_plan_data,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/followup", response_model=ChatFollowupResponse)
async def process_followup(request: ChatFollowupRequest):
    """Process a contextual follow-up query"""
    try:
        response = await agent.process_contextual_chat(
            query=request.query,
            feature_type=request.feature_type,
            original_result=request.original_result,
            history=request.history
        )
        return ChatFollowupResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
