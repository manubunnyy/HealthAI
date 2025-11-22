from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="HealthAI API", description="Backend for HealthAI Liquid Glass App")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import chat, diet, prediction, report, image

app.include_router(chat.router)
app.include_router(diet.router)
app.include_router(prediction.router)
app.include_router(report.router)
app.include_router(image.router)

@app.get("/")
async def root():
    return {"message": "HealthAI API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
