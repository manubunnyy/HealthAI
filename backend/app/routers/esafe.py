from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
import resend
import base64

load_dotenv()

router = APIRouter(prefix="/esafe", tags=["esafe"])

# Configure logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Resend
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "re_4d4S623A_DSKoC5aXLmXUv6as1BoEXYJK")
resend.api_key = RESEND_API_KEY

class EmergencyAlert(BaseModel):
    type: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    text_address: Optional[str] = None

def send_email_alert(alert: EmergencyAlert, photos: List[bytes] = []):
    """Send emergency details via email using Resend API"""
    receiver_email = "mangalarapumanu@gmail.com"
    
    try:
        # Build email body
        body = (
            f"🚨 NEW EMERGENCY ALERT 🚨\n\n"
            f"Type: {alert.type}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        if alert.latitude and alert.longitude:
            maps_link = f"https://www.google.com/maps?q={alert.latitude},{alert.longitude}"
            body += (
                f"📍 Location Coordinates: {alert.latitude}, {alert.longitude}\n"
                f"🗺️ Google Maps: {maps_link}\n"
            )

        if alert.text_address:
            body += f"🏠 Provided Address: {alert.text_address}\n"

        # Prepare attachments
        attachments = []
        for i, photo_bytes in enumerate(photos):
            attachments.append({
                "filename": f"emergency_photo_{i+1}.jpg",
                "content": base64.b64encode(photo_bytes).decode()
            })

        # Send email via Resend
        params = {
            "from": "HealthAI eSafe <onboarding@resend.dev>",
            "to": [receiver_email],
            "subject": f"🚨 EMERGENCY ALERT: {alert.type}",
            "text": body,
        }
        
        if attachments:
            params["attachments"] = attachments

        response = resend.Emails.send(params)
        logger.info(f"Email alert sent successfully via Resend: {response}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email alert via Resend: {e}")
        return False

@router.post("/alert")
async def create_alert(
    background_tasks: BackgroundTasks,
    type: str = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    text_address: Optional[str] = Form(None),
    photos: List[UploadFile] = File(None)
):
    """Create an emergency alert and send email notification"""
    try:
        alert = EmergencyAlert(
            type=type,
            latitude=latitude,
            longitude=longitude,
            text_address=text_address
        )
        
        photo_contents = []
        if photos:
            for photo in photos:
                content = await photo.read()
                photo_contents.append(content)
        
        # Send alert in background
        background_tasks.add_task(send_email_alert, alert, photo_contents)
        
        return {"status": "success", "message": "Emergency alert dispatched"}
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))
