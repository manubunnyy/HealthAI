from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import json

load_dotenv()

router = APIRouter(prefix="/esafe", tags=["esafe"])

# Configure logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

class EmergencyAlert(BaseModel):
    type: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    text_address: Optional[str] = None

# Email Configuration File
EMAIL_CONFIG_FILE = "email_config.json"

class EmailConfig(BaseModel):
    sender_email: str
    sender_password: str
    receiver_email: str

def load_email_config():
    if os.path.exists(EMAIL_CONFIG_FILE):
        with open(EMAIL_CONFIG_FILE, "r") as f:
            return json.load(f)
    return None

def save_email_config(config: EmailConfig):
    with open(EMAIL_CONFIG_FILE, "w") as f:
        json.dump(config.dict(), f)

def send_email_alert(alert: EmergencyAlert, photos: List[bytes] = []):
    """Send emergency details via email"""
    # Temporary hardcoded credentials
    config = {
        "sender_email": "mangalarapumanu@gmail.com",
        "sender_password": "itea ctjl brvv rklk", # Spaces are usually ignored by Gmail SMTP
        "receiver_email": "mangalarapumanu@gmail.com"
    }
    
    if not config:
        logger.warning("Email configuration not found. Skipping email alert.")
        return

    try:
        print(f"Attempting to send email from {config['sender_email']} to {config['receiver_email']}")
        msg = MIMEMultipart()
        msg['From'] = config['sender_email']
        msg['To'] = config['receiver_email']
        msg['Subject'] = f"🚨 EMERGENCY ALERT: {alert.type}"

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

        msg.attach(MIMEText(body, 'plain'))

        # Attach photos
        for i, photo_bytes in enumerate(photos):
            img = MIMEImage(photo_bytes)
            img.add_header('Content-Disposition', 'attachment', filename=f"emergency_photo_{i+1}.jpg")
            msg.attach(img)

        # Send email - Using port 465 (SSL) instead of 587 (TLS) for Render compatibility
        print("Connecting to SMTP server on port 465...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.set_debuglevel(1) # Enable SMTP debug output
            print("Logging in...")
            server.login(config['sender_email'], config['sender_password'])
            print("Sending message...")
            server.send_message(msg)
        
        print("Email sent successfully!")
        logger.info("Email alert sent successfully")
        return True

    except Exception as e:
        print(f"ERROR SENDING EMAIL: {e}")
        logger.error(f"Failed to send email alert: {e}")
        return False

@router.post("/config")
async def update_email_config(config: EmailConfig):
    """Update email configuration"""
    try:
        save_email_config(config)
        return {"status": "success", "message": "Email configuration updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_email_config():
    """Get current email configuration (masking password)"""
    config = load_email_config()
    if config:
        config['sender_password'] = "********"
        return config
    return {}

@router.post("/alert")
async def create_alert(
    background_tasks: BackgroundTasks,
    type: str = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    text_address: Optional[str] = Form(None),
    photos: List[UploadFile] = File(None)
):
    """
    Create an emergency alert.
    Accepts form data to handle both text fields and file uploads in one request.
    """
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
        
        # Send alerts in background
        background_tasks.add_task(send_email_alert, alert, photo_contents)
        
        return {"status": "success", "message": "Emergency alert dispatched"}
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))
