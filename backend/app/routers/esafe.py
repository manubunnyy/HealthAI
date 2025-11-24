from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import os
import logging
import requests
from datetime import datetime
from geopy.geocoders import Nominatim
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/esafe", tags=["esafe"])

# Configure logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

class EmergencyAlert(BaseModel):
    type: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    text_address: Optional[str] = None

def send_telegram_alert(alert: EmergencyAlert, photos: List[bytes] = []):
    """Send emergency details and images to admin chat"""
    try:
        if not TELEGRAM_BOT_TOKEN or not ADMIN_CHAT_ID:
            logger.error("Telegram credentials not found")
            return

        base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
        
        alert_message = (
            "🚨 NEW EMERGENCY ALERT 🚨\n\n"
            f"Type: {alert.type}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # Handle location information
        if alert.latitude and alert.longitude:
            lat, lon = alert.latitude, alert.longitude
            
            # Create Google Maps link
            maps_link = f"https://www.google.com/maps?q={lat},{lon}"
            
            # Add location information to message
            alert_message += (
                f"📍 Location Coordinates: {lat}, {lon}\n"
                f"🗺️ Google Maps: {maps_link}\n"
            )

            # Try to get address from coordinates using Nominatim
            try:
                geolocator = Nominatim(user_agent="healthai_emergency_app")
                location = geolocator.reverse(f"{lat}, {lon}")
                if location and location.address:
                    alert_message += f"📌 Reverse Geocoded Address: {location.address}\n"
            except Exception as geo_error:
                logger.error(f"Geocoding error: {geo_error}")

        if alert.text_address:
            alert_message += f"🏠 Provided Address: {alert.text_address}\n"
            # Try to get coordinates for the text address
            try:
                geolocator = Nominatim(user_agent="healthai_emergency_app")
                location = geolocator.geocode(alert.text_address)
                if location:
                    maps_link = f"https://www.google.com/maps?q={location.latitude},{location.longitude}"
                    alert_message += f"🗺️ Address Google Maps: {maps_link}\n"
            except Exception as geo_error:
                logger.error(f"Address geocoding error: {geo_error}")

        # Send text message
        message_data = {
            "chat_id": ADMIN_CHAT_ID,
            "text": alert_message,
            "parse_mode": "HTML"
        }
        requests.post(f"{base_url}/sendMessage", json=message_data)

        # Send photos if any
        if photos:
            for photo_bytes in photos:
                files = {"photo": photo_bytes}
                photo_data = {
                    "chat_id": ADMIN_CHAT_ID,
                    "caption": "Emergency situation photo"
                }
                requests.post(f"{base_url}/sendPhoto", data=photo_data, files=files)

        return True
    except Exception as e:
        logger.error(f"Failed to send emergency alert: {e}")
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
        
        # Send alert in background to not block response
        background_tasks.add_task(send_telegram_alert, alert, photo_contents)
        
        return {"status": "success", "message": "Emergency alert dispatched"}
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))
