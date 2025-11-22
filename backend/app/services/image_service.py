import os
import base64
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class ImageAnalysisService:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not set")
        
        self.analysis_prompt = """You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the medical image and structure your response as follows:

### 1. Image Type & Region
- Identify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Specify anatomical region and positioning
- Evaluate image quality and technical adequacy

### 2. Key Findings
- Highlight primary observations systematically
- Identify potential abnormalities with detailed descriptions
- Include measurements and densities where relevant

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses ranked by likelihood
- Support each diagnosis with observed evidence
- Highlight critical/urgent findings

### 4. Patient-Friendly Explanation
- Simplify findings in clear, non-technical language
- Avoid medical jargon or provide easy definitions
- Include relatable visual analogies

### 5. Recommendations
- Suggest follow-up imaging if needed
- Recommend specialist consultations
- Note any urgent actions required

Ensure a structured and medically accurate response using clear markdown formatting."""

    def analyze_medical_image(self, image_content: bytes) -> str:
        """Analyzes medical image using Groq's vision model"""
        try:
            # Convert image to base64
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            
            # Prepare request for Groq vision API
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 2048
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                response_json = response.json()
                return response_json['choices'][0]['message']['content']
            elif response.status_code == 429:
                return """⚠️ **Rate Limit Reached**

The Groq API has reached its rate limit. Please try again in a few minutes.

**What you can do:**
- Wait 1-2 minutes before trying again
- The free tier has limited requests per minute (30/min)

**Note:** This is a temporary limitation from Groq's API."""
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"⚠️ Analysis error: API returned status {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "⚠️ Request timed out. Please try again with a smaller image or wait a moment."
        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            return f"⚠️ Analysis error: {str(e)}"
