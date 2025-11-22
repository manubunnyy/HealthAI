import os
import requests
import base64
import random
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class DietService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set")

    def get_image_response(self, input_prompt: str, image_data: str) -> str:
        """Get response from Groq's vision model for image analysis"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        content = [
            {"type": "text", "text": input_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            }
        ]
        
        payload = {
            "model": "llama-3.2-11b-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0.7,
            "max_completion_tokens": 1024
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                response_json = response.json()
                return response_json['choices'][0]['message']['content']
            else:
                logger.error(f"Error from Groq API: {response.status_code} - {response.text}")
                raise Exception(f"Groq API Error: {response.status_code}")
        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            raise

    def get_chatbot_response(self, user_input: str) -> str:
        """Get response from Groq's text model for nutrition advice"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""You are an expert dietitian with a creative flair. Based on the following details about available foods and nutritional targets:
{user_input}
Provide an exciting and practical dietary plan that includes:
1. Recommended food portions in a visually appealing Markdown table format
2. Optimal meal timings with creative names for each meal
3. Clear explanations for your suggestions
4. One unexpected but scientifically-backed nutrition tip"""
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_completion_tokens": 1024
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                response_json = response.json()
                return response_json['choices'][0]['message']['content']
            else:
                logger.error(f"Error from Groq API: {response.status_code} - {response.text}")
                raise Exception(f"Groq API Error: {response.status_code}")
        except Exception as e:
            logger.error(f"Diet plan generation error: {str(e)}")
            raise

    def generate_fun_fact(self) -> str:
        fun_facts = [
            "Did you know? The average adult eats about 1,000 calories worth of food just to maintain basic bodily functions while sleeping.",
            "Fascinating fact: Celery is often called a 'negative-calorie food' because it takes more calories to digest than it contains.",
            "Cool calorie tip: Laughing for 10-15 minutes can burn between 10-40 calories!",
            "Nutrition nugget: Your brain uses about 20% of your daily calorie intake, despite being only 2% of your body weight.",
            "Food for thought: Spicy foods containing capsaicin can temporarily boost your metabolism by up to 8%!"
        ]
        return random.choice(fun_facts)
