import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image as PILImage
import random
import base64
import requests
import json
from io import BytesIO

# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="NutriVision AI | Smart Calorie Advisor",
    page_icon="🥗",
    layout="wide",
)

# Load environment variables
load_dotenv()

# Get Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Please set your GROQ_API_KEY in the .env file")

# Custom CSS for dark theme and creative styling
def apply_custom_styling():
    st.markdown("""
    <style>
    /* Dark theme with gradient background */
    .stApp {
        background: linear-gradient(135deg, #121212 0%, #242424 100%);
    }
    
    /* Headers with neon effect */
    h1, h2, h3 {
        color: #00ffaa;
        text-shadow: 0 0 10px rgba(0, 255, 170, 0.5);
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Buttons with gradient effect */
    .stButton>button {
        background: linear-gradient(45deg, #4a00e0, #8e2de2);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(45deg, #8e2de2, #4a00e0);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Text areas and inputs */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #444;
        border-radius: 8px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #333;
        border-radius: 6px;
        padding: 8px 16px;
        color: #ddd;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4a00e0;
        color: white;
    }
    
    /* File uploader */
    .stFileUploader>div>button {
        background: linear-gradient(45deg, #00c6ff, #0072ff);
    }
    
    /* Results area */
    .results-container {
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-left: 4px solid #00ffaa;
    }
    </style>
    """, unsafe_allow_html=True)

def get_image_response(input_prompt, image_data):
    """Get response from Groq's vision model for image analysis"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create the message content with text and image
    content = [
        {"type": "text", "text": input_prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}"
            }
        }
    ]
    
    # Prepare the API request payload
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
    
    # Make the API request
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    # Check if the request was successful
    if response.status_code == 200:
        response_json = response.json()
        return response_json['choices'][0]['message']['content']
    else:
        st.error(f"Error from Groq API: {response.status_code}")
        st.error(response.text)
        return "Sorry, there was an error analyzing the image."

def get_chatbot_response(user_input):
    """Get response from Groq's text model for nutrition advice"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are an expert dietitian with a creative flair. Based on the following details about available foods and nutritional targets:
{user_input}
Provide an exciting and practical dietary plan that includes:
1. Recommended food portions in a visually appealing table format
2. Optimal meal timings with creative names for each meal
3. Clear explanations for your suggestions
4. One unexpected but scientifically-backed nutrition tip"""
    
    # Prepare the API request payload - Updated model name to an available Groq model
    payload = {
        "model": "llama-3.1-8b-instant",  # Updated model name
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_completion_tokens": 1024
    }
    
    # Make the API request
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    # Check if the request was successful
    if response.status_code == 200:
        response_json = response.json()
        return response_json['choices'][0]['message']['content']
    else:
        st.error(f"Error from Groq API: {response.status_code}")
        st.error(response.text)
        return "Sorry, there was an error generating the dietary plan."

def process_image_for_api(uploaded_file):
    """Convert the uploaded image file to base64 for API request"""
    if uploaded_file is not None:
        # Read the file and encode it in base64
        bytes_data = uploaded_file.getvalue()
        base64_encoded = base64.b64encode(bytes_data).decode('utf-8')
        return base64_encoded
    else:
        raise FileNotFoundError("No file uploaded")

def generate_fun_fact():
    fun_facts = [
        "Did you know? The average adult eats about 1,000 calories worth of food just to maintain basic bodily functions while sleeping.",
        "Fascinating fact: Celery is often called a 'negative-calorie food' because it takes more calories to digest than it contains.",
        "Cool calorie tip: Laughing for 10-15 minutes can burn between 10-40 calories!",
        "Nutrition nugget: Your brain uses about 20% of your daily calorie intake, despite being only 2% of your body weight.",
        "Food for thought: Spicy foods containing capsaicin can temporarily boost your metabolism by up to 8%!"
    ]
    return random.choice(fun_facts)

# Apply custom styling
apply_custom_styling()

# App header with animation effect
st.markdown("""
<div style="text-align: center; animation: fadeIn 1.5s ease-in-out;">
    <h1>🥗 NutriVision AI</h1>
    <p style="color: #00ffaa; font-size: 1.2em; margin-bottom: 30px;">Your Intelligent Calorie & Nutrition Guide</p>
</div>
""", unsafe_allow_html=True)

# Display random fun fact
st.info(generate_fun_fact())

# Create two tabs with icons
tab1, tab2 = st.tabs(["📸 Food Scanner", "💬 Nutrition Advisor"])

with tab1:
    st.markdown("<h2>Scan Your Food for Instant Calorie Analysis</h2>", unsafe_allow_html=True)
    st.markdown("Take a photo of your meal and get instant nutritional insights powered by AI.")
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📤 Upload your food image", type=["jpg", "jpeg", "png"])
        
        analyze_button = st.button("🔍 Analyze Nutrition", use_container_width=True)
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Your delicious meal", use_column_width=True)
    
    with col2:
        if uploaded_file is not None and analyze_button:
            if not GROQ_API_KEY:
                st.error("Groq API key is missing. Please set the GROQ_API_KEY in your .env file.")
            else:
                with st.spinner("🤖 AI analyzing your food..."):
                    try:
                        base64_image = process_image_for_api(uploaded_file)
                        input_prompt = """You are a creative and engaging nutritionist. Analyze this food image and provide:
                        1. A list of all visible food items
                        2. Estimated calories for each item
                        3. Macronutrient breakdown (protein, carbs, fats)
                        4. One fun nutrition fact about the main ingredient
                        Format your response with emoji icons and clear headings."""
                        
                        response = get_image_response(input_prompt, base64_image)
                        
                        st.markdown("""
                        <div class="results-container">
                            <h3>✨ Nutrition Analysis Results</h3>
                            <div style="white-space: pre-line;">
                        """, unsafe_allow_html=True)
                        st.write(response)
                        st.markdown("</div></div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Oops! Something went wrong: {str(e)}")
        else:
            st.markdown("""
            <div style="background-color: #2d2d2d; border-radius: 10px; padding: 20px; height: 300px; display: flex; align-items: center; justify-content: center; text-align: center;">
                <div>
                    <h3 style="color: #888;">Your analysis will appear here</h3>
                    <p>Upload an image and click Analyze to get started</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h2>Personal Nutrition Advisor</h2>", unsafe_allow_html=True)
    st.markdown("Get customized meal plans based on your available ingredients and fitness goals.")
    
    user_input = st.text_area(
        "💭 Tell me about your available foods and nutritional goals",
        placeholder="Example: I have chicken breast, quinoa, spinach, sweet potatoes, and almonds. I'm trying to build lean muscle while maintaining energy for my morning workouts.",
        height=150
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        get_plan = st.button("🧠 Generate Meal Plan", use_container_width=True)
    
    if get_plan and user_input:
        if not GROQ_API_KEY:
            st.error("Groq API key is missing. Please set the GROQ_API_KEY in your .env file.")
        else:
            with st.spinner("Creating your personalized nutrition plan..."):
                try:
                    response = get_chatbot_response(user_input)
                    
                    st.markdown("""
                    <div class="results-container">
                        <h3>🍽️ Your Personalized Nutrition Plan</h3>
                        <div style="white-space: pre-line;">
                    """, unsafe_allow_html=True)
                    st.write(response)
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    # Add a download button for the plan
                    st.download_button(
                        label="📥 Download Your Plan",
                        data=response,
                        file_name="my_nutrition_plan.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Oops! Something went wrong: {str(e)}")