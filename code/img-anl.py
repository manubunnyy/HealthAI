import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st
import tempfile
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API Key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure API Key is provided
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Please set your Google API Key in the .env file as GOOGLE_API_KEY")

# Initialize the Medical Agent
medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# Medical Analysis Query with reduced reliance on DuckDuckGo search
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the medical image and structure your response as follows:

### 1. Image Type & Region
- Identify imaging modality (X-ray/MRI/CT/Ultrasound/etc.).
- Specify anatomical region and positioning.
- Evaluate image quality and technical adequacy.

### 2. Key Findings
- Highlight primary observations systematically.
- Identify potential abnormalities with detailed descriptions.
- Include measurements and densities where relevant.

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level.
- List differential diagnoses ranked by likelihood.
- Support each diagnosis with observed evidence.
- Highlight critical/urgent findings.

### 4. Patient-Friendly Explanation
- Simplify findings in clear, non-technical language.
- Avoid medical jargon or provide easy definitions.
- Include relatable visual analogies.

### 5. Research Context
- Include general treatment protocols and management approaches based on your existing knowledge.
- Note that online search may not be available, so rely on your internal medical knowledge.
- If search functionality is working, provide 1-2 key references.

Ensure a structured and medically accurate response using clear markdown formatting.
"""

# Function to analyze medical image
def analyze_medical_image(image_path):
    """Processes and analyzes a medical image using AI."""
    
    try:
        # Open and resize image
        image = PILImage.open(image_path)
        width, height = image.size
        aspect_ratio = width / height
        new_width = 500
        new_height = int(new_width / aspect_ratio)
        resized_image = image.resize((new_width, new_height))

        # Create temporary file with proper extension
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_path = temp_file.name
            resized_image.save(temp_path)

        # Create AgnoImage object
        agno_image = AgnoImage(filepath=temp_path)

        # Run AI analysis with timeout handling
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = medical_agent.run(query, images=[agno_image])
                return response.content
            except Exception as e:
                if "timeout" in str(e).lower() and attempt < max_retries - 1:
                    st.warning(f"Search timed out, retrying... ({attempt+1}/{max_retries})")
                    time.sleep(2)  # Wait before retrying
                else:
                    # Continue with analysis even if search fails
                    st.warning("External search failed. Analysis will continue with AI's internal knowledge.")
                    # Try one more time with disabled search
                    try:
                        response = medical_agent.run(query, images=[agno_image])
                        return response.content
                    except Exception as final_e:
                        return f"⚠️ Analysis error: {final_e}"

    except Exception as e:
        return f"⚠️ Image processing error: {e}"
    finally:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass  # Ignore errors during cleanup

# Streamlit UI setup
st.set_page_config(page_title="Medical Image Analysis", layout="centered")
st.title("🩺 Medical Image Analysis Tool 🔬")
st.markdown(
    """
    Welcome to the **Medical Image Analysis** tool! 📸
    Upload a medical image (X-ray, MRI, CT, Ultrasound, etc.), and our AI-powered system will analyze it, providing detailed findings, diagnosis, and research insights.
    Let's get started!
    """
)

# Upload image section
st.sidebar.header("Upload Your Medical Image:")
uploaded_file = st.sidebar.file_uploader("Choose a medical image file", type=["jpg", "jpeg", "png", "bmp", "gif"])

# Button to trigger analysis
if uploaded_file is not None:
    # Display the uploaded image in Streamlit
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    if st.sidebar.button("Analyze Image"):
        with st.spinner("🔍 Analyzing the image... Please wait."):
            try:
                # Create a temporary file with proper extension
                file_extension = uploaded_file.type.split('/')[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    image_path = temp_file.name
                
                # Run analysis on the uploaded image
                report = analyze_medical_image(image_path)
                
                # Display the report
                st.subheader("📋 Analysis Report")
                st.markdown(report, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
            finally:
                # Clean up the saved image file if it exists
                if 'image_path' in locals() and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except:
                        pass  # Ignore errors during cleanup
else:
    st.warning("⚠️ Please upload a medical image to begin analysis.")
