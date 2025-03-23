import os
import cv2
import torch
import uuid
import time
import json
import numpy as np
import streamlit as st
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from transformers import (
    AutoImageProcessor,
    AutoModelForVideoClassification,
    VideoClassificationPipeline
)
from dotenv import load_dotenv

# Force CUDA visible devices to ensure GPU is detected
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load environment variables
load_dotenv(override=True)  # Force reload and override any existing variables

# Diagnostics: Print CUDA availability information
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i} name: {torch.cuda.get_device_name(i)}")
        print(f"CUDA device {i} capability: {torch.cuda.get_device_capability(i)}")
        print(f"CUDA device {i} properties: {torch.cuda.get_device_properties(i)}")

# Configure GPU memory usage to prevent OOM errors
if torch.cuda.is_available():
    # Set memory usage growth instead of allocating all at once
    torch.cuda.empty_cache()
    # Enable memory optimization where available
    if hasattr(torch.cuda, 'amp'):
        print("Enabling automatic mixed precision for faster inference")

# Check if .env file exists
env_path = '.env'
if not os.path.exists(env_path):
    print("WARNING: .env file not found!")
else:
    # Read .env file manually to check contents
    with open(env_path, 'r') as f:
        env_contents = f.read()
        print(f".env file contents:\n{env_contents}")

# Email configuration
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS', "mangalarapumanu@gmail.com")  # Set default email directly
EMAIL_PASSWORD = os.getenv('EMAIL_APP_PASSWORD')
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587

# Debug email configuration
print("Email configuration loaded from environment:")
print(f"EMAIL_ADDRESS: {EMAIL_ADDRESS}")
print(f"EMAIL_APP_PASSWORD: {'*****' if EMAIL_PASSWORD else 'Not set'}")
print(f"EMERGENCY_CONTACTS: {os.getenv('EMERGENCY_CONTACTS', 'Not set')}")

# Emergency contacts (email addresses)
EMERGENCY_CONTACTS = os.getenv('EMERGENCY_CONTACTS', "").split(',') if os.getenv('EMERGENCY_CONTACTS') else []
EMERGENCY_CONTACTS = [contact.strip() for contact in EMERGENCY_CONTACTS if contact.strip()]

# Set up page configuration
st.set_page_config(
    page_title="Health Care Emergency Service",
    page_icon="🚨",
    layout="wide"
)

# Create required directories if they don't exist
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'static/frames'
for folder in [UPLOAD_FOLDER, FRAMES_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Global variables for model loading
model = None
processor = None
pipeline = None

def load_model():
    global model, processor, pipeline
    
    with st.spinner("Loading model... This might take a minute."):
        # Force CUDA detection again just to be safe
        if not torch.cuda.is_available():
            st.error("💥 CUDA not available despite having an NVIDIA GPU!")
            st.info("Attempting to force CUDA initialization...")
            # Try to manually initialize CUDA as a last resort
            try:
                torch.cuda.init()
                st.success("Manual CUDA initialization successful!")
            except Exception as e:
                st.error(f"Manual CUDA initialization failed: {e}")
                st.info("Check NVIDIA drivers and PyTorch CUDA compatibility")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Hide detailed GPU info from main UI
        if st.session_state.get('show_debug_info', False):
            st.info(f"Using device: {device}")
        
        # Check GPU availability more explicitly
        if device == "cuda":
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            
            # Only show in debug mode
            if st.session_state.get('show_debug_info', False):
                st.success(f"🚀 GPU detected: {gpu_name} (Count: {gpu_count})")
                torch.backends.cudnn.benchmark = True
                
                # Show memory information
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                reserved_mem = torch.cuda.memory_reserved(0) / 1024**3
                allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
                st.info(f"GPU Memory: {allocated_mem:.2f}GB allocated / {reserved_mem:.2f}GB reserved / {total_mem:.2f}GB total")
            else:
                torch.backends.cudnn.benchmark = True
        else:
            st.warning("No GPU detected - model will run slower on CPU")

        # Load the processor and model from local files
        try:
            processor = AutoImageProcessor.from_pretrained(
                "./model",
                local_files_only=True,
                use_fast=False
            )
            
            if torch.cuda.is_available():
                try:
                    # Explicitly move model to CUDA with forced device assignment
                    torch.cuda.set_device(0)  # Force using the first GPU
                    model = (AutoModelForVideoClassification
                            .from_pretrained("./model", local_files_only=True)
                            .cuda()  # Force CUDA
                            .half())  # Use half precision for speed
                    
                    # Verify model is actually on GPU - only show in debug mode
                    model_device = next(model.parameters()).device
                    if model_device.type != "cuda":
                        st.error(f"Model not on CUDA despite forcing it! Current device: {model_device}")
                        # Last resort: explicitly move every parameter
                        model = model.to("cuda:0")
                    elif st.session_state.get('show_debug_info', False):
                        st.success(f"✅ Model successfully loaded on GPU: {model_device}")
                except Exception as gpu_err:
                    st.error(f"GPU loading failed: {gpu_err}")
                    st.warning("Falling back to CPU as last resort")
                    model = AutoModelForVideoClassification.from_pretrained(
                        "./model",
                        local_files_only=True
                    )
            else:
                model = AutoModelForVideoClassification.from_pretrained(
                    "./model",
                    local_files_only=True
                )
                if st.session_state.get('show_debug_info', False):
                    st.info(f"Model loaded on CPU: {next(model.parameters()).device}")
            
            # Build a video classification pipeline with batch processing
            # Force device=0 to ensure GPU usage if available
            pipeline = VideoClassificationPipeline(
                model=model,
                feature_extractor=processor,
                device=0 if torch.cuda.is_available() else -1,  # Explicitly use first GPU (0)
                batch_size=1  # To reduce memory usage and increase reliability
            )
            
            # Verify pipeline is using the correct device - only in debug mode
            if hasattr(pipeline, 'device') and st.session_state.get('show_debug_info', False):
                st.info(f"Pipeline device: {pipeline.device}")
            
            st.success("Model loaded successfully!")
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.error("Try reinstalling PyTorch with CUDA support if GPU is not detected")
            return False

def send_email(subject, body, image_paths=None, recipients=None):
    """
    Send an email with optional image attachments using Gmail's secure SMTP
    
    Args:
        subject: Email subject
        body: Email body text
        image_paths: List of paths to image files to attach
        recipients: List of email addresses to send to
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use the configured email address
        sender_email = EMAIL_ADDRESS
            
        if not EMAIL_PASSWORD:
            st.error("Email password not configured. Please set EMAIL_APP_PASSWORD in .env file")
            return False

        # Use default recipient if none provided
        recipients = recipients if recipients else [sender_email]
        
        # Filter out empty recipients
        recipients = [r.strip() for r in recipients if r.strip()]
        
        if not recipients:
            st.error("No valid email recipients provided")
            return False
            
        # Create a multipart message
        message = MIMEMultipart('related')
        message["From"] = sender_email
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject
        
        # Create the HTML body part
        html_part = MIMEMultipart('alternative')
        html_content = MIMEText(body, "html")
        html_part.attach(html_content)
        message.attach(html_part)
        
        # Attach images if provided
        if image_paths:
            for i, img_path in enumerate(image_paths):
                try:
                    if not os.path.exists(img_path):
                        st.warning(f"Image not found: {img_path}")
                        continue
                        
                    with open(img_path, 'rb') as img_file:
                        img_data = img_file.read()
                        img_name = os.path.basename(img_path)
                        
                        # Create image attachment
                        image = MIMEImage(img_data)
                        image.add_header('Content-ID', f'<image{i}>')
                        image.add_header('Content-Disposition', 'attachment', filename=img_name)
                        message.attach(image)
                except Exception as img_error:
                    st.error(f"Error attaching image {img_path}: {img_error}")
                    continue
        
        # Connect to server and send email
        try:
            # Make sure app password has no extra spaces
            app_password = EMAIL_PASSWORD.strip() if EMAIL_PASSWORD else ""
            
            with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
                server.ehlo()  # Identify ourselves to the server
                
                # Start TLS encryption
                server.starttls()
                server.ehlo()  # Re-identify ourselves over TLS connection
                
                # Login to the server
                try:
                    if not app_password:
                        raise ValueError("App password is empty")
                        
                    login_response = server.login(sender_email, app_password)
                    st.info(f"Login successful with {sender_email}")
                except smtplib.SMTPAuthenticationError as auth_error:
                    st.error(f"Authentication failed: {auth_error}")
                    st.error(f"Email address used: {sender_email}")
                    st.error("Please double-check your app password and ensure it's correctly entered in the .env file")
                    return False
                except ValueError as ve:
                    st.error(f"Configuration error: {ve}")
                    return False
                
                # Send email
                server.send_message(message)
                
            st.success(f"Email sent successfully to {', '.join(recipients)}")
            return True
            
        except smtplib.SMTPAuthenticationError:
            st.error("""
            Gmail authentication failed. Please ensure:
            1. You've enabled 2-Step Verification in your Google Account
            2. You've generated an App Password:
               - Go to Google Account settings
               - Search for 'App Passwords'
               - Generate a new App Password for 'Mail'
            3. Use the generated App Password in your .env file:
               EMAIL_APP_PASSWORD=your_16_digit_app_password
            """)
            st.error(f"Current email address: {sender_email}")
            return False
        except Exception as smtp_error:
            st.error(f"SMTP error: {smtp_error}")
            return False
            
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def detect_anomalies(video_path, threshold=0.75):
    """
    Detect collisions or accidents in video and send alerts
    
    Args:
        video_path: Path to video file
        threshold: Anomaly detection threshold
    
    Returns:
        List of collision/accident frames
    """
    global model, processor, pipeline
    
    # Make sure model is loaded
    if model is None or processor is None or pipeline is None:
        load_model()
        if model is None:
            st.error("Failed to load model. Cannot proceed with detection.")
            return []
    
    # Force GPU usage if available but not already used - only show verbose info in debug mode
    if torch.cuda.is_available():
        if next(model.parameters()).device.type != "cuda":
            st.warning("Model not on GPU. Forcing GPU usage...")
            # First free memory
            torch.cuda.empty_cache()
            # Then move model to GPU explicitly
            torch.cuda.set_device(0)
            model = model.cuda().half()  # Force move to GPU with half precision
        
        # Show GPU memory usage only in debug mode
        if st.session_state.get('show_debug_info', False):
            allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
            st.info(f"✅ Using device for inference: {next(model.parameters()).device} | GPU Memory in use: {allocated_mem:.2f}GB")
    
    # Clear existing frames
    for filename in os.listdir(FRAMES_FOLDER):
        file_path = os.path.join(FRAMES_FOLDER, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    if st.session_state.get('show_debug_info', False):
        st.info(f"Video duration: {duration:.2f} seconds ({frame_count} frames)")
    progress_bar = st.progress(0)
    
    # Placeholder for detected collision/accident frames
    collision_frames = []
    
    # Variables for person detection (for human damage detection)
    # Use HOG for CPU or CUDA for GPU
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Load face detection for human damage detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Process in segments for the machine learning model (32 frames each)
    segment_size = 32  # Number of frames in each segment for the model (from config)
    total_segments = frame_count // segment_size + (1 if frame_count % segment_size > 0 else 0)
    
    # Variables for improved motion detection
    motion_threshold = 0.5  # Threshold for significant motion (percentage of image)
    prev_gray = None
    optical_flow_history = []
    
    # Variables to track potential crash indicators
    person_fall_detected = False
    sudden_motion_change = False
    significant_deformation = False
    
    # Get required image size from model config
    required_size = 224  # From the model config
    
    # Process the video in segments
    for segment_idx in range(total_segments):
        start_frame = segment_idx * segment_size
        end_frame = min(start_frame + segment_size, frame_count)
        
        # Update progress
        progress_bar.progress(segment_idx / total_segments)
        
        # Collect frames for this segment
        segment_frames = []
        frame_indices = []
        
        # Process each frame in the segment
        for frame_idx in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB (CV2 uses BGR, but our model expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to the expected dimensions for the model
            resized_frame = cv2.resize(rgb_frame, (required_size, required_size))
            
            # Save original frame for later reference
            frame_filename = f'frame_{frame_idx}.jpg'
            frame_path = os.path.join(FRAMES_FOLDER, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            # Store resized RGB frame and its index
            segment_frames.append(resized_frame)
            frame_indices.append(frame_idx)
            
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # For optical flow calculation
            if prev_gray is not None:
                # Calculate optical flow using Farneback method
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Calculate magnitude and angle of flow
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Calculate motion statistics
                mean_magnitude = np.mean(magnitude)
                max_magnitude = np.max(magnitude)
                std_magnitude = np.std(magnitude)
                
                # Store in history
                optical_flow_history.append({
                    'mean': mean_magnitude,
                    'max': max_magnitude,
                    'std': std_magnitude,
                    'frame_idx': frame_idx
                })
                
                # Keep only the last 10 frames of motion history
                if len(optical_flow_history) > 10:
                    optical_flow_history.pop(0)
                
                # Analyze motion patterns for collision detection
                if len(optical_flow_history) >= 10:
                    # Calculate recent motion statistics
                    recent_mean = np.mean([f['mean'] for f in optical_flow_history[-5:]])
                    prev_mean = np.mean([f['mean'] for f in optical_flow_history[-10:-5]])
                    
                    # Detect sudden motion change (spike or drop)
                    motion_ratio = recent_mean / max(prev_mean, 0.0001)
                    sudden_motion_change = (motion_ratio > 2.0) or (motion_ratio < 0.3 and prev_mean > 0.5)
                    
                    # Detect significant deformation (high standard deviation)
                    significant_deformation = optical_flow_history[-1]['std'] > 10.0
            
            prev_gray = gray
            
            # Person detection for human damage assessment
            # Resize frame for faster detection
            detect_frame = cv2.resize(frame, (640, 480))
            gray_detect = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect people
            people, _ = hog.detectMultiScale(detect_frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray_detect, 1.1, 4)
            
            # If people detected, analyze posture
            if len(people) > 0:
                for (x, y, w, h) in people:
                    # Check aspect ratio - a fallen person will have a wider than tall ratio
                    aspect_ratio = w / h
                    person_fall_detected = aspect_ratio > 1.5  # If width > 1.5*height, might be fallen
                    
                    if person_fall_detected:
                        break
        
        # Ensure we have exactly segment_size frames for prediction
        if len(segment_frames) < segment_size:
            # If we don't have enough frames, pad the segment_frames
            # by repeating the last frame until we have segment_size frames
            last_frame = segment_frames[-1] if segment_frames else np.zeros((required_size, required_size, 3), dtype=np.uint8)
            padding_needed = segment_size - len(segment_frames)
            segment_frames.extend([last_frame] * padding_needed)
            
            # Also pad the frame_indices
            last_idx = frame_indices[-1] if frame_indices else frame_count - 1
            frame_indices.extend([last_idx] * padding_needed)
        
        # When we have enough frames, run ML prediction
        if len(segment_frames) == segment_size:
            try:
                # Process frames individually first
                processed_frames = []
                for i in range(len(segment_frames)):
                    # Process each frame individually
                    single_frame = segment_frames[i]  # Shape: (224, 224, 3)
                    # Don't use the processor here, just convert to tensor and normalize
                    frame_tensor = torch.from_numpy(single_frame).float() / 255.0
                    # Rearrange from (H, W, C) to (C, H, W) which is what the model expects for each frame
                    frame_tensor = frame_tensor.permute(2, 0, 1)
                    processed_frames.append(frame_tensor)
                
                # Stack processed frames into a single tensor with shape (num_frames, channels, height, width)
                frames_tensor = torch.stack(processed_frames)
                
                # Add batch dimension and reshape to match ViViT expected shape
                # ViViT expects (batch_size, num_frames, channels, height, width)
                pixel_values = frames_tensor.unsqueeze(0)  # Shape becomes (1, num_frames, channels, height, width)
                
                # Explicitly move tensor to the same device as the model
                device = next(model.parameters()).device
                
                # Check if model is using half precision and convert input accordingly - hide message in normal mode
                if next(model.parameters()).dtype == torch.float16:
                    if st.session_state.get('show_debug_info', False):
                        st.info("Model is using half precision, converting input to match")
                    pixel_values = pixel_values.half()  # Convert to half precision (float16)
                
                pixel_values = pixel_values.to(device)
                
                # Create inputs dictionary
                inputs = {"pixel_values": pixel_values}
                
                # Get model predictions directly
                with torch.no_grad():
                    # Log shape and device for debugging - only in debug mode
                    if st.session_state.get('show_debug_info', False):
                        st.info(f"Input shape: {pixel_values.shape}, Device: {pixel_values.device}, Dtype: {pixel_values.dtype}")
                    
                    # Run inference
                    outputs = model(**inputs)
                    logits = outputs.logits
                
                # Convert to probabilities
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                # Get index of road accidents class
                road_accidents_idx = model.config.label2id.get('roadaccidents', 2)  # Default to 2 which was seen in config
                
                # Get road accident probabilities
                road_accident_prob = probs[0, road_accidents_idx].item()  # Taking first item as we process one video
                
                # Check if roadaccident probability exceeds threshold
                ml_accident_detected = road_accident_prob > threshold
                
                # Combine ML prediction with motion analysis for more accurate detection
                is_accident = ml_accident_detected or (
                    (sudden_motion_change or significant_deformation) and 
                    road_accident_prob > 0.4  # Lower threshold when motion indicators are present
                ) or (
                    person_fall_detected and road_accident_prob > 0.3  # Even lower when a person fall is detected
                )
                
                if is_accident:
                    # Add frames to collision frames
                    for i, frame_idx in enumerate(frame_indices):
                        time_in_seconds = frame_idx / fps
                        timestamp = time.strftime('%H:%M:%S', time.gmtime(time_in_seconds))
                        
                        # Frame path from saved frames
                        frame_path = os.path.join(FRAMES_FOLDER, f'frame_{frame_idx}.jpg')
                        
                        # Save the frame with classification info
                        accident_frame_filename = f'accident_{frame_idx}_{road_accident_prob:.3f}.jpg'
                        accident_frame_path = os.path.join(FRAMES_FOLDER, accident_frame_filename)
                        
                        # Copy the frame file with the accident label
                        if os.path.exists(frame_path):
                            # Add text annotation to indicate detection confidence
                            frame = cv2.imread(frame_path)
                            cv2.putText(frame, f"Accident: {road_accident_prob:.2f}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imwrite(accident_frame_path, frame)
                            
                            # Add to collision frames
                            collision_frames.append({
                                'frame': frame_idx,
                                'time': time_in_seconds,
                                'timestamp': timestamp,
                                'ml_score': road_accident_prob,
                                'motion_change': sudden_motion_change,
                                'deformation': significant_deformation,
                                'human_fall': person_fall_detected,
                                'frame_path': accident_frame_path
                            })
            except Exception as e:
                st.error(f"Error in ML prediction: {str(e)}")
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
    
    cap.release()
    progress_bar.progress(1.0)
    
    # Filter duplicates and sort
    # Keep only one frame per second to avoid too many similar frames
    filtered_frames = {}
    for frame in collision_frames:
        second = int(frame['time'])
        if second not in filtered_frames or frame['ml_score'] > filtered_frames[second]['ml_score']:
            filtered_frames[second] = frame
    
    # Convert back to list and sort
    collision_frames = list(filtered_frames.values())
    collision_frames.sort(key=lambda x: x['time'])
    
    # If collisions detected, send alerts
    if collision_frames:
        st.success(f"Detected {len(collision_frames)} potential collisions/accidents")
        
        # Sort collision frames by score to get top frames
        top_frames = sorted(collision_frames, key=lambda x: x['ml_score'], reverse=True)[:10]
        
        # Display top 10 frames using a more flexible layout
        st.subheader("Top 10 Accident Frames")
        
        # First row of 5
        if len(top_frames) > 0:
            cols1 = st.columns(min(5, len(top_frames)))
            for i, col in enumerate(cols1):
                if i < len(top_frames):
                    col.image(top_frames[i]['frame_path'], 
                             caption=f"Frame {top_frames[i]['frame']} - Score: {top_frames[i]['ml_score']:.2f}")
        
        # Second row of 5
        if len(top_frames) > 5:
            cols2 = st.columns(min(5, len(top_frames)-5))
            for i, col in enumerate(cols2):
                if i+5 < len(top_frames):
                    col.image(top_frames[i+5]['frame_path'], 
                             caption=f"Frame {top_frames[i+5]['frame']} - Score: {top_frames[i+5]['ml_score']:.2f}")
        
        # Send emergency alerts
        with st.spinner("Sending emergency email alerts..."):
            send_emergency_alerts(collision_frames, threshold)
    else:
        st.info("No collisions or accidents detected in the video")
    
    return collision_frames

def send_emergency_alerts(collision_frames, threshold=0.4):
    """
    Send emergency email alerts with collision/accident information
    
    Args:
        collision_frames: List of detected collisions/accidents
        threshold: User-defined model threshold (frames below this won't be sent)
    """
    # Exit if no collision frames detected
    if not collision_frames:
        st.warning("No collision/accident frames detected to send alerts")
        return
        
    # Create message content
    detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Sort all frames by score (highest to lowest)
    sorted_frames = sorted(collision_frames, key=lambda x: x['ml_score'], reverse=True)
    
    # Filter frames below the threshold
    filtered_frames = [f for f in sorted_frames if f['ml_score'] >= threshold]
    
    if not filtered_frames:
        st.warning(f"No frames with score above threshold ({threshold})")
        return
    
    # Create score segments in 10% increments starting from threshold
    # Round threshold down to nearest 10%
    threshold_base = int(threshold * 10) / 10
    
    # Create segments from threshold to 1.0 in 0.1 increments
    segments = {}
    current = threshold_base
    while current < 1.0:
        next_segment = min(current + 0.1, 1.0)
        segment_key = f"{int(current*100)}-{int(next_segment*100)}"
        segments[segment_key] = []
        current = next_segment
    
    # Special case for 90-100
    if "90-100" not in segments:
        segments["90-100"] = []
    
    # Categorize frames into score segments
    for frame in filtered_frames:
        score = frame['ml_score']
        for segment_key in segments.keys():
            lower, upper = map(lambda x: int(x)/100, segment_key.split('-'))
            if lower <= score < upper or (upper == 1.0 and score == 1.0):
                segments[segment_key].append(frame)
                break
    
    # Get the top frame from each segment
    frames_to_send = []
    for segment_key, segment_frames in segments.items():
        if segment_frames:
            # Take the highest scoring frame from this segment
            frames_to_send.append(segment_frames[0])  # Already sorted by score
    
    # Log how many frames we're sending
    st.info(f"Sending {len(frames_to_send)} frames in email from {len([k for k, v in segments.items() if v])} score segments")
    
    # Build HTML email body for notification
    subject = f"🚨 EMERGENCY: Vehicle Collision/Accident Detected at {detection_time}"
    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h2 {{ color: #cc0000; }}
            .emergency {{ background-color: #ffeeee; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            .details {{ background-color: #f8f8f8; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            .frames {{ display: flex; flex-direction: column; gap: 15px; }}
            .frame-info {{ background-color: #f8f8f8; padding: 10px; border-radius: 5px; margin-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h2>🚨 EMERGENCY ALERT: Vehicle Collision/Accident Detected</h2>
        
        <div class="emergency">
            <h3>Alert Information</h3>
            <p><strong>Time:</strong> {detection_time}</p>
            <p><strong>Model Threshold:</strong> {threshold:.2f} ({threshold*100:.0f}%)</p>
            <p><strong>Score Segments with Frames:</strong> {", ".join([k + "%" for k, v in segments.items() if v])}</p>
            <p><strong>Number of Frames in Email:</strong> {len(frames_to_send)}</p>
            <p><strong>Priority:</strong> HIGH</p>
        </div>
    """
    
    # Add each frame's info
    html_body += """
        <h3>Top Frames by Score Segment</h3>
        <div class="frames">
    """
    
    for frame in frames_to_send:
        score = frame['ml_score'] * 100  # Convert to percentage
        # Find which segment this frame belongs to
        segment_label = None
        for segment_key in segments.keys():
            lower, upper = map(lambda x: int(x)/100, segment_key.split('-'))
            if lower <= frame['ml_score'] < upper or (upper == 1.0 and frame['ml_score'] == 1.0):
                segment_label = f"{segment_key}%"
                break
                
        if not segment_label:
            segment_label = f"{int(score/10)*10}-{int(score/10)*10+10}%"
            
        html_body += f"""
            <div class="frame-info">
                <p><strong>Score Segment:</strong> {segment_label}</p>
                <p><strong>Frame Number:</strong> {frame['frame']}</p>
                <p><strong>Timestamp in Video:</strong> {frame['timestamp']}</p>
                <p><strong>Confidence Score:</strong> {frame['ml_score']:.2f} ({frame['ml_score']*100:.1f}%)</p>
                <p><strong>Motion Change:</strong> {"Yes" if frame.get('motion_change', False) else "No"}</p>
                <p><strong>Vehicle Deformation:</strong> {"Yes" if frame.get('deformation', False) else "No"}</p>
                <p><strong>Human Fall Detection:</strong> {"Yes" if frame.get('human_fall', False) else "No"}</p>
            </div>
        """
    
    # Close the HTML
    html_body += """
        </div>
        <p>The top frame from each score segment above the threshold is attached to this email.</p>
        <p>This is an automated emergency alert. Please respond immediately.</p>
    </body>
    </html>
    """
    
    # Include all selected frames in the email
    image_paths = [frame['frame_path'] for frame in frames_to_send]
    
    # Send to emergency contacts
    recipients = EMERGENCY_CONTACTS
    if not recipients or all(not r.strip() for r in recipients):
        recipients = [EMAIL_ADDRESS]  # Fallback to primary email if no emergency contacts
    
    # Send email with collision frame attachments
    email_sent = send_email(subject, html_body, image_paths, recipients)
    
    if email_sent:
        recipient_list = ", ".join(r for r in recipients if r.strip())
        st.success(f"Emergency email alert sent to: {recipient_list}")
        
        # Display all the frames that were sent
        st.subheader(f"Sent Accident Frames ({len(frames_to_send)} frames)")
        
        # Create columns for display
        max_cols = min(4, len(frames_to_send))
        rows_needed = (len(frames_to_send) + max_cols - 1) // max_cols
        
        # Display frames in rows of up to 4 columns each
        for row in range(rows_needed):
            cols = st.columns(max_cols)
            start_idx = row * max_cols
            end_idx = min(start_idx + max_cols, len(frames_to_send))
            
            for i in range(start_idx, end_idx):
                col_idx = i % max_cols
                frame = frames_to_send[i]
                score = frame['ml_score'] * 100
                
                # Find segment for this frame
                segment_label = None
                for segment_key in segments.keys():
                    lower, upper = map(lambda x: int(x)/100, segment_key.split('-'))
                    if lower <= frame['ml_score'] < upper or (upper == 1.0 and frame['ml_score'] == 1.0):
                        segment_label = f"{segment_key}%"
                        break
                        
                cols[col_idx].image(frame['frame_path'], 
                        caption=f"Segment: {segment_label}\nFrame: {frame['frame']}\nScore: {frame['ml_score']:.2f}")
    else:
        st.error("Failed to send emergency email alerts")

def main():
    # Declare global variables
    global EMAIL_ADDRESS
    
    # Initialize session state for debug mode if not exists
    if 'show_debug_info' not in st.session_state:
        st.session_state.show_debug_info = False
    
    # Check CUDA compatibility - move to debug section
    if not torch.cuda.is_available() and st.session_state.get('show_debug_info', False):
        st.error("CUDA not available! Your NVIDIA RTX 3060 is not being detected.")
        st.info("Debugging information:")
        st.code(f"""
        PyTorch version: {torch.__version__}
        CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}
        CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}
        Number of GPUs reported by PyTorch: {torch.cuda.device_count()}
        
        Potential solutions:
        1. Make sure NVIDIA drivers are properly installed
        2. Reinstall PyTorch with CUDA support matching your CUDA version
        3. Make sure no other process is using your GPU exclusively
        4. Try running the app with administrator privileges
        """)
    elif torch.cuda.is_available() and st.session_state.get('show_debug_info', False):
        st.success(f"CUDA available! Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            st.info(f"GPU {i}: {gpu_name}")
    
    # App title and header
    st.title("Health Care Emergency Service")
    st.markdown("---")
    
    # Debug section
    with st.expander("Debug Information"):
        # Toggle for debug info
        st.session_state.show_debug_info = st.checkbox("Show detailed technical information", value=st.session_state.show_debug_info)
        
        st.subheader("Environment Variables")
        st.code(f"""
EMAIL_ADDRESS = {EMAIL_ADDRESS}
EMERGENCY_CONTACTS = {EMERGENCY_CONTACTS}
APP_PASSWORD_SET = {"Yes" if EMAIL_PASSWORD else "No"}
        """)
        
        if st.button("Reload Environment Variables"):
            load_dotenv(override=True)
            st.success("Environment variables reloaded")
            st.experimental_rerun()
            
        if st.button("Test SMTP Connection"):
            try:
                with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    st.success("SMTP connection successful")
            except Exception as e:
                st.error(f"SMTP connection failed: {e}")
        
        # GPU information (only shown in debug section)
        if torch.cuda.is_available():
            st.subheader("GPU Information")
            st.markdown(f"""
            - **Device**: {torch.cuda.get_device_name(0)}
            - **Memory**: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB Total
            - **CUDA Version**: {torch.version.cuda}
            - **PyTorch Version**: {torch.__version__}
            """)
            
            # Current memory usage
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            st.progress(allocated / (torch.cuda.get_device_properties(0).total_memory / 1024**3))
            st.caption(f"Memory Usage: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Email notification setup
        st.subheader("Email Notifications")
        primary_email = st.text_input("Primary email address", value=EMAIL_ADDRESS)
        
        # Emergency contacts
        st.subheader("Emergency Contacts")
        emergency_contacts = st.text_area(
            "Enter emergency email addresses (one per line)", 
            value="\n".join(EMERGENCY_CONTACTS) if EMERGENCY_CONTACTS else ""
        )
        
        # Model settings
        st.subheader("Model Settings")
        threshold = st.slider("Anomaly detection threshold", 0.0, 1.0, 0.75)
        
        # Email Configuration Test
        st.subheader("Email Test")
        sender_email = st.text_input("Sender email for test", value=primary_email)
        
        # Test email button
        if st.button("Test Email Notification"):
            # Store original value and temporarily override it
            original_email = EMAIL_ADDRESS
            EMAIL_ADDRESS = sender_email
            
            test_html = """
            <html>
            <body>
                <h2>🔄 Email Connection Test</h2>
                <p>This is a test email from the HAWKEYEZ Emergency Anomaly Detection System.</p>
                <p>If you received this email, your email configuration is working correctly!</p>
            </body>
            </html>
            """
            email_addresses = [line.strip() for line in emergency_contacts.split('\n') if line.strip()]
            if not email_addresses and primary_email.strip():
                email_addresses = [primary_email]
            
            send_result = send_email(
                "HAWKEYEZ - Email Test", 
                test_html, 
                recipients=email_addresses
            )
            
            # Reset email address
            EMAIL_ADDRESS = original_email
            
            if send_result:
                st.success("Test email sent successfully!")
    
    # Main content
    st.header("Upload Video for Anomaly Detection")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        temp_file_path = os.path.join(UPLOAD_FOLDER, f"{str(uuid.uuid4())}_{uploaded_file.name}")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.video(temp_file_path)
        
        # Process button
        if st.button("Process Video"):
            # Load model on first run
            if model is None:
                load_model()
            
            # Process the video
            with st.spinner("Processing video for anomalies..."):
                anomalies = detect_anomalies(temp_file_path, threshold)
            
            # Display results summary
            if anomalies:
                st.subheader("Detection Results")
                st.info(f"Found {len(anomalies)} potential anomalies")
                
                # Check email status
                email_addresses = [line.strip() for line in emergency_contacts.split('\n') if line.strip()]
                if not email_addresses and primary_email.strip():
                    email_addresses = [primary_email]
                
                if email_addresses:
                    st.success(f"Emergency email alerts sent to {len(email_addresses)} recipients")
                else:
                    st.warning("No email addresses configured. Please set up email notifications in settings.")

if __name__ == "__main__":
    main() 
