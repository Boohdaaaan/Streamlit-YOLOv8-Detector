import streamlit as st
from utils import *

# Set Streamlit page configuration
st.set_page_config(
    page_title=" YOLOv8 Assistant",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title for the web app
st.title("Streamlit Object Tracker with YOLOv8")

# Sidebar for selecting image source
st.sidebar.title("Model Settings")
source = st.sidebar.radio("Select source:", ("Image", "Video", "YouTube", "Webcam"))

uploaded_image = None
uploaded_video = None
youtube_url = None

# Widget for uploading files
if source == "Image":
    uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

elif source == "Video":
    uploaded_video = st.sidebar.file_uploader("Choose a video...", type=["mp4"])

elif source == "YouTube":
    youtube_url = st.sidebar.text_input("Enter YouTube URL:")

# Confidence threshold and max detections sliders
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
max_detections = st.sidebar.slider("Max Detections", min_value=1, max_value=500, value=300, step=1)

# Add a multiselect widget
coco128_classes = load_yaml('classes.yaml')
class_names = list(coco128_classes.values())
selected_class_names = st.sidebar.multiselect('Select classes:', class_names, placeholder='Choose a class')

# Convert selected names back to corresponding class IDs
selected_class_ids = [class_id for class_id, class_name in coco128_classes.items() if class_name in selected_class_names]


# Perform object detection based on the selected source
if uploaded_image is not None:
    # Object detection for uploaded image
    image_detect(image=uploaded_image, confidence_threshold=confidence_threshold,
                 max_detections=max_detections, class_ids=selected_class_ids)

elif uploaded_video:
    # Object detection for uploaded video
    video_detect(source='video', uploaded_video=uploaded_video, confidence_threshold=confidence_threshold,
                 max_detections=max_detections, class_ids=selected_class_ids)

    # Remove temporary files
    remove_temp()

elif youtube_url:
    # Object detection for YouTube video
    video_detect(source='youtube', uploaded_video=youtube_url, confidence_threshold=confidence_threshold,
                 max_detections=max_detections, class_ids=selected_class_ids)

    # Remove temporary files
    remove_temp()

elif source == 'Webcam':
    # Real-time object detection from webcam feed
    video_detect(source='webcam', uploaded_video=None, confidence_threshold=confidence_threshold,
                 max_detections=max_detections, class_ids=selected_class_ids)

    # Remove temporary files
    remove_temp()
