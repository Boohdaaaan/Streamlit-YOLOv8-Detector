import os
import io
import cv2
import yaml
import torch
import numpy as np
from PIL import Image
import streamlit as st
from typing import Union
from pytube import YouTube
from ultralytics import YOLO

# Define the device to be used for computation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize YOLO model
model = YOLO('yolov8n.pt')


# Function for loading data from yaml file
def load_yaml(file_path: str) -> dict:
    """
    Load YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Loaded YAML data.
    """
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(exc)


# Function for removing temporary files
def remove_temp(temp_file: str = 'temp') -> None:
    """
    Remove all files in the specified temporary directory.

    Args:
        temp_file (str, optional): Path to the temporary directory. Defaults to 'temp'.
    """
    for file in os.listdir(temp_file):
        os.remove(os.path.join(temp_file, file))


# Function for downloading an image with detected objects
def download_image(image: np.ndarray) -> None:
    """
    Downloads the image with detected objects.

    Args:
        image (np.ndarray): Image array with detected objects.
    """
    # Convert NumPy array to PIL.Image object
    image = Image.fromarray(image)

    # Convert image to bytes
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()

    # Display download button
    if st.download_button(label="Download Image", data=img_byte_array, file_name='detected_image.png',
                          mime='image/png'):
        st.success("Downloaded successfully!")


# Function for detecting objects in an image
def image_detect(image: str, confidence_threshold: float, max_detections: int, class_ids: list) -> None:
    """
    Detects objects in an image using YOLO model.

    Args:
        image (str): Path to the input image.
        confidence_threshold (float): Confidence threshold for object detection.
        max_detections (int): Maximum number of detections.
        class_ids (list): List of class IDs to consider for detection.
    """
    # Open the image
    image = Image.open(image)

    # Perform object detection
    results = model.predict(image, conf=confidence_threshold,
                            max_det=max_detections,classes=class_ids, device=DEVICE)

    # Plot the detected objects on the image
    plot = results[0].plot()

    # Convert color space from BGR to RGB
    processed_image = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)

    # Show the detected image
    st.image(processed_image, caption='Detected Image.', use_column_width='auto', output_format='auto', width=None)

    # Offer download option for the detected image
    download_image(processed_image)


# Function for real-time object detection in a video stream
def video_detect(source: str, uploaded_video: Union[None, io.BytesIO], confidence_threshold: float,
                 max_detections: int, class_ids: list) -> None:
    """
    Performs real-time object detection in a video stream.

    Args:
        source (str): Video source
        uploaded_video (Union[None, io.BytesIO, str]): Uploaded video file.
        confidence_threshold (float): Confidence threshold for object detection.
        max_detections (int): Maximum number of detections.
        class_ids (list): List of class IDs to consider for detection.
    """
    # Display for video feed
    video_feed = st.empty()

    # Check if a video is uploaded or using webcam
    if source == "video":
        # Create a temporary file to save the uploaded video
        temp_video_path = f"temp/temp_{uploaded_video.name}"

        # Write uploaded video content to the temporary file
        with open(temp_video_path, "wb") as temp_video_file:
            temp_video_file.write(uploaded_video.getvalue())

        # Open the uploaded video file
        cap = cv2.VideoCapture(temp_video_path)

    elif source == "youtube":
        try:
            # Create a temporary file to save the YouTube video
            yt = YouTube(uploaded_video)
            video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            video_stream.download(output_path="temp", filename="yt_video.mp4")
            cap = cv2.VideoCapture("temp/yt_video.mp4")
        except Exception as e:
            st.error("Error: Invalid YouTube Link")
            return

    elif source == "webcam":
        try:
            # Open webcam
            cap = cv2.VideoCapture(0)
        except Exception as e:
            st.error("Error: Unable to Access Webcam")
            return

    # Loop through frames of the video
    while cap.isOpened():
        success, frame = cap.read()
        if success:

            # Perform object detection on the frame
            results = model.track(frame, persist=True, conf=confidence_threshold,
                                  max_det=max_detections, classes=class_ids, device=DEVICE)

            # Plot the detected objects on the frame
            processed_frame = results[0].plot()

            # Display processed frame with detected objects
            video_feed.image(processed_frame, caption='Detected Video.', channels="BGR",
                             use_column_width='auto', output_format='auto', width=None)

        else:
            break
    # Release video capture
    cap.release()
