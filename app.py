import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Vision X", layout="wide")
st.title("🔍 Vision X")
st.subheader("See the Unseen — Real-Time Object Detection")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Use yolov8s.pt or higher for more accuracy

model = load_model()

# Start webcam
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not accessible.")
            break

        results = model(frame)[0]
        annotated_frame = results.plot()

        FRAME_WINDOW.image(annotated_frame, channels='BGR')
    cap.release()
