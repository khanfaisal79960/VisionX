import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np

st.set_page_config(page_title="Vision X", layout="wide")
st.title("🔍 Vision X")
st.subheader("See the Unseen — Real-Time Object Detection")

@st.cache_resource
def load_model():
    # Load YOLOv8 model (make sure yolov8n.pt is in the right place or accessible)
    return YOLO("yolov8n.pt")

model = load_model()

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO detection on the image
        results = model(img)[0]

        # Get annotated frame (this returns a numpy array with boxes drawn)
        annotated_img = results.plot()

        # Convert annotated image from RGB to BGR if needed (YOLO returns RGB)
        # But results.plot() in ultralytics returns BGR by default, so no conversion required here.

        # Return frame to streamlit-webrtc
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

webrtc_streamer(
    key="yolo",
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
