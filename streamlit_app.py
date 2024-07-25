import streamlit as st
import requests
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
import imutils
import time
from playsound import playsound
from threading import Thread, Event
from collections import deque

# Load models
prototxtPath = "Project/source/face_detector/deploy.prototxt"
weightsPath = "Project/source/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")

# Alarm function
def play_warning_sound(sound_event):
    while True:
        sound_event.wait()  # Wait until signaled to play sound
        playsound('alarm.mp3')

# Function to detect and predict mask usage
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces, locs, preds = [], [], []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Function to fetch sensor data from the server
def fetch_sensor_data():
    try:
        response = requests.get("http://192.168.1.44:5000/")
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching sensor data: {e}")
        return {}

# Display video stream and detect masks
def display_video_stream(stop_event):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    frame_placeholder = st.empty()

    # Initialize sound event and thread
    sound_event = Event()
    sound_thread = Thread(target=play_warning_sound, args=(sound_event,))
    sound_thread.daemon = True
    sound_thread.start()

    alarm_playing = False
    mask_timers = deque(maxlen=45)  # 1.5 seconds of frames at 30 FPS

    try:
        while not stop_event.is_set():
            frame = vs.read()
            if frame is None:
                st.error("Unable to read frame from video stream.")
                break

            frame = imutils.resize(frame, width=800)
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            sensor_data = fetch_sensor_data()

            temperature = sensor_data.get('temperature', 'N/A')
            humidity = sensor_data.get('humidity', 'N/A')
            air_quality = sensor_data.get('air_quality', 'N/A')

            sensor_text = f"Temp: {temperature} C, Humidity: {humidity} %, Air Quality: {air_quality}"
            warning_texts = []

            if temperature != 'N/A':
                if temperature < 23.0:
                    warning_texts.append("Suhu rendah")
                elif temperature > 35.0:
                    warning_texts.append("Suhu tinggi")

            if humidity != 'N/A':
                if humidity < 50.0:
                    warning_texts.append("Kelembapan udara rendah")
                elif humidity > 88.0:
                    warning_texts.append("Kelembapan udara tinggi")

            if air_quality != 'N/A' and air_quality >= 50:
                warning_texts.append("Kualitas udara buruk")

            show_mask_warning = False
            show_specific_warning = False

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                if label.startswith("No Mask"):
                    show_mask_warning = True

            # Update mask timers
            mask_timers.append(show_mask_warning)

            # Check if mask warning should be shown
            if sum(mask_timers) / len(mask_timers) > 0.5:
                show_mask_warning = True
            else:
                show_mask_warning = False

            cv2.putText(frame, sensor_text, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            y_offset = 30
            for warning_text in warning_texts:
                cv2.putText(frame, warning_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                y_offset += 30

                if warning_text in ["Suhu rendah", "Suhu tinggi", "Kelembapan udara rendah", "Kelembapan udara tinggi",
                                    "Kualitas udara buruk"]:
                    show_specific_warning = True

            if show_specific_warning and show_mask_warning:
                cv2.putText(frame, "Harap menggunakan masker", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                if not alarm_playing:
                    sound_event.set()  # Activate sound
                    alarm_playing = True
            else:
                sound_event.clear()  # Deactivate sound
                alarm_playing = False

            frame_placeholder.image(frame, channels="BGR")

    finally:
        vs.stop()
        sound_event.clear()
        sound_thread.join(timeout=1)

# Streamlit UI
st.title("Air D Pollution")
st.write("Monitoring air quality and mask usage in real-time.")

# Initialize session state for video streaming
if 'video_streaming' not in st.session_state:
    st.session_state.video_streaming = False
    st.session_state.stop_event = None

# Create placeholders for buttons
start_button = st.empty()
stop_button = st.empty()

# Manage button actions
if not st.session_state.video_streaming:
    if start_button.button("Start Video Stream"):
        st.session_state.video_streaming = True
        st.session_state.stop_event = Event()

if st.session_state.video_streaming:
    if stop_button.button("Stop Video Stream"):
        st.session_state.video_streaming = False
        if st.session_state.stop_event:
            st.session_state.stop_event.set()
        st.write("Video stream stopped.")

# Display video stream
if st.session_state.video_streaming:
    display_video_stream(st.session_state.stop_event)
