import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.applications.resnet50 import preprocess_input
import tempfile
import os

from audio_utils import extract_audio
from audio_model import get_audio_confidence


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Deepfake Detector",
    page_icon="🎭",
    layout="wide"
)

# ---------------- CUSTOM STYLE ---------------- #

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
}
.sub-text {
    font-size:18px;
    color:gray;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🎭 AI Deepfake Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Multimodal Analysis using Video & Audio Signals</p>', unsafe_allow_html=True)

st.divider()

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("⚙️ Settings")

THRESHOLD = st.sidebar.slider(
    "Video Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

video_weight = st.sidebar.slider(
    "Video Weight",
    0.0, 1.0, 0.7, 0.05
)

audio_weight = 1 - video_weight

st.sidebar.markdown("---")
st.sidebar.info("Adjust sensitivity and confidence weighting.")


# ---------------- LOAD MODELS ---------------- #

@st.cache_resource
def load_video_model():
    return tf.keras.models.load_model("deepfake_detector.h5")

video_model = load_video_model()
detector = MTCNN()

IMG_SIZE = 224


# ---------------- FACE FUNCTIONS ---------------- #

def extract_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)

    if len(results) == 0:
        return None

    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)

    face = rgb_frame[y:y+h, x:x+w]

    if face.size == 0:
        return None

    return face


def predict_face(face):
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = np.expand_dims(face, axis=0)
    face = preprocess_input(face)

    prediction = video_model.predict(face, verbose=0)[0][0]
    return prediction


# ---------------- ANALYSIS FUNCTION ---------------- #

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    predictions = []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 25

    frame_interval = fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            face = extract_face(frame)
            if face is not None:
                prob = predict_face(face)
                predictions.append(prob)

        frame_count += 1

    cap.release()

    # ---------------- VIDEO RESULT ----------------
    if len(predictions) == 0:
        video_prediction = "No Face Detected"
        video_confidence = 0.0
    else:
        avg_prob = np.mean(predictions)

        if avg_prob < THRESHOLD:
            video_prediction = "Deepfake"
            video_confidence = (1 - avg_prob) * 100
        else:
            video_prediction = "Authentic"
            video_confidence = avg_prob * 100

    # ---------------- AUDIO RESULT ----------------
    audio_path = extract_audio(video_path)

    if audio_path is None:
        audio_confidence = 0.0
    else:
        audio_confidence = get_audio_confidence(audio_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

    # ---------------- FINAL SCORE ----------------
    final_score = (video_weight * video_confidence) + \
                  (audio_weight * audio_confidence)

    return {
        "video_prediction": video_prediction,
        "video_confidence": video_confidence,
        "audio_confidence": audio_confidence,
        "final_score": final_score
    }


# ---------------- MAIN UI ---------------- #

uploaded_file = st.file_uploader(
    "Upload Video File",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📹 Video Preview")
        st.video(uploaded_file)

    with col2:
        st.subheader("📊 Analysis Results")

        if st.button("🔍 Run Deepfake Analysis"):

            with st.spinner("Analyzing video and audio signals..."):
                result = analyze_video(temp_path)

            st.divider()

            # Status Badge
            if result["video_prediction"] == "Deepfake":
                st.error("🚨 Deepfake Detected")
            elif result["video_prediction"] == "Authentic":
                st.success("✅ Authentic Video")
            else:
                st.warning("⚠️ No Face Detected")

            st.divider()

            # Confidence Bars
            st.write("🎥 Video Confidence")
            st.progress(int(result["video_confidence"]))

            st.write("🔊 Audio Confidence")
            st.progress(int(result["audio_confidence"]))

            st.divider()

            # Final Metric
            st.metric(
                label="🎯 Final Deepfake Score",
                value=f"{result['final_score']:.2f}%"
            )

    os.remove(temp_path)

st.divider()

st.caption("Powered by Deep Neural Networks | Multimodal Deepfake Detection System")
