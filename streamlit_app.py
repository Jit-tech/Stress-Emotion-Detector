import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import dlib
from deepface import DeepFace
import gdown
import os

def download_predictor():
    file_id = "1TK2XoVcKTTei3MjFVpuHQuXiFLxN-G0F"
    output = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

download_predictor()

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def detect_stress(landmarks):
    left_eye = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)])
    right_eye = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])
    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    return "Stressed" if ear < 0.2 else "Relaxed"

class EmotionStressTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            landmarks = landmark_predictor(gray, face)
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_img = image[y:y+h, x:x+w]

            try:
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)[0]
                emotion = result['dominant_emotion']
            except:
                emotion = "Unknown"

            stress = detect_stress(landmarks)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{emotion}, {stress}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return image

st.title("🎥 Real-Time Stress & Emotion Detection")
st.write("Allow access to your webcam. Analysis will appear live.")

webrtc_streamer(
    key="emotion-stress",
    video_processor_factory=EmotionStressTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
