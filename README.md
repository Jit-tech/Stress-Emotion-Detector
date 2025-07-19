# Stress & Emotion Detector (Webcam + Streamlit)

This app uses DeepFace and dlib to detect emotion and stress in real time from webcam video.

## How to Use

1. Deploy on [Streamlit Cloud](https://streamlit.io/cloud)
2. Upload this repo to GitHub
3. Create a new app pointing to `streamlit_app.py`

## Features

- Real-time face detection
- Emotion classification (DeepFace)
- Eye aspect ratio–based stress detection
- Auto-downloads `shape_predictor_68_face_landmarks.dat` from Google Drive

## Requirements

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
