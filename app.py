import streamlit as st
import cv2
from fer import FER
import numpy as np
import librosa
import time
import threading
import pygame
from tensorflow.keras.models import load_model
from moviepy.editor import VideoFileClip
import tempfile

def detect_video_emotion(frame, detector):
    emotions = detector.detect_emotions(frame)
    if emotions:
        dominant_emotion = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
        return dominant_emotion[0]
    return "No face detected"

def extract_mfcc(audio_data, sample_rate):
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    return mfccs

def get_audio_emotion(prediction):
    emotion_map = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
    max_index = np.argmax(prediction)
    return emotion_map.get(max_index, "Unknown")

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    video.close()
    return audio_path

def process_video_and_audio(video_path):
    video_detector = FER(mtcnn=True)
    audio_model_path = "C:/Users/BASKARANKASTHURI/PycharmProjects/pythonProject1/model.h5"
    audio_model = load_model(audio_model_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    audio_path = extract_audio(video_path)
    audio_data, sr = librosa.load(audio_path, sr=22050)
    chunk_duration = 1
    chunk_samples = int(chunk_duration * sr)

    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

    start_time = time.time()
    audio_emotion = "Initializing..."

    def process_audio():
        nonlocal audio_emotion
        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            mfcc_features = extract_mfcc(chunk, sr)
            mfcc_features = mfcc_features.reshape(1, 40, 1)
            prediction = audio_model.predict(mfcc_features)
            audio_emotion = get_audio_emotion(prediction[0])
            time.sleep(chunk_duration)

    audio_thread = threading.Thread(target=process_audio)
    audio_thread.start()

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        video_emotion = detect_video_emotion(frame, video_detector)

        cv2.putText(frame, f"Video Emotion: {video_emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Audio Emotion: {audio_emotion}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        stframe.image(frame, channels="BGR")

        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break


        current_time = time.time() - start_time
        expected_frame = int(current_time * fps)
        actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if expected_frame > actual_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame)

    cap.release()
    pygame.mixer.music.stop()
    audio_thread.join()

def main():
    st.title("Video and Audio Emotion Detection")

    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

    if st.button("Process Video"):
        if video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
                temp_video_file.write(video_file.read())
                temp_video_file_path = temp_video_file.name
                process_video_and_audio(temp_video_file_path)

if __name__ == "__main__":
    main()
