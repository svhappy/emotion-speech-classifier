import streamlit as st
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("lstm_emotion_classifier.h5")
scaler = joblib.load("scaler.pkl")

emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def preprocess_audio(file):
    y, sr = librosa.load(file, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    mfcc = librosa.util.fix_length(mfcc, size=130, axis=0)
    mfcc = scaler.transform(mfcc)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc

st.title("🎙️ Emotion Detection from Speech")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    features = preprocess_audio("temp.wav")
    prediction = model.predict(features)
    predicted_emotion = emotion_labels[np.argmax(prediction)]

    st.markdown(f"### 🧠 Predicted Emotion: **{predicted_emotion.capitalize()}**")
