import sys
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the saved model and scaler
model = load_model("lstm_emotion_classifier.h5")
scaler = joblib.load("scaler.pkl")

# Emotion labels (must match training order)
emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    mfcc = librosa.util.fix_length(mfcc, size=130, axis=0)
    mfcc = scaler.transform(mfcc)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_app.py <audio_file.wav>")
        sys.exit()

    file_path = sys.argv[1]
    features = extract_features(file_path)
    prediction = model.predict(features)
    predicted = emotions[np.argmax(prediction)]
    print(f"🎧 Predicted Emotion: {predicted.capitalize()}")
