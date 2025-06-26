# 🎙️ Emotion Recognition from Speech using LSTM (TensorFlow)

This project focuses on building an end-to-end pipeline for **emotion classification from speech audio files** using deep learning techniques. The final model is based on a **Long Short-Term Memory (LSTM)** neural network trained with **TensorFlow/Keras**, and is deployed in a **Streamlit web application** that allows users to upload `.wav` files and receive real-time emotion predictions.

---

## 📁 Project Structure

```
emotion-speech-classifier/
│
├── mars project.ipynb                # Jupyter Notebook with full training pipeline
├── lstm_emotion_classifier.h5       # Trained TensorFlow/Keras LSTM model
├── scaler.pkl                        # Scaler for feature normalization (StandardScaler)
├── app_mars.py                       # Streamlit web app
├── test_app.py                       # Script for testing model with .wav input
├── requirements.txt                  # List of Python dependencies
├── demo.mp4                          # 2-minute demo video of the web app
└── README.md                         # Project documentation (this file)
```

---

## 📌 Project Overview

This project seeks to classify emotions from speech signals using **MFCC (Mel Frequency Cepstral Coefficients)** and a **TensorFlow-based LSTM model**.

Supported emotion classes:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## 🔄 Preprocessing Workflow

### 1. Audio Preprocessing

- Input audio: `.wav` files
- Clip duration: \~3 seconds using `librosa.load(..., duration=3, offset=0.5)`
- Output: consistent shape for MFCC extraction

### 2. Feature Extraction using MFCC

- Extracted 40 MFCC features using `librosa`
- MFCC matrix padded to shape `(130, 40)` to standardize input size

### 3. Normalization and Label Encoding

- **StandardScaler** used for normalizing MFCCs → saved as `scaler.pkl`
- Emotion labels encoded to integers for model training

---

## 🧠 Model Architecture (TensorFlow / Keras LSTM)

| Layer      | Details               |
| ---------- | --------------------- |
| Input      | Shape: (130, 40)      |
| LSTM 1     | 128 units             |
| Dropout    | 0.4                   |
| LSTM 2     | 64 units              |
| Dropout    | 0.4                   |
| Dense (FC) | 64 → 8 output classes |

### Training Parameters:

- Loss Function: `categorical_crossentropy`
- Optimizer: `Adam`
- Epochs: `100`
- Batch Size: `32`

---

## 📈 Evaluation Results

✅ The model meets the accuracy criteria as per the project guidelines.

### 🔹 Accuracy Metrics

| Metric            | Value |
| ----------------- | ----- |
| Overall Accuracy  | 80%   |
| F1 Score (macro)  | 79%   |
| Weighted F1 Score | 80%   |

### 🔹 Per-Class Performance



### 🔹 Confusion Matrix



### 🔹 Training vs Validation Accuracy



### 🔹 Training vs Validation Loss



---

## 🌐 Streamlit Web App

### Run Locally:

```bash
streamlit run app_mars.py
```

### Features:

- Upload `.wav` file
- Hear playback in app
- See real-time emotion prediction

---

## 🧪 Command-line Testing

To test a `.wav` file outside the app:

```bash
python test_app.py test1.wav
```

This will return the predicted emotion in the terminal.

---

## 🗃️ Dataset

- **RAVDESS** dataset
- Audio files from 24 professional actors (12 male, 12 female)
- Covers all 8 emotions
- Used only speech (not singing) samples

---

## 🚀 Reproducibility Steps

1. Download RAVDESS dataset and set correct folder path in notebook
2. Run `mars project.ipynb` to extract features, train model
3. Save model as `lstm_emotion_classifier.h5`
4. Save scaler as `scaler.pkl`
5. Use app or test script to predict on new audio files

---

## 🎥 Demo Video

Demo video `demo.mp4` includes:

- Streamlit app launch
- Uploading a `.wav` file
- Playback and prediction demo
- Quick look at GitHub repo

---

## 🙋‍♂️ Author

Developed as part of a deep learning project on **emotion-aware audio classification** using TensorFlow and deployed via Streamlit.

---

Let us know if you found this helpful or wish to contribute! ✨

