# 🙂 Emotion Vision AI

A simple [Streamlit](https://streamlit.io/) web app that uses your webcam to detect faces and predict the emotion shown, powered by a Keras/TensorFlow CNN model trained on facial expression data.

## Features

- 📸 Capture a snapshot directly from your webcam (no file upload needed)
- 🧠 Face detection using OpenCV's Haar Cascade classifier
- 🎭 Emotion classification into 7 categories: **angry, disgust, fear, happy, neutral, sad, surprise**
- 🖼️ Annotated output image with bounding boxes and predicted labels
- 📊 Confidence score displayed for each detected face

## Demo

1. Open the app in your browser.
2. Click **"Take a photo"** to activate your webcam and snap a picture.
3. The app detects any faces in the image, predicts the emotion for each, and displays:
   - An annotated image with bounding boxes and emotion labels
   - A results list showing the predicted emotion and confidence percentage per face

## Project Structure

```
.
├── app.py                    # Main Streamlit application
├── emotiondetector.json      # Model architecture (required if .h5 is weights-only)
├── emotiondetector.h5        # Trained model or model weights
└── README.md
```

> **Note:** `emotiondetector.h5` can be either a full saved model (`model.save(...)`) or weights-only (`model.save_weights(...)`). The app automatically detects which format it is:
> - If it's a full model, it loads directly — no JSON file needed.
> - If it's weights-only, `emotiondetector.json` (the architecture file) must be present alongside it.

## Requirements

- Python 3.8+
- A webcam (for live capture) or a browser environment that supports camera access

### Dependencies

```
streamlit
opencv-python
numpy
pillow
keras
tensorflow
```

## Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. Install dependencies:
   ```bash
   pip install streamlit opencv-python numpy pillow keras tensorflow
   ```

3. Make sure `emotiondetector.h5` (and `emotiondetector.json`, if needed) are placed in the same folder as `app.py`.

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`) in your browser.

## How It Works

1. **Model loading** — The app attempts to load `emotiondetector.h5` as a complete Keras model. If that fails, it falls back to reconstructing the architecture from `emotiondetector.json` and loading the weights separately. The model is cached with `@st.cache_resource` so it only loads once per session.
2. **Face detection** — Each captured frame is converted to grayscale and passed through OpenCV's `haarcascade_frontalface_default` classifier to locate faces.
3. **Preprocessing** — Each detected face is cropped, resized to 48×48 pixels, reshaped to `(1, 48, 48, 1)`, and normalized to the `[0, 1]` range.
4. **Prediction** — The preprocessed face is passed to the model, which outputs probabilities across 7 emotion classes. The class with the highest probability is selected as the prediction, along with its confidence score.
5. **Display** — The original snapshot is annotated with bounding boxes and emotion labels, and a summary of results is printed below the image.

## Troubleshooting

- **"Missing emotiondetector.h5"** — Ensure the model file is in the same directory as `app.py`.
- **"emotiondetector.json (architecture file) is missing"** — Your `.h5` file only contains weights; place the matching `emotiondetector.json` next to it.
- **"No face detected"** — Try better lighting, move closer to the camera, or make sure your face is fully visible and unobstructed.
- **Webcam not working in browser** — Make sure you've granted camera permissions to your browser, and that no other application is using the webcam.

