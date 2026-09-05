# 🙂 Emotion Vision AI

A [Streamlit](https://streamlit.io/) web app that uses your webcam to detect faces and predict the emotion shown, powered by a Keras/TensorFlow CNN trained on facial expression data. The repo also includes the model-training notebook and a standalone real-time (live video) version of the detector.

## Features

- 📸 Capture a snapshot directly from your browser's webcam (no file upload needed)
- 🧠 Face detection using OpenCV's Haar Cascade classifier
- 🎭 Emotion classification into 7 categories: **angry, disgust, fear, happy, neutral, sad, surprise**
- 🖼️ Annotated output image with bounding boxes and predicted labels
- 📊 Confidence score displayed for each detected face
- 🎥 Optional live-webcam variant (`realtimedetection.py`) for continuous, real-time detection outside the browser
- 📓 Full training notebook (`face.ipynb`) showing how the CNN was built and trained

DATASET LINK : https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

## Project Structure

```
.
├── app.py                    # Main Streamlit web app (snapshot-based detection)
├── realtimedetection.py      # Standalone OpenCV script for live webcam detection (runs locally, opens a video window)
├── face.ipynb                # Jupyter notebook: CNN training pipeline (data loading, preprocessing, model build/train)
├── emotiondetector.json      # Model architecture
├── emotiondetector.h5        # Trained model weights
├── requirements.txt          # Python dependencies (pip)
├── packages.txt              # System-level dependencies (for Streamlit Cloud deployment)
└── README.md
```

## Demo (Streamlit app)

1. Open the app in your browser.
2. Click **"Take a photo"** to activate your webcam and snap a picture.
3. The app detects any faces in the image, predicts the emotion for each, and displays:
   - An annotated image with bounding boxes and emotion labels
   - A results list showing the predicted emotion and confidence percentage per face

## Requirements

- Python 3.8+
- A webcam (for live capture) or a browser environment that supports camera access

### Dependencies (`requirements.txt`)

```
streamlit
tensorflow-cpu
keras
numpy
opencv-python-headless
pillow
```

### System packages (`packages.txt`)

```
libglib2.0-0
```

This is needed by `opencv-python-headless` on minimal Linux environments (e.g. Streamlit Community Cloud). If you're deploying to Streamlit Cloud, this file is picked up automatically — no action needed. For local installs on Debian/Ubuntu, install it manually if you hit an OpenCV import error:

```bash
sudo apt-get install -y libglib2.0-0
```

## Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd Emotion_vision_AI
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure `emotiondetector.h5` and `emotiondetector.json` are present in the project root (they're already included in this repo).

## Usage

### Streamlit web app (recommended)

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`) in your browser and use the **"Take a photo"** button.

### Real-time webcam detection (local script)

For continuous live detection using your machine's webcam directly (opens an OpenCV window instead of a browser):

```bash
python realtimedetection.py
```

Press `Esc` or close the window to stop. This script requires a local webcam and a display — it won't work in a headless/remote environment.

### Training your own model (`face.ipynb`)

The notebook expects an emotion-labeled image dataset (e.g. FER-2013) organized as:

```
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── (same subfolders)
```

Run the notebook cells to load and preprocess the images, build the CNN, train it, and export `emotiondetector.json` / `emotiondetector.h5`.

## How It Works

1. **Model loading** (`app.py`) — Loads the architecture from `emotiondetector.json` via `model_from_json`, then loads weights from `emotiondetector.h5`. It's cached with `@st.cache_resource` so it only loads once per session.
2. **Face detection** — Each captured frame is converted to grayscale and passed through OpenCV's `haarcascade_frontalface_default` classifier to locate faces.
3. **Preprocessing** — Each detected face is cropped, resized to 48×48 pixels, reshaped to `(1, 48, 48, 1)`, and normalized to the `[0, 1]` range.
4. **Prediction** — The preprocessed face is passed to the model, which outputs probabilities across the 7 emotion classes. The class with the highest probability is selected, along with its confidence score.
5. **Display** — The snapshot is annotated with bounding boxes and emotion labels, and a results summary is shown below the image.

## Troubleshooting

- **"Missing emotiondetector.h5"** — Ensure the model file is in the same directory as `app.py`.
- **"emotiondetector.json (architecture file) is missing"** — The `.h5` file only contains weights; make sure the matching `emotiondetector.json` is present alongside it.
- **"No face detected"** — Try better lighting, move closer to the camera, or make sure your face is fully visible and unobstructed.
- **Webcam not working in browser** — Make sure you've granted camera permissions to your browser, and that no other application is using the webcam.
- **OpenCV import errors on Linux** — Install the system package listed in `packages.txt` (`libglib2.0-0`).
- **`realtimedetection.py` window doesn't open** — This script needs a local display and webcam; it won't run in headless servers, containers, or over SSH without X forwarding.
