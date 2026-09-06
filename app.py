import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from keras.models import model_from_json, load_model as keras_load_model

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Emotion Vision AI", page_icon="🙂", layout="centered")
st.title("🙂 Emotion Vision AI")
st.write("Developed by Shrusti Diggavi, IPEC Solutions Private Limited Bangalore")
st.write(
    "Take a snapshot with your webcam and the model will detect faces and "
    "predict the emotion shown."
)

MODEL_JSON_PATH = "emotiondetector.json"
MODEL_WEIGHTS_PATH = "emotiondetector.h5"
LABELS = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}


# ----------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        return None, "Missing emotiondetector.h5 (trained model/weights file)."

    # Case 1: emotiondetector.h5 is a FULL saved model (architecture + weights,
    # e.g. saved with model.save(...)). No json needed for this case.
    try:
        model = keras_load_model(MODEL_WEIGHTS_PATH, compile=False)
        return model, None
    except Exception:
        pass

    # Case 2: emotiondetector.h5 is WEIGHTS ONLY (e.g. saved with
    # model.save_weights(...)) and needs the architecture json alongside it.
    if not os.path.exists(MODEL_JSON_PATH):
        return None, (
            "emotiondetector.h5 doesn't load as a full model, and "
            "emotiondetector.json (architecture file) is missing to pair with it."
        )
    try:
        with open(MODEL_JSON_PATH, "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(MODEL_WEIGHTS_PATH)
        return model, None
    except Exception as e:
        return None, f"Found both files but failed to load the model: {e}"


@st.cache_resource
def load_face_cascade():
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(haar_path)


def extract_features(face_img):
    feature = np.array(face_img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


model, load_error = load_model()
face_cascade = load_face_cascade()

if load_error:
    st.error(
        f"⚠️ Could not load the model: {load_error}\n\n"
        "Make sure both `emotiondetector.json` and `emotiondetector.h5` are in the "
        "same folder as this app before deploying."
    )
    st.stop()

# ----------------------------------------------------------------------
# Webcam snapshot input
# ----------------------------------------------------------------------
snapshot = st.camera_input("Take a photo")

if snapshot is not None:
    # Convert the uploaded snapshot into an OpenCV BGR image
    pil_image = Image.open(snapshot).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected. Try again with better lighting or a closer shot.")
    else:
        results = []
        for (x, y, w, h) in faces:
            face_gray = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_gray, (48, 48))
            features = extract_features(face_resized)
            prediction = model.predict(features, verbose=0)
            label = LABELS[int(np.argmax(prediction))]
            confidence = float(np.max(prediction)) * 100
            results.append((label, confidence))

            # Draw bounding box + label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
            )

        annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detected faces & emotions", use_container_width=True)

        st.subheader("Results")
        for i, (label, confidence) in enumerate(results, start=1):
            st.write(f"Face {i}: **{label}** ({confidence:.1f}% confidence)")
