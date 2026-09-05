# Deploying Emotion Vision AI on Streamlit

## 1. Add the missing model weights
This folder needs `emotiondetector.h5` (the trained weights) sitting right next
to `app.py` and `emotiondetector.json`. It wasn't in the original zip.

- If you already trained the model before (e.g. ran `face.ipynb` locally),
  copy `emotiondetector.h5` from that machine into this folder.
- If you don't have it, you'll need to re-run the training notebook
  (`face.ipynb`) — which requires the FER2013-style `dataset/train` and
  `dataset/test` folders (also not included in the zip) — then it will
  produce both `emotiondetector.json` and `emotiondetector.h5`.

## 2. Files in this project
- `app.py` — the Streamlit app (snapshot-based webcam capture + emotion prediction)
- `emotiondetector.json` — model architecture (included)
- `emotiondetector.h5` — trained weights (**you need to add this**)
- `requirements.txt` — Python dependencies for Streamlit Cloud
- `packages.txt` — system packages Streamlit Cloud needs for OpenCV

## 3. Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (make sure `emotiondetector.h5` is
   included — if it's over 100MB, use Git LFS).
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", pick your repo/branch, and set the main file to `app.py`.
4. Deploy. Streamlit Cloud will read `requirements.txt` and `packages.txt`
   automatically.

## Notes
- The original `realtimedetection.py` used `cv2.VideoCapture(0)` +
  `cv2.imshow(...)`, which opens a native window and only works when run
  directly on your own machine — it cannot work in a deployed web app.
  `app.py` replaces that with `st.camera_input()`, a browser-based snapshot
  button that works both locally and on Streamlit Cloud.
