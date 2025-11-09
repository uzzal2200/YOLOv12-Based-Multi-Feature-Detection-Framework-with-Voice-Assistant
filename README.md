# Object Text Detection for Visually Impaired

This repository is a small research / demo project for real-time object detection and text/object feedback aimed at assisting visually impaired users. It contains demo scripts, Jupyter notebooks, example model files, and small example datasets used during development and experiments.

Contents
--------
- `app.py` — a demo runner for real-time object detection that uses an Ultralytics YOLO model and provides Bengali audio feedback (via gTTS + pydub).
- `Footpath Detection/` — notebooks and model files for footpath detection experiments (example `.ipynb`, `.pt`).
- `Known_unknown_detection/` — a simple face detection/recognition script, sample images, and a movement log folder.
- `Object detection self dataset/` — example notebooks and resources for experiments using your own dataset.
- `object detection with COCO dataset/` — (optional) experiments with COCO-format data.

Quick overview
--------------
The project demonstrates two main flows:

1. Real-time object detection (camera) with spoken feedback in Bengali. `app.py` loads a local YOLO model and plays a short Bengali phrase for recognized class labels.
2. A simple known/unknown face detector (`Known_unknown_detection/known_unknown_detection.py`) that uses OpenCV Haar cascades and template-matching against stored face images. Detections are saved as images and recorded in `movement_log.xlsx`.

Important notes
---------------
- Large files: model files (`*.pt`) and notebooks can be large. GitHub disallows single files larger than 100 MB. Consider using Git LFS for models or store large datasets/models in cloud storage (Google Drive, S3, etc.).
- Virtual environments (e.g., `.venv`) are ignored by `.gitignore` and should not be committed.
- The face-recognition approach in `Known_unknown_detection` is a simple template-matching method — it is easy to understand but not robust to pose, scale, or lighting changes. For production-quality recognition use embeddings (e.g., `face_recognition`, `dlib`, or a deep model).

Setup (recommended)
-------------------
1. Create and activate a virtual environment (Windows PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies. There isn't a committed `requirements.txt` by default, but the main packages used across scripts include:

```powershell
pip install opencv-python ultralytics gTTS pydub numpy openpyxl
# optionally: pip install torch torchvision  # if your model requires it
```

3. If you will run `app.py`, make sure the model path in `app.py` matches the model file you have (for example `Footpath Detection/best.pt` or `save model/best.pt`).

Running the demos
-----------------
- Run the YOLO demo:

```powershell
python app.py
```

- Run the simple face detector (Known/Unknown):

```powershell
python Known_unknown_detection\known_unknown_detection.py
```

Customizing model paths
-----------------------
Edit `app.py` and change the path passed to `YOLO()` to the actual location of your `.pt` model file in the repository (or an absolute path). Example:

```python
# inside app.py
model = YOLO('Footpath Detection/best.pt')
```

Object-text-detection-for-visually-impaired