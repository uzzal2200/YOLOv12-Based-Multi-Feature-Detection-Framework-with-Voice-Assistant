import os
import sys
import time
import uuid
from pathlib import Path
import tempfile
import winsound
from collections import defaultdict
import cv2
import face_recognition
import numpy as np
import pytesseract
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from ultralytics import YOLO

# ===================== PATH SETUP =====================
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)
os.environ.setdefault("PYDUB_TEMP", AUDIO_DIR.as_posix())
os.environ.setdefault("TEMP", AUDIO_DIR.as_posix())
os.environ.setdefault("TMP", AUDIO_DIR.as_posix())
tempfile.tempdir = AUDIO_DIR.as_posix()

# Uncomment if tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# ===================== MODEL PATHS =====================
CURRENCY_WEIGHTS = BASE_DIR / "Bangldesh currencey Detection" / "Save Model" / "best.pt"
FOOTPATH_WEIGHTS = BASE_DIR / "Footpath Detection" / "Save Model" / "best.pt"
GENERAL_WEIGHTS = BASE_DIR / "Object detection Custom dataset" / "Save Model" / "best.pt"

# Default video (falls back to webcam if not found)
DEFAULT_VIDEO = Path(r"D:\Research\Research\Real time object detection feedback with vedio\Object-text-detection-for-visually-impaired\video_2026-01-21_22-08-00.mp4")

# ===================== AUDIO COOLDOWN =====================
COOLDOWN_SEC = 2.0
last_spoken: dict[str, float] = {}
FACE_ENABLED = True  # Enable face detection
OVERLAY_TTL_FRAMES = 90

# ===================== AUDIO FILE MAPPING =====================
# Maps detected class labels to audio files from the audio folder
AUDIO_FILES = {
    # Object Detection classes
    "Vehicle": "Vehicle.mp3",
    "Chair": "Chair.mp3",
    "Door": "Door.mp3",
    "Man": "Man.mp3",
    "Road": "Road.mp3",
    "Stair": "Stair.mp3",
    "Table": "Table.mp3",
    "Tree": "Tree.mp3",
    "wall": "wall.mp3",
    
    # Currency Detection classes
    "1 Tk": "1 tk.mp3",
    "2 Taka": "2 taka.mp3",
    "5 Tk": "5 tk.mp3",
    "10 Tk": "10 Tk.mp3",
    "20 Tk": "20 tk.mp3",
    "50 Tk": "50 tk.mp3",
    "100 Tk": "100 tk.mp3",
    "500 Tk": "500 tk.mp3",
    "1000 Tk": "1000 tk.mp3",
    
    # Footpath Detection classes
    "free for use": "free for use.mp3",
    "Fully Occupied": "Fully Occupied .mp3",
    "Not for Safe": "Not safe for use.mp3",
    "Partially Occupied": "Partially Occupied .mp3",
    
    # Face Recognition
    "known_face": "Known Face Uzzal .mp3",
    "unknown_face": "Unknown Face.mp3"
}




# ===================== UTILS =====================
def cooldown_ok(key):
    now = time.time()
    if key not in last_spoken or now - last_spoken[key] > COOLDOWN_SEC:
        last_spoken[key] = now
        return True
    return False

def play_audio_file(label):
    """Play audio file from audio folder for detected class"""
    audio_file = AUDIO_FILES.get(label)
    if not audio_file:
        return False
    
    audio_path = AUDIO_DIR / audio_file
    if not audio_path.exists():
        print(f"[WARN] Audio file not found: {audio_file}")
        return False
    
    try:
        audio = AudioSegment.from_mp3(audio_path.as_posix())
        wav_path = audio_path.with_stem(f"{audio_path.stem}_{uuid.uuid4().hex[:4]}").with_suffix(".wav")
        audio.export(wav_path.as_posix(), format="wav")
        
        winsound.PlaySound(wav_path.as_posix(), winsound.SND_FILENAME | winsound.SND_ASYNC)
        print(f"[AUDIO] Playing: {audio_file}")
        return True
    except Exception as e:
        print(f"[WARN] Audio play error for {label}: {e}")
        return False

def bangla_speak(text, tag):
    """Generate and play Bangla speech"""
    filename = f"{tag}_{int(time.time())}_{uuid.uuid4().hex[:6]}.mp3"
    filepath = AUDIO_DIR / filename
    try:
        tts = gTTS(text=text, lang="bn")
        tts.save(filepath.as_posix())
        audio = AudioSegment.from_mp3(filepath.as_posix())

        # Play via winsound using a WAV export to avoid temp folder permissions
        wav_path = filepath.with_suffix(".wav")
        audio.export(wav_path.as_posix(), format="wav")
        try:
            winsound.PlaySound(wav_path.as_posix(), winsound.SND_FILENAME | winsound.SND_ASYNC)
            print(f"[BANGLA] Audio played: {text}")
        except Exception as play_err:
            print(f"[WARN] Audio play fallback error: {play_err}")
            play(audio)

        print(f"[BANGLA] Audio saved: {filepath}")
    except Exception as e:
        print(f"[WARN] Bangla speech error: {e}")

def debug_frame(frame, reason=""):
    # Print one-line frame diagnostics to help debug invalid images
    try:
        print(f"[DEBUG] {reason} shape={getattr(frame, 'shape', None)} dtype={getattr(frame, 'dtype', None)} min={frame.min() if frame is not None else None} max={frame.max() if frame is not None else None}")
    except Exception:
        print(f"[DEBUG] {reason} frame info unavailable")

def ensure_8bit_bgr(frame):
    # Normalize frame to 8-bit 3-channel BGR for downstream libs that require it
    if frame is None:
        return None
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)
    return np.ascontiguousarray(frame)


def draw_overlay(frame, text, color=(0, 255, 0)):
    # Draw a simple text overlay on the top-left of the frame
    if frame is None:
        return
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

def get_bn_label(label, mapping):
    return mapping.get(label.lower(), f"এটা {label}")

# ===================== VIDEO SOURCE =====================
def get_video_source():
    # Command line: --video <path>
    if len(sys.argv) > 2 and sys.argv[1] == "--video":
        candidate = Path(sys.argv[2])
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        if candidate.exists():
            print(f"[INFO] Using video: {candidate}")
            return candidate.as_posix()
        else:
            print(f"[WARN] Video not found: {candidate}. Falling back to defaults.")

    # Default video if present
    if DEFAULT_VIDEO.exists():
        print(f"[INFO] Using default video: {DEFAULT_VIDEO}")
        return DEFAULT_VIDEO.as_posix()

    # Webcam fallback
    print("[INFO] Using webcam (no video provided).")
    return 0

# ===================== LOAD MODELS =====================
def load_yolo(path, name):
    if not path.exists():
        print(f"[WARN] {name} model not found:", path)
        return None
    try:
        model = YOLO(path.as_posix())
        print(f"[INFO] Loaded {name} model from {path.name}")
        return model
    except Exception as exc:
        print(f"[WARN] Failed to load {name} model: {exc}")
        return None

currency_model = load_yolo(CURRENCY_WEIGHTS, "Currency")
footpath_model = load_yolo(FOOTPATH_WEIGHTS, "Footpath")
general_model = load_yolo(GENERAL_WEIGHTS, "General")

# ===================== FACE LOAD =====================
def load_known_faces():
    encodings, names = [], []
    faces_dir = BASE_DIR / "Known_unknown_detection" / "known_faces_folder"
    if not faces_dir.exists():
        return encodings, names

    for img in faces_dir.iterdir():
        if img.suffix.lower() in [".jpg", ".png"]:
            try:
                bgr = cv2.imread(img.as_posix())
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                enc = face_recognition.face_encodings(rgb)
                if enc:
                    encodings.append(enc[0])
                    names.append(img.stem)
            except Exception as exc:
                print(f"[WARN] Skipping face file {img.name}: {exc}")
    return encodings, names

known_encodings, known_names = load_known_faces()

# ===================== FACE DETECTION =====================
def handle_faces(frame):
    """Detect faces and play corresponding audio"""
    frame = ensure_8bit_bgr(frame)
    if frame is None:
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    try:
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
    except RuntimeError as exc:
        debug_frame(frame, "face_detection_error")
        print(f"[WARN] Face detection skipped: {exc}")
        return

    for enc in encs:
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
        if True in matches:
            key = "known_face"
            if cooldown_ok(key):
                play_audio_file("known_face")
                print("[FACE] Known face detected")
        else:
            key = "unknown_face"
            if cooldown_ok(key):
                play_audio_file("unknown_face")
                print("[FACE] Unknown face detected")

# ===================== OCR =====================
def handle_ocr(frame):
    """Detect text and generate voice output"""
    frame = ensure_8bit_bgr(frame)
    if frame is None:
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="eng").strip()
    if text and cooldown_ok("ocr"):
        # Generate voice output for detected text
        bangla_speak(f"লেখা শনাক্ত হয়েছে: {text}", "ocr")
        print(f"[OCR] Detected text: {text}")

# ===================== MAIN =====================
def main():
    video_src = get_video_source()
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print("[ERROR] Camera/video not opened")
        return

    print("[INFO] Stream opened, starting detection loop...")
    print("[INFO] Loaded Models:")
    print(f"  - Currency: {CURRENCY_WEIGHTS.name} ({CURRENCY_WEIGHTS.exists()})")
    print(f"  - Footpath: {FOOTPATH_WEIGHTS.name} ({FOOTPATH_WEIGHTS.exists()})")
    print(f"  - General Object: {GENERAL_WEIGHTS.name} ({GENERAL_WEIGHTS.exists()})")
    print(f"  - Known Faces: {len(known_names)}")
    print("[INFO] Press 'Q' to quit, 'C' for currency, 'F' for footpath, 'O' for object, 'X' for face, 'R' for OCR")

    frame_count = 0
    active_modes = {'currency': True, 'footpath': True, 'object': True, 'face': True, 'ocr': True}
    detection_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream")
            break

        frame = ensure_8bit_bgr(frame)
        if frame is None:
            continue

        frame_count += 1
        
        # Display current active modes
        mode_display = f"Modes: C={active_modes['currency']} F={active_modes['footpath']} O={active_modes['object']} X={active_modes['face']} R={active_modes['ocr']}"
        cv2.putText(frame, mode_display, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # ===== CURRENCY DETECTION =====
        if active_modes['currency'] and frame_count % 3 == 0 and currency_model is not None:
            try:
                results = currency_model(frame, verbose=False, imgsz=320, conf=0.35, max_det=5)
                if results and results[0].boxes:
                    for cls_id in results[0].boxes.cls:
                        label = currency_model.names[int(cls_id)]
                        key = f"currency_{label}"
                        if cooldown_ok(key):
                            if play_audio_file(label):
                                print(f"[CURRENCY] Detected: {label}")
            except Exception as e:
                print(f"[WARN] Currency detection error: {e}")

        # ===== FOOTPATH DETECTION =====
        if active_modes['footpath'] and frame_count % 3 == 0 and footpath_model is not None:
            try:
                results = footpath_model(frame, verbose=False, imgsz=320, conf=0.35, max_det=5)
                if results and results[0].boxes:
                    for cls_id in results[0].boxes.cls:
                        label = footpath_model.names[int(cls_id)]
                        key = f"footpath_{label}"
                        if cooldown_ok(key):
                            if play_audio_file(label):
                                print(f"[FOOTPATH] Detected: {label}")
            except Exception as e:
                print(f"[WARN] Footpath detection error: {e}")

        # ===== GENERAL OBJECT DETECTION =====
        if active_modes['object'] and frame_count % 3 == 0 and general_model is not None:
            try:
                results = general_model(frame, verbose=False, imgsz=320, conf=0.35, max_det=5)
                if results and results[0].boxes:
                    for cls_id in results[0].boxes.cls:
                        label = general_model.names[int(cls_id)]
                        key = f"general_{label}"
                        if cooldown_ok(key):
                            if play_audio_file(label):
                                print(f"[OBJECT] Detected: {label}")
            except Exception as e:
                print(f"[WARN] Object detection error: {e}")

        # ===== FACE DETECTION =====
        if active_modes['face'] and frame_count % 5 == 0:
            handle_faces(frame)

        # ===== OCR DETECTION =====
        if active_modes['ocr'] and frame_count % 30 == 0:
            handle_ocr(frame)

        # Display and handle user input
        cv2.imshow("Real-time Object Detection - Press Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("[INFO] Quitting...")
            break
        elif key == ord('c') or key == ord('C'):
            active_modes['currency'] = not active_modes['currency']
            print(f"[MODE] Currency detection: {active_modes['currency']}")
        elif key == ord('f') or key == ord('F'):
            active_modes['footpath'] = not active_modes['footpath']
            print(f"[MODE] Footpath detection: {active_modes['footpath']}")
        elif key == ord('o') or key == ord('O'):
            active_modes['object'] = not active_modes['object']
            print(f"[MODE] Object detection: {active_modes['object']}")
        elif key == ord('x') or key == ord('X'):
            active_modes['face'] = not active_modes['face']
            print(f"[MODE] Face detection: {active_modes['face']}")
        elif key == ord('r') or key == ord('R'):
            active_modes['ocr'] = not active_modes['ocr']
            print(f"[MODE] OCR detection: {active_modes['ocr']}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection stopped. Goodbye!")

if __name__ == "__main__":
    main()
