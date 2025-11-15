import cv2
import os
import openpyxl
import time
import numpy as np

cam = cv2.VideoCapture(0)

# Known faces directory setup
known_faces_folder = "known_faces_folder"
known_face_images = []
known_names = []

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Excel, log setup
log_folder = "movement_logs"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

excel_file_path = "movement_log.xlsx"
if not os.path.exists(excel_file_path):
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = "Movement Logs"
    sheet.append(["Name", "Detection Time", "Date", "Confidence"])
    wb.save(excel_file_path)

# Load known faces (store actual images for simple comparison)
if os.path.exists(known_faces_folder):
    print(f"Checking folder: {known_faces_folder}")
    files = os.listdir(known_faces_folder)
    print(f"Files found: {files}")

    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(known_faces_folder, filename)
            print(f"Loading: {filepath}")
            image = cv2.imread(filepath)
            if image is not None:
                print(f"Image loaded successfully: {image.shape}")
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                print(f"Faces detected in {filename}: {len(faces)}")

                if len(faces) > 0:
                    # Take the first detected face
                    x, y, w, h = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    # Resize to standard size for comparison
                    face_roi = cv2.resize(face_roi, (100, 100))
                    known_face_images.append(face_roi)
                    known_names.append(os.path.splitext(filename)[0])
                    print(f"Added face: {os.path.splitext(filename)[0]}")
                else:
                    print(f"No face detected in {filename}")
            else:
                print(f"Failed to load image: {filepath}")
else:
    print(f"Folder not found: {known_faces_folder}")

print(f"Loaded {len(known_face_images)} known faces: {known_names}")
print("Instructions:")
print("- Press 'c' to capture new face")
print("- Press 'q' to quit")
print("- Detection will work automatically")

frame_count = 0
detection_threshold = 0.6  # Lower threshold for better detection sensitivity
capture_mode = False

def simple_face_match(face1, face2, threshold=0.7):
    """Simple face matching using template matching"""
    try:
        # Ensure both faces are the same size
        face1 = cv2.resize(face1, (100, 100))
        face2 = cv2.resize(face2, (100, 100))

        # Template matching
        result = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        return max_val > threshold
    except:
        return False

def capture_new_face(frame, face_roi, gray_roi):
    """Capture and save a new face"""
    name = input("\nEnter name for this face: ").strip()
    if name:
        # Save the face image
        filename = f"{name}.jpg"
        filepath = os.path.join(known_faces_folder, filename)

        # Save the original face region from frame
        y1, y2, x1, x2 = face_roi
        face_image = frame[y1:y2, x1:x2]
        cv2.imwrite(filepath, face_image)

        # Add to known faces list
        face_roi_resized = cv2.resize(gray_roi, (100, 100))
        known_face_images.append(face_roi_resized)
        known_names.append(name)

        print(f"Face saved as '{name}' in {filepath}")
        print(f"Now {len(known_face_images)} known faces loaded")
        return True
    return False

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue  # Process every 3rd frame for efficiency

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in current frame
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray_frame[y:y+h, x:x+w]
        face_roi_resized = cv2.resize(face_roi, (100, 100))

        # Try to match with known faces
        name = "Unknown"
        best_match = 0
        confidence = 0

        for i, known_face in enumerate(known_face_images):
            # Calculate similarity using template matching
            result = cv2.matchTemplate(face_roi_resized, known_face, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > detection_threshold and max_val > best_match:
                best_match = max_val
                confidence = max_val
                name = known_names[i]

        # Draw rectangle and label
        if name != "Unknown":
            color = (0, 255, 0)  # Green for known faces
            status = "KNOWN"
            label = f"{status}: {name} ({confidence:.2f})"
        else:
            color = (0, 0, 255)  # Red for unknown faces
            status = "UNKNOWN"
            label = f"{status}"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Add background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y-35), (x + text_size[0] + 10, y), color, -1)
        cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Only log if confidence is high enough
        if confidence > 0.5 or name == "Unknown":
            # Log detection
            image_path = os.path.join(log_folder, f"{name}_{timestamp.replace(':', '-')}.jpg")
            cv2.imwrite(image_path, frame)

            # Save to Excel with error handling
            try:
                wb = openpyxl.load_workbook(excel_file_path)
                sheet = wb.active
                sheet.append([name, timestamp, time.strftime("%Y-%m-%d"), f"{confidence:.2f}"])
                wb.save(excel_file_path)
            except Exception as e:
                print(f"Excel error: {e}")
                # Create new file if corrupted
                wb = openpyxl.Workbook()
                sheet = wb.active
                sheet.title = "Movement Logs"
                sheet.append(["Name", "Detection Time", "Date", "Confidence"])
                sheet.append([name, timestamp, time.strftime("%Y-%m-%d"), f"{confidence:.2f}"])
                wb.save(excel_file_path)
                print("Created new Excel file")

            print(f"Detected: {name} (confidence: {confidence:.2f}) at {timestamp}")

    cv2.imshow("Face Detection - Simple Version", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("Face detection stopped.")