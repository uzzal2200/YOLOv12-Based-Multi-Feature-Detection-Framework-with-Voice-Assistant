import cv2
import time
import os
from ultralytics import YOLO
from gtts import gTTS
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from pydub.playback import play

# Load the trained model
model = YOLO('save model/best.pt')

# Bengali labels
bn_labels = {
    'car': 'এইখানে দরজা আছে  তুমি বের হতে পারো বা প্রবেশ করতে পারো',
    'chairr': 'এটা আলমারি দরজা ',
    'Door': 'এটা ফ্রিজের দরজা',
    'Man': 'এটা মাইক্রোওয়েভ দরজা',
    'Road': 'এটা ড্রয়ার তুমি চাইলে কিছু রাখতে পারো ',
    'Stair': 'এটা জানলা ',
    'Table': 'এটা ডিশওয়াশার দরজা',
    'Tree': 'এটা ওভেন দরজা',
    'wall': 'এদিক দিয়ে গ্যারেজের দরজা '
}

# Cooldown mechanism
cooldown = 5  # seconds
last_spoken_time = {}

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run YOLO detection
        results = model(frame, verbose=False)

        # Annotate and show frame
        annotated_frame = results[0].plot()
        cv2.imshow("Real-time Object Detection", annotated_frame)

        current_time = time.time()

        # Voice output for each object
        for box in results[0].boxes.cls:
            label = model.names[int(box)]

            if label in bn_labels:
                if label not in last_spoken_time or current_time - last_spoken_time[label] > cooldown:
                    text = bn_labels[label]
                    tts = gTTS(text=text, lang='bn')

                    try:
                        with NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                            tts.save(temp_audio.name)
                            print(f"Temporary audio file created at: {temp_audio.name}")

                            # Use pydub to play the audio
                            sound = AudioSegment.from_mp3(temp_audio.name)
                            play(sound)
                            last_spoken_time[label] = current_time
                    except Exception as e:
                        print(f"Error playing sound: {e}")

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
