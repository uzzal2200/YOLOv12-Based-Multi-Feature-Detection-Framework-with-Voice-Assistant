# import os

# # Allow duplicate OpenMP runtimes to avoid libiomp5md.dll conflicts.
# if "KMP_DUPLICATE_LIB_OK" not in os.environ:
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import cv2
# import easyocr
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont

# def load_bengali_font(font_size: int = 26) -> ImageFont.FreeTypeFont:
#     """Try to load a Bengali-capable TrueType font.

#     Priority: project font file -> common Windows fonts.
#     """
#     candidate_paths = [
#         os.path.join(os.path.dirname(__file__), 'NotoSansBengali-Regular.ttf'),
#         os.path.join(os.path.dirname(__file__), 'fonts', 'NotoSansBengali-Regular.ttf'),
#         # Common Windows fonts that support Bengali
#         r"C:\\Windows\\Fonts\\Nirmala.ttf",        # Nirmala UI
#         r"C:\\Windows\\Fonts\\Vrinda.ttf",         # Vrinda
#     ]

#     # Also search Windows Fonts directory for likely Bengali fonts
#     windows_fonts_dir = r"C:\\Windows\\Fonts"
#     if os.path.isdir(windows_fonts_dir):
#         try:
#             for fname in os.listdir(windows_fonts_dir):
#                 lower = fname.lower()
#                 if any(key in lower for key in ['bengali', 'nirmala', 'vrinda', 'bangla', 'notosansbengali']):
#                     candidate_paths.append(os.path.join(windows_fonts_dir, fname))
#         except Exception:
#             pass

#     for path in candidate_paths:
#         if os.path.exists(path):
#             try:
#                 font = ImageFont.truetype(path, font_size)
#                 print(f"[OCR] Using Bengali font: {path}")
#                 return font
#             except Exception:
#                 continue

#     # Fallback to a default PIL bitmap font (won’t shape Bengali correctly,
#     # but prevents crashes if no font is available).
#     print("[OCR] WARNING: No Bengali font found. Falling back to default font.")
#     return ImageFont.load_default()


# def draw_detections(pil_img: Image.Image, results, font: ImageFont.FreeTypeFont) -> Image.Image:
#     """Render bounding boxes and text onto a PIL image."""
#     draw = ImageDraw.Draw(pil_img)
#     for (bbox, text, prob) in results:
#         top_left, _, bottom_right, _ = bbox
#         top_left_xy = (int(top_left[0]), int(top_left[1]))
#         bottom_right_xy = (int(bottom_right[0]), int(bottom_right[1]))

#         draw.rectangle([top_left_xy, bottom_right_xy], outline=(255, 0, 0), width=2)
#         text_pos = (top_left_xy[0], max(0, top_left_xy[1] - 28))
#         draw.text(text_pos, text, font=font, fill=(255, 0, 0))

#     try:
#         draw.text((10, 10), "Font is OK", font=font, fill=(255, 255, 0))
#     except Exception:
#         pass
#     return pil_img


# def main() -> None:
#     reader = easyocr.Reader(['en', 'bn'])
#     font = load_bengali_font(28)

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("[OCR] ERROR: Camera not accessible.")
#         return

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("[OCR] WARNING: No frame captured; exiting.")
#                 break

#             small_frame = cv2.resize(frame, (640, 480))
#             results = reader.readtext(small_frame)

#             rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#             pil_img = Image.fromarray(rgb_frame)
#             pil_img = draw_detections(pil_img, results, font)

#             small_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
#             cv2.imshow("Real-Time Bengali OCR", small_frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()




import os
import time
import csv
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ===============================
# Fix OpenMP issue (Windows)
# ===============================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===============================
# Bengali Font Loader
# ===============================
def load_bengali_font(font_size=28):
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "NotoSansBengali-Regular.ttf"),
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\Vrinda.ttf",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                print(f"[OCR] Using font: {path}")
                return ImageFont.truetype(path, font_size)
            except:
                pass
    print("[OCR] WARNING: Bengali font not found, using default.")
    return ImageFont.load_default()

# ===============================
# Draw OCR Results
# ===============================
def draw_detections(pil_img, results, font):
    draw = ImageDraw.Draw(pil_img)
    for (bbox, text, prob) in results:
        tl, _, br, _ = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        draw.rectangle([tl, br], outline=(255, 0, 0), width=2)
        draw.text((tl[0], max(0, tl[1] - 30)), text, font=font, fill=(255, 0, 0))
    return pil_img

# ===============================
# CSV Logger
# ===============================
CSV_FILE = "ocr_runtime_log.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "detected_text", "confidence", "ocr_latency_ms"])

# ===============================
# MAIN
# ===============================
def main():
    reader = easyocr.Reader(['en', 'bn'], gpu=False)
    font = load_bengali_font(28)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[OCR] Camera not accessible")
        return

    print("\n--- Real-Time Bangla–English OCR + Performance Logging ---\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))

            # ---------------------------
            # OCR Inference + Latency
            # ---------------------------
            start_time = time.time()
            results = reader.readtext(frame)
            ocr_latency = (time.time() - start_time) * 1000  # ms

            # ---------------------------
            # Draw Results
            # ---------------------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img = draw_detections(pil_img, results, font)
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # ---------------------------
            # Log Results
            # ---------------------------
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            for (_, text, prob) in results:
                with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp,
                        text,
                        round(prob, 3),
                        round(ocr_latency, 2)
                    ])

                print(
                    f"OCR: '{text}' | "
                    f"Conf: {prob:.2f} | "
                    f"Latency: {ocr_latency:.2f} ms"
                )

            # ---------------------------
            # Display
            # ---------------------------
            cv2.putText(
                frame,
                f"OCR Latency: {ocr_latency:.1f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Bangla–English OCR (Real-Time)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[OCR] Stopped. Logs saved to:", CSV_FILE)

# ===============================
if __name__ == "__main__":
    main()
