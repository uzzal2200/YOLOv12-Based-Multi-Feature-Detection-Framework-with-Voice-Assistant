import csv
import numpy as np
import Levenshtein

CSV_FILE = "OCR detection/ocr_runtime_log_normalized.csv"

cer_list = []
wer_list = []
conf_list = []

def char_error_rate(gt, pred):
    return Levenshtein.distance(gt, pred) / max(len(gt), 1)

def word_error_rate(gt, pred):
    gt_words = gt.split()
    pred_words = pred.split()
    return Levenshtein.distance(
        " ".join(gt_words),
        " ".join(pred_words)
    ) / max(len(gt_words), 1)

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    skipped = 0
    for row in reader:
        gt = row["ground_truth_text"].strip()
        pred = row["detected_text"].strip()

        if gt == "" or pred == "":
            skipped += 1
            continue

        try:
            conf = float(row["confidence"])
        except:
            skipped += 1
            continue

        cer = char_error_rate(gt, pred)
        wer = word_error_rate(gt, pred)

        cer_list.append(cer)
        wer_list.append(wer)
        conf_list.append(conf)

# ================= RESULTS =================
print("\n=== OCR Performance (After Normalization) ===")
print(f"Total Valid Samples: {len(cer_list)}")
print(f"Skipped Samples: {skipped}")

print(f"Average CER: {np.mean(cer_list):.3f}")
print(f"Average WER: {np.mean(wer_list):.3f}")
print(f"Word Recognition Accuracy: {(1 - np.mean(wer_list)) * 100:.2f}%")
print(f"Mean Confidence Score: {np.mean(conf_list):.3f}")
