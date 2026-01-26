import csv
import numpy as np

csv_file = "Known_unknown_detection/face_eval_log.csv"

TP = FP = TN = FN = 0
conf_known = []
conf_unknown = []

with open(csv_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gt = row["ground_truth"]
        pred = row["predicted"]
        conf = float(row["confidence"])

        if gt == "Known":
            conf_known.append(conf)
            if pred != "Unknown":
                TP += 1
            else:
                FN += 1

        elif gt == "Unknown":
            conf_unknown.append(conf)
            if pred == "Unknown":
                TN += 1
            else:
                FP += 1

total = TP + TN + FP + FN

print("\n=== Face Recognition Evaluation Summary ===")
print(f"Total samples: {total}")
print(f"TP={TP}, FN={FN}, FP={FP}, TN={TN}")

# ---- Metrics ----
accuracy = (TP + TN) / total if total else 0
recall = TP / (TP + FN) if (TP + FN) else 0

print(f"\nRecognition Accuracy: {accuracy*100:.2f}%")
print(f"Recall (Known faces): {recall*100:.2f}%")

if FP + TN > 0:
    FAR = FP / (FP + TN)
    print(f"False Acceptance Rate (FAR): {FAR*100:.2f}%")
else:
    print("False Acceptance Rate (FAR): Not applicable (no unknown samples)")

if FN + TP > 0:
    FRR = FN / (FN + TP)
    print(f"False Rejection Rate (FRR): {FRR*100:.2f}%")

# ---- Confidence stats ----
print("\nConfidence Statistics:")
print(f"Mean confidence (Known): {np.mean(conf_known):.3f}")
print(f"Min confidence (Known):  {np.min(conf_known):.3f}")
print(f"Max confidence (Known):  {np.max(conf_known):.3f}")

if conf_unknown:
    print(f"Mean confidence (Unknown): {np.mean(conf_unknown):.3f}")
