import csv
import os
from collections import Counter

# CONFIGURATION ─────────────────────────────────────────────────────────────
CSV_PATH = 'keypoint.csv'
LABEL_PATH = 'keypoint_classifier_label.csv'

def verify_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return
    if not os.path.exists(LABEL_PATH):
        print(f"Error: {LABEL_PATH} not found.")
        return

    # 1. Read labels
    labels = []
    with open(LABEL_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        labels = [row[0] for row in reader if row]

    num_classes = len(labels)
    print(f"Found {num_classes} labels in '{LABEL_PATH}'.")

    # 2. Count occurrences in CSV
    counts = Counter()
    total_rows = 0
    try:
        with open(CSV_PATH, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                total_rows += 1
                try:
                    label_id = int(row[0])
                    counts[label_id] += 1
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Print Summary Table
    print("\n" + "="*60)
    print(f"{'ID':<4} | {'Label Name':<25} | {'Row Count':<10}")
    print("-" * 60)

    missing_labels = []
    for i in range(num_classes):
        label_name = labels[i]
        count = counts[i]
        
        status_marker = ""
        if count == 0:
            status_marker = "⚠️ MISSING"
            missing_labels.append(label_name)
        
        print(f"{i:<4} | {label_name[:25]:<25} | {count:<10} {status_marker}")

    print("="*60)
    print(f"Total Rows: {total_rows}")
    
    if missing_labels:
        print(f"\n❌ ALERT: {len(missing_labels)} labels have NO data in the CSV:")
        for name in missing_labels:
            print(f"  - {name}")
    else:
        print("\n✅ SUCCESS: All labels have at least some data in the CSV.")

if __name__ == '__main__':
    # Run from model/urdu_alphabet/
    verify_dataset()
