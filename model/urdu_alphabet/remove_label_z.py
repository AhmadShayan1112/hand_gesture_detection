import csv
import os
import shutil
from datetime import datetime

# CONFIGURATION ─────────────────────────────────────────────────────────────
# Path to the data file
CSV_PATH = 'keypoint.csv'

# Label ID to remove.
# In the 38-class system (app2.py), key 'z' maps to ID: 34
LABEL_ID_TO_REMOVE = 34

def remove_label_data(csv_path, label_id):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 1. Create a safety backup first
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{csv_path}.backup_{timestamp}"
    shutil.copy2(csv_path, backup_path)
    print(f"Safety backup created: {backup_path}")

    # 2. Read the data and filter out the target label_id
    rows_to_keep = []
    removed_count = 0
    total_count = 0

    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                
                total_count += 1
                try:
                    # The first column is the label ID
                    current_id = int(row[0])
                    if current_id == label_id:
                        removed_count += 1
                    else:
                        rows_to_keep.append(row)
                except (ValueError, IndexError):
                    # Keep row if it's malformed or header-like (though it shouldn't be here)
                    rows_to_keep.append(row)

        if removed_count == 0:
            print(f"No data found for label ID {label_id}. No changes made.")
            return

        # 3. Write the filtered data back to the original CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_keep)

        print(f"Successfully cleaned '{csv_path}'.")
        print(f"Total rows processed: {total_count}")
        print(f"Rows removed (Label ID {label_id}): {removed_count}")
        print(f"Remaining rows: {len(rows_to_keep)}")
        print("\nDO NOT delete the backup file until you have verified the cleanup!")

    except Exception as e:
        print(f"An error occurred during cleaning: {e}")
        print("Restoring from backup...")
        shutil.copy2(backup_path, csv_path)
        print("Restoration complete.")

if __name__ == '__main__':
    # Since we are in model/urdu_alphabet/, the CSV should be right here.
    # We use local path 'keypoint.csv'
    remove_label_data(CSV_PATH, LABEL_ID_TO_REMOVE)
