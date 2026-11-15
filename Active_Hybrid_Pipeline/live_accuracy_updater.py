"""
Live Accuracy Report Auto-Updater
Monitors batch_predictions.json and continuously updates ACCURACY_REPORT.md
as each new video completes during bulk testing.
"""

import os
import time
import json
import importlib
import calculate_accuracy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "test_results", "batch_predictions.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results")

def watch_and_update():
    last_count = 0
    print("[Watcher] Live ACCURACY_REPORT auto-updater active...")
    while True:
        try:
            if os.path.exists(JSON_PATH):
                with open(JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if len(data) != last_count and len(data) > 0:
                    importlib.reload(calculate_accuracy)
                    calculate_accuracy.generate_report(JSON_PATH, OUTPUT_DIR)
                    last_count = len(data)
                    print(f"[Watcher] Updated ACCURACY_REPORT.md ({last_count} videos processed).", flush=True)
        except Exception as e:
            print(f"[Watcher Error] {e}", flush=True)
        time.sleep(15)

if __name__ == "__main__":
    watch_and_update()
