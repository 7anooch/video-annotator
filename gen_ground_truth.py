import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import tkinter as tk
from tkinter import filedialog
import csv

from funcs import get_longest_common_substring

def load_annotations(csv_path):
    annotations = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        annotations = {row['frame']: row['label'] for _, row in df.iterrows()}
        print(f"Loaded annotations from {csv_path}")
    else:
        print(f"No annotation file found at {csv_path}")
    return annotations


def get_common_basename_prefix(csv_paths):
    basenames = [os.path.basename(path.split('.csv')[0]) for path in csv_paths]
    return get_longest_common_substring(basenames)

def get_csv_paths():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, nargs='*', help="Paths to the CSV files")
    args = parser.parse_args()

    if args.csv:
        return args.csv
    else:
        root = tk.Tk()
        root.withdraw()
        csv_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        return csv_paths

if __name__ == "__main__":
    csv_paths = get_csv_paths()
    if csv_paths:
        all_annotations = {}
        min_length = float('inf')

        for csv_path in csv_paths:
            annotations = load_annotations(csv_path)
            all_annotations[csv_path] = annotations
            min_length = min(min_length, len(annotations))

        max_frame = int(max(max(annotations.keys()) for 
                            annotations in all_annotations.values()))

        frame_label_counts = {frame: {} for frame in range(max_frame + 1)}

        for annotations in all_annotations.values():
            for frame, label in annotations.items():
                if frame not in frame_label_counts:
                    frame_label_counts[frame] = {}
                if label not in frame_label_counts[frame]:
                    frame_label_counts[frame][label] = 0
                frame_label_counts[frame][label] += 1

        # Determine the most frequent label for each frame
        ground_truth = {}
        for frame, label_counts in frame_label_counts.items():
            if label_counts:
                most_frequent_label = max(label_counts, key=label_counts.get)
                ground_truth[frame] = most_frequent_label

        common_prefix = get_common_basename_prefix(csv_paths).lstrip('_').replace(' ', '_')
        basedir = os.path.dirname(csv_paths[0])

        output_csv_path = os.path.join(basedir, common_prefix + '_ground_truth.csv')
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "label"])  # Write the header
            for frame, label in sorted(ground_truth.items()):
                writer.writerow([frame, label])
            print(f"Ground truth saved to {output_csv_path}")
    else:
        print("No CSV file selected.")