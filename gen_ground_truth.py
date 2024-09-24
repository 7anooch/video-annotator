import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
import csv

from funcs import get_common_substring

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
    return get_common_substring(basenames)

def get_csv_paths():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, nargs='*', help="Paths to the CSV files")
    parser.add_argument('--ds', action='store_true')
    args = parser.parse_args()

    if args.csv:
        return args.csv
    else:
        root = tk.Tk()
        root.withdraw()
        csv_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        return csv_paths, args.ds
    
def get_all_annotations(csv_paths):
    all_annotations = {}
    min_length = float('inf')

    for csv_path in csv_paths:
        annotations = load_annotations(csv_path)
        all_annotations[csv_path] = annotations
        min_length = min(min_length, len(annotations))

    return all_annotations, min_length

def count_frame_labels(all_annotations, min_length):
    frame_label_counts = {frame: {} for frame in range(min_length)}

    for annotations in all_annotations.values():
        for frame, label in annotations.items():
            if frame >= min_length:
                continue
            if frame not in frame_label_counts:
                frame_label_counts[frame] = {}
            if label not in frame_label_counts[frame]:
                frame_label_counts[frame][label] = 0
            frame_label_counts[frame][label] += 1

    return frame_label_counts

def determine_ground_truth(frame_label_counts, default_label=None):
    ground_truth = {}
    for frame, label_counts in frame_label_counts.items():
        if label_counts:
            most_frequent_label = max(label_counts, key=label_counts.get)
            ground_truth[frame] = most_frequent_label
        else:
            ground_truth[frame] = default_label
    return ground_truth

def save_ground_truth(ground_truth, csv_paths, gt_suffix = '_ground_truth.csv'):
    common_prefix = get_common_basename_prefix(csv_paths).lstrip('_').replace(' ', '_')
    basedir = os.path.dirname(csv_paths[0])
    output_csv_path = os.path.join(basedir, common_prefix + gt_suffix)

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "label"])  # Write the header
        for frame, label in sorted(ground_truth.items()):
            writer.writerow([frame, label])
        print(f"Ground truth saved to {output_csv_path}")

def initialize_probabilities(frame_label_counts):
    probabilities = {}
    for frame, label_counts in frame_label_counts.items():
        total = sum(label_counts.values())
        probabilities[frame] = {label: count / total for label, 
                                count in label_counts.items()}
    return probabilities

def e_step(probabilities, annotations, min_length):
    annotator_reliability = defaultdict(lambda: defaultdict(float))
    for frame in range(min_length):
        for annotator, labels in annotations.items():
            if frame in labels:
                label = labels[frame]
                for true_label in probabilities[frame]:
                    annotator_reliability[annotator][true_label] += \
                        probabilities[frame][true_label] if label == true_label else 0
    return annotator_reliability

def m_step(annotator_reliability, annotations, min_length):
    probabilities = defaultdict(lambda: defaultdict(float))
    for frame in range(min_length):
        for annotator, labels in annotations.items():
            if frame in labels:
                label = labels[frame]
                for true_label in annotator_reliability[annotator]:
                    probabilities[frame][true_label] += annotator_reliability[
                        annotator][true_label] if label == true_label else 0
        total = sum(probabilities[frame].values())
        for true_label in probabilities[frame]:
            probabilities[frame][true_label] /= total
    return probabilities

def dawid_skene(frame_label_counts, annotations, min_length, max_iter=100):
    probabilities = initialize_probabilities(frame_label_counts)
    for _ in range(max_iter):
        annotator_reliability = e_step(probabilities, annotations, min_length)
        probabilities = m_step(annotator_reliability, annotations, min_length)
    return probabilities

def determine_ground_truth_dawid_skene(probabilities):
    ground_truth = {}
    for frame, label_probs in probabilities.items():
        ground_truth[frame] = max(label_probs, key=label_probs.get)
    return ground_truth

def main():
    csv_paths, use_ds = get_csv_paths()
    if csv_paths:
        all_annotations, min_length = get_all_annotations(csv_paths)
        frame_label_counts = count_frame_labels(all_annotations, min_length)
        
        if use_ds:
            probabilities = dawid_skene(frame_label_counts, all_annotations, min_length)
            ground_truth = determine_ground_truth_dawid_skene(probabilities)
            save_ground_truth(ground_truth, csv_paths, gt_suffix='_ground_truth_DS.csv')
        else:
            ground_truth = determine_ground_truth(frame_label_counts, default_label='-1')
            save_ground_truth(ground_truth, csv_paths)
            
    else:
        print("No CSV file selected.")

if __name__ == "__main__":
    main()