import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import tkinter as tk
from tkinter import filedialog
import csv

def load_annotations(csv_path):
    annotations = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        annotations = {row['frame']: row['label'] for _, row in df.iterrows()}
        print(f"Loaded annotations from {csv_path}")
    else:
        print(f"No annotation file found at {csv_path}")
    return annotations

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
    
def compute_precision_recall(ground_truth, annotations, labels=[0, 1, 2]):
    precision_recall = {label: {'true_positive': 0, 
                                'false_positive': 0, 'false_negative': 0} for label in labels}

    for frame, gt_label in ground_truth.items():
        pred_label = annotations.get(frame)
        for label in labels:
            if gt_label == label:
                if pred_label == label:
                    precision_recall[label]['true_positive'] += 1
                else:
                    precision_recall[label]['false_negative'] += 1
            elif pred_label == label:
                precision_recall[label]['false_positive'] += 1

    for label in labels:
        tp = precision_recall[label]['true_positive']
        fp = precision_recall[label]['false_positive']
        fn = precision_recall[label]['false_negative']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_recall[label]['precision'] = precision
        precision_recall[label]['recall'] = recall

    return precision_recall

def compute_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) \
        if (precision + recall) > 0 else 0

def calculate_mismatches(ground_truth, annotations):
    total_frames = len(ground_truth)
    mismatched_frames = sum(1 for frame, gt_label in ground_truth.items()
                             if annotations.get(frame) != gt_label)
    mismatch_percentage = mismatched_frames / total_frames \
        if total_frames > 0 else 0
    return mismatched_frames, mismatch_percentage

if __name__ == "__main__":
    csv_paths = get_csv_paths()
    if csv_paths:
        ground_truth_path = None
        other_csv_paths = []

        for csv_path in csv_paths:
            if 'ground_truth.csv' in os.path.basename(csv_path):
                ground_truth_path = csv_path
            else:
                other_csv_paths.append(csv_path)

        ground_truth = load_annotations(ground_truth_path)
        other_annotations = {}
        for csv_path in other_csv_paths:
            annotations = load_annotations(csv_path)
            other_annotations[csv_path] = annotations

        label_map = {0:'stop', 1:'run', 2:'turn'}

        for csv_path, annotations in other_annotations.items():
            print(f"Comparing {os.path.basename(csv_path).split('.csv')[0]} to the ground_truth\n")
            mismatch_frames, mismatch_percent = calculate_mismatches(ground_truth,
                                                                      annotations)
            precision_recall = compute_precision_recall(ground_truth, annotations)

            print(f"Total mismatched frames: {mismatch_frames}")
            print(f"Mismatch percentage: {mismatch_percent:.2%}\n")

            for i in label_map:
                precision = precision_recall[i]['precision']
                recall = precision_recall[i]['recall']
                f1_score = compute_f1_score(precision, recall)
                print(f"Label: {label_map[i]}")
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F1 Score: {f1_score:.2f}\n")

    else:
        print("No CSV file selected.")