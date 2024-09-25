import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
from tqdm import tqdm

from funcs import get_common_substring, format_frames_and_ranges

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
            max_count = max(label_counts.values())
            most_frequent_labels = [label for label, count in 
                                    label_counts.items() if count == max_count]
            
            if len(most_frequent_labels) == 1:
                ground_truth[frame] = most_frequent_labels[0]
            else:
                # Handle the case where there is no clear most frequent label
                ground_truth[frame] = default_label
        else:
            ground_truth[frame] = default_label
    return ground_truth

def save_ground_truth(ground_truth, csv_paths, gt_suffix='_ground_truth.csv'):
    common_prefix = get_common_basename_prefix(csv_paths).lstrip('_').replace(' ', '_')
    basedir = os.path.dirname(csv_paths[0])
    output_csv_path = os.path.join(basedir, common_prefix + gt_suffix)

    ground_truth_list = [{'frame': frame, 'label': int(label)} 
                         for frame, label in sorted(ground_truth.items())]
    df = pd.DataFrame(ground_truth_list)
    df.to_csv(output_csv_path, index=False)
    print(f"Ground truth saved to {output_csv_path}")

def calculate_confidence(frame_label_counts):
    confidence_levels = {}
    frames_with_confidence_3 = []
    
    for frame, label_counts in frame_label_counts.items():
        total_annotations = sum(label_counts.values())
        if total_annotations == 0:
            confidence_levels[frame] = 3
        elif len(label_counts) == 1:
            confidence_levels[frame] = 5
        elif len(label_counts) == 2:
            confidence_levels[frame] = 4
        else:
            confidence_levels[frame] = 3

        if confidence_levels[frame] == 3:
            frames_with_confidence_3.append(frame)

    return confidence_levels, frames_with_confidence_3

def generate_confidence_annotations(confidence_levels):
    confidence_annotations = []
    for frame, confidence in confidence_levels.items():
        confidence_annotations.append({'frame': frame, 'label': int(confidence)})
    return confidence_annotations

def compute_confidence_stats(confidence_levels):
    total_frames = len(confidence_levels)
    if total_frames == 0:
        return {"high": 0, "medium": 0, "low": 0}

    high_confidence_count = sum(1 for confidence in 
                                confidence_levels.values() if confidence == 5)
    medium_confidence_count = sum(1 for confidence in 
                                  confidence_levels.values() if confidence == 4)
    low_confidence_count = sum(1 for confidence in 
                               confidence_levels.values() if confidence == 3)

    high_confidence_percentage = high_confidence_count / total_frames
    medium_confidence_percentage = medium_confidence_count / total_frames
    low_confidence_percentage = low_confidence_count / total_frames

    return {
        "high": high_confidence_percentage,
        "medium": medium_confidence_percentage,
        "low": low_confidence_percentage
    }

def save_confidence_annotations(confidence_annotations, csv_paths, 
                                confidence_suffix='_gt_confidence.csv'):
    common_prefix = get_common_basename_prefix(csv_paths).lstrip('_').replace(' ', '_')
    basedir = os.path.dirname(csv_paths[0])
    output_path = os.path.join(basedir, common_prefix + confidence_suffix)

    sorted_annotations = sorted(confidence_annotations, 
                                key=lambda x: x['frame'])
    df = pd.DataFrame(sorted_annotations)
    df.to_csv(output_path, index=False)
    print(f"Confidence annotations saved to {output_path}")

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

def dawid_skene(frame_label_counts, annotations, min_length, 
                max_iter=100, convergence_threshold=1e-10):
    probabilities = initialize_probabilities(frame_label_counts)
    previous_probabilities = defaultdict(float, probabilities)

    for iteration in range(max_iter):
        annotator_reliability = e_step(probabilities, annotations, min_length)
        probabilities = m_step(annotator_reliability, annotations, min_length)
        
        all_labels = set(label for frame_probs in probabilities.values()
                          for label in frame_probs)
        prob_array = np.array([[probabilities[frame].get(label, 0)
                                 for label in sorted(all_labels)] for frame
                                   in sorted(probabilities.keys())])
        prev_prob_array = np.array([[previous_probabilities[frame].get(label, 0)
                                      for label in sorted(all_labels)] for frame
                                        in sorted(previous_probabilities.keys())])
        
        if np.linalg.norm(prob_array - prev_prob_array) < convergence_threshold:
            break
        
        previous_probabilities = defaultdict(float, probabilities)
    else:
        distance_from_convergence = np.linalg.norm(prob_array - prev_prob_array)
        print("Max iterations reached without converging. "
              f"Distance from convergence: {distance_from_convergence:.6f}")
    
    return probabilities

def determine_ground_truth_dawid_skene(probabilities):
    ground_truth = {}
    for frame, label_probs in probabilities.items():
        ground_truth[frame] = max(label_probs, key=label_probs.get)
    return ground_truth

def calculate_confidence_from_probabilities(probabilities):
    confidence_levels = {}
    low_confidence = []
    for frame, prob in probabilities.items():
        max_prob = max(prob.values())
        if max_prob >= 0.9:
            confidence_levels[frame] = 5  # high confidence
        elif max_prob >= 0.7:
            confidence_levels[frame] = 4  # medium confidence
        else:
            confidence_levels[frame] = 3  # low confidence

        if max_prob < 0.60:
            low_confidence.append(frame)
    return confidence_levels, low_confidence

def plot_probability_histogram(probabilities):
    max_probs = [max(prob.values()) for prob in probabilities.values()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Maximum Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def main():
    csv_paths, use_ds = get_csv_paths()
    if csv_paths:
        all_annotations, min_length = get_all_annotations(csv_paths)
        frame_label_counts = count_frame_labels(all_annotations, min_length)

        gt_suffix = '_ground_truth_DS.csv' if use_ds else '_ground_truth.csv'
        confidence_suffix = '_gt_confidence_DS.csv'if use_ds else '_gt_confidence.csv'
        
        if use_ds:
            probabilities = dawid_skene(frame_label_counts, all_annotations, min_length)
            confidence_levels, no_agreement = calculate_confidence_from_probabilities(probabilities)
            ground_truth = determine_ground_truth_dawid_skene(probabilities)
            formatted_frames = format_frames_and_ranges(no_agreement)
            print(f"\nFound {len(no_agreement)} frames with less than 60% label probability:")
            print(f"{formatted_frames}\n")
        else:
            ground_truth = determine_ground_truth(frame_label_counts, default_label='-1')
            confidence_levels, no_agreement = calculate_confidence(frame_label_counts)
            formatted_frames = format_frames_and_ranges(no_agreement)
            print(f"\nFound {len(no_agreement)} frames with no annotation agreement:")
            print(f"{formatted_frames}\n")

        confidence_annotations = generate_confidence_annotations(confidence_levels)
        confidence_stats = compute_confidence_stats(confidence_levels)
        print(f"Ground truth confidence statistics:\n"
            f"High Confidence: {confidence_stats['high']:.2%}\n"
            f"Medium Confidence: {confidence_stats['medium']:.2%}\n"
            f"Low Confidence: {confidence_stats['low']:.2%}\n")
        save_ground_truth(ground_truth, csv_paths, gt_suffix=gt_suffix)
        save_confidence_annotations(confidence_annotations, csv_paths, 
                                    confidence_suffix=confidence_suffix)
        if use_ds:
            plot_probability_histogram(probabilities)
        
    else:
        print("No CSV file selected.")

if __name__ == "__main__":
    main()