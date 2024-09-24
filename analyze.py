import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import confusion_matrix
import itertools 
import csv

def load_annotations(csv_path, verbose=True):
    annotations = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        annotations = {row['frame']: row['label'] for _, row in df.iterrows()}
        if verbose:
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

def analyze_sequence(annotations):
    if not annotations:
        return []

    sorted_frames = sorted(annotations.keys())
    current_label = annotations[sorted_frames[0]]
    start_frame = sorted_frames[0]
    sequence_summary = []

    for frame in sorted_frames[1:]:
        label = annotations[frame]
        if label != current_label:
            duration = frame - start_frame
            sequence_summary.append((current_label, duration))
            current_label = label
            start_frame = frame

    duration = sorted_frames[-1] - start_frame + 1
    sequence_summary.append((current_label, duration))

    return sequence_summary

def compute_segment_stats(sequence):
    segment_lengths = {}
    segment_counts = {}
    
    for label, duration in sequence:
        if label not in segment_lengths:
            segment_lengths[label] = []
            segment_counts[label] = 0
        segment_lengths[label].append(duration)
        segment_counts[label] += 1
    
    average_lengths = {label: np.nanmean(durations) for label, durations in segment_lengths.items()}
    
    return average_lengths, segment_counts, segment_lengths

def pad_sequences(seq1, seq2, pad_value=-1, extra_pads=0):
    len1, len2 = len(seq1), len(seq2)
    max_len = max(len1, len2) + extra_pads
    seq1 = np.pad(seq1, (0, max_len - len1), constant_values=pad_value)
    seq2 = np.pad(seq2, (0, max_len - len2), constant_values=pad_value)
    return seq1, seq2

def needleman_wunsch(seq1, seq2, match_score=1, mismatch_penalty=-1, gap_penalty=-1):
    n = len(seq1)
    m = len(seq2)
    
    # Initialize the scoring matrix
    score_matrix = np.zeros((n + 1, m + 1))
    
    # Initialize the first row and column
    for i in range(1, n + 1):
        score_matrix[i][0] = i * gap_penalty
    for j in range(1, m + 1):
        score_matrix[0][j] = j * gap_penalty
    
    # Fill the scoring matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score_matrix[i - 1][j - 1] + (
                match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty)
            delete = score_matrix[i - 1][j] + gap_penalty
            insert = score_matrix[i][j - 1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)
    
    # Traceback to find the optimal alignment
    align1, align2 = [], []
    i, j = n, m
    while i > 0 and j > 0:
        score_current = score_matrix[i][j]
        score_diagonal = score_matrix[i - 1][j - 1]
        score_up = score_matrix[i][j - 1]
        score_left = score_matrix[i - 1][j]
        
        if score_current == score_diagonal + (
            match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty):
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1.append(seq1[i - 1])
            align2.append('-')
            i -= 1
        elif score_current == score_up + gap_penalty:
            align1.append('-')
            align2.append(seq2[j - 1])
            j -= 1
    
    # Add remaining gaps if necessary
    while i > 0:
        align1.append(seq1[i - 1])
        align2.append('-')
        i -= 1
    while j > 0:
        align1.append('-')
        align2.append(seq2[j - 1])
        j -= 1
    
    align1.reverse()
    align2.reverse()
    
    align1 = ''.join(map(str, align1))
    align2 = ''.join(map(str, align2))
    return align1, align2, score_matrix[n][m]

def merge_intervals(intervals):
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged_intervals = []
    
    for interval in sorted_intervals:
        if not merged_intervals or merged_intervals[-1][1] < interval[0]:
            merged_intervals.append(interval)
        else:
            # There is overlap, so merge the intervals
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], interval[1]))
    
    return merged_intervals

def compare_sequences(sequences):
    comparison_results = {}
    
    for (key1, data1), (key2, data2) in itertools.combinations(sequences.items(), 2):
        seq1, seq2 = data1['sequence'], data2['sequence']
        count1, count2 = data1['counts'], data2['counts']
        
        aligned_seq1, aligned_seq2, alignment_score = needleman_wunsch(seq1, seq2)
        
        if len(aligned_seq1) != len(aligned_seq2):
            print(f"Aligned sequences have different lengths: {len(aligned_seq1)} and {len(aligned_seq2)}")
            continue

        match_score = (np.array(list(aligned_seq1)) == np.array(list(
            aligned_seq2))).sum() / len(aligned_seq1)
        
        missing_indices1 = [i for i, x in enumerate(aligned_seq1) if x == '-']
        missing_indices2 = [i for i, x in enumerate(aligned_seq2) if x == '-']
        mismatch_indices = [i for i, (x, y) in enumerate(
            zip(aligned_seq1, aligned_seq2)) if x != y and x != '-' and y != '-']

        original_indices11 = map_indices_to_original(count1, missing_indices1)
        original_indices21 = map_indices_to_original(count2, missing_indices1)
        original_indices12 = map_indices_to_original(count1, missing_indices2)
        original_indices22 = map_indices_to_original(count2, missing_indices2)
        all_missing = original_indices11 + original_indices12 + original_indices21 + original_indices22
        all_missing = merge_intervals(all_missing)

        original_mismatch_indices1 = map_indices_to_original(count1, mismatch_indices)
        original_mismatch_indices2 = map_indices_to_original(count2, mismatch_indices)
        all_mismatch = original_mismatch_indices1 + original_mismatch_indices2
        all_mismatch = merge_intervals(all_mismatch)
        
        # Store results
        comparison_results[(key1, key2)] = {
            'best_match': match_score,
            'best_config': (aligned_seq1, aligned_seq2),
            'alignment_score': alignment_score,
            'missing_indices': all_missing,
            'missmatch_indices': all_mismatch
        }
    
    return comparison_results

def naive_comparison(sequences):
    comparison_results = {}
    
    for (key1, data1), (key2, data2) in itertools.combinations(sequences.items(), 2):
        seq1, seq2 = pad_sequences(data1['sequence'], data2['sequence'])
        comparison = (seq1 == seq2).sum() / len(seq1)
        comparison_results[(key1, key2)] = comparison
    
    return comparison_results

def map_indices_to_original(sequence, missing_indices):
    original_indices = []
    for idx in missing_indices:
        if idx < len(sequence):
            counts = np.sum(sequence[:idx])
            counts_next = np.sum(sequence[:idx+1])
            original_indices.append((counts, counts_next))
    return original_indices

def compute_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) \
        if (precision + recall) > 0 else 0

def compute_confusion_matrix(ground_truth, annotations, labels=[0, 1, 2]):
    y_true = [ground_truth[frame] for frame in ground_truth]
    y_pred = [annotations.get(frame, -1) for frame in ground_truth]  # Use -1 for missing frames
    return confusion_matrix(y_true, y_pred, labels=labels)

def print_confusion_matrix(conf_matrix, labels):
    df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print(df_cm)

def compute_accuracy(ground_truth, annotations):
    total_frames = len(ground_truth)
    correct_frames = sum(1 for frame, gt_label in ground_truth.items() if annotations.get(frame) == gt_label)
    return correct_frames / total_frames if total_frames > 0 else 0

def calculate_mismatches(ground_truth, annotations):
    total_frames = len(ground_truth)
    mismatched_frames = sum(1 for frame, gt_label in ground_truth.items()
                             if annotations.get(frame) != gt_label)
    mismatch_percentage = mismatched_frames / total_frames \
        if total_frames > 0 else 0
    return mismatched_frames, mismatch_percentage

def generate_mismatch_annotations(ground_truth, annotations):
    mismatch_annotations = []
    for frame, gt_label in ground_truth.items():
        pred_label = annotations.get(frame, -1)  # Use -1 for missing frames
        mismatch_label = 1 if gt_label == pred_label else 0
        mismatch_annotations.append({'frame': frame, 'label': mismatch_label})
    return mismatch_annotations

def save_mismatch_annotations(mismatch_annotations, output_path):
    df = pd.DataFrame(mismatch_annotations)
    df.to_csv(output_path, index=False)

def plot_segment_lengths(seg_lengths, label_map):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Plot histograms for each label
    for i, (label_index, label_name) in enumerate(label_map.items()):
        for key, label_durations in seg_lengths.items():
            if label_index in label_durations:
                durations = label_durations[label_index]
                axes[i].hist(durations, bins=20, alpha=0.5, label=f"{key}")
        axes[i].set_title(f'Histogram of segment lengths for {label_name}')
        axes[i].set_xlabel('Segment Length')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

    # Remove the empty subplot (bottom right)
    fig.delaxes(axes[3])

    plt.tight_layout()
    plt.show()

def main():
    csv_paths = get_csv_paths()
    if not csv_paths:
        return

    ground_truth_path = None
    other_csv_paths = []

    for csv_path in csv_paths:
        if 'ground_truth.csv' in os.path.basename(csv_path):
            ground_truth_path = csv_path
        else:
            other_csv_paths.append(csv_path)

    ground_truth = load_annotations(ground_truth_path) if ground_truth_path else None

    other_annotations = {}
    for csv_path in other_csv_paths:
        annotations = load_annotations(csv_path)
        other_annotations[csv_path] = annotations

    all_annotations = {}
    for csv_path in csv_paths:
        annotations = load_annotations(csv_path, verbose=False)
        all_annotations[csv_path] = annotations

    label_map = {0:'stop', 1:'run', 2:'turn'}
    sequences = {}
    seg_lengths = {}

    for csv_path, annotations in all_annotations.items():
        print(f"Analyzing {os.path.basename(csv_path).split('.csv')[0]}:")
        sequenced = analyze_sequence(annotations)
        sequence  = np.array([int(frame[0]) for frame in sequenced])
        seq_counts = np.array([int(frame[1]) for frame in sequenced])
        key = os.path.basename(csv_path).split('.csv')[0]
        lengths, counts, segment_lengths = compute_segment_stats(sequenced)
        seg_lengths[key] = {label: np.array(durations) for label, durations in segment_lengths.items()}
        sequences[key] = {'sequence': sequence, 'counts': seq_counts,
                            'segments': sum(counts.values())}
        for label in label_map:
            if label in lengths:
                print(f"Average segment length for {label_map[label]}: {lengths[label]:.2f} frames")
                print(f"           Total segments: {counts[label]}")
        print("\n")

    for csv_path, annotations in other_annotations.items():
        if ground_truth:
            print(f"Comparing {os.path.basename(csv_path).split('.csv')[0]} to the ground truth\n")
            mismatch_frames, mismatch_percent = calculate_mismatches(ground_truth,
                                                                    annotations)
            precision_recall = compute_precision_recall(ground_truth, annotations)
            accuracy = compute_accuracy(ground_truth, annotations)
            conf_matrix = compute_confusion_matrix(ground_truth, annotations)
            labels = [label_map[i] for i in sorted(label_map.keys())]

            print(f"Total mismatched frames: {mismatch_frames}")
            print(f"Mismatch percentage: {mismatch_percent:.2%}")
            print(f"Accuracy: {accuracy:.2%}\n")

            print("(Rows: Ground Truth labels, Columns: prediction/annotation)")
            print("Confusion Matrix:")
            print_confusion_matrix(conf_matrix, labels)
            print("\n")

            for i in label_map:
                precision = precision_recall[i]['precision']
                recall = precision_recall[i]['recall']
                f1_score = compute_f1_score(precision, recall)
                print(f"Label: {label_map[i]}")
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F1 Score: {f1_score:.2f}\n")

            mismatch_annotations = generate_mismatch_annotations(ground_truth, annotations)
            output_path = os.path.join(os.path.dirname(csv_path), 
                                    os.path.basename(csv_path).split('.csv')[0] + '_mismatch.csv')
            save_mismatch_annotations(mismatch_annotations, output_path)
            print(f"Mismatch annotations saved to {output_path}\n")

    plot_segment_lengths(seg_lengths, label_map)   
    comparison_results = compare_sequences(sequences)
    for (key1, key2), result in comparison_results.items():
        aggregate_results = merge_intervals(result['missing_indices'] + result['missmatch_indices'])
        print(f"Comparison between {key1} and {key2}:")
        print(f"  Best Match Score: {result['best_match']:.2f}")
        print(f"  Alignment Score: {result['alignment_score']:.2f}\n")

        print(f"  Annotation sequences:")
        print(f" \t\t {result['best_config'][0]}")
        print(f" \t\t {result['best_config'][1]}\n")
    
        print(f"  Potential missed annotations in frames:\n{result['missing_indices']}")
        print(f"  Potential incorrect annotations in frames:\n{result['missmatch_indices']}\n")

        print(f"  Indices of concern: \n{aggregate_results}\n\n")
        

if __name__ == "__main__":
    main()