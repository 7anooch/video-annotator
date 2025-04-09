import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import confusion_matrix
import itertools

def load_annotations(csv_path, verbose=True, col=2):
    """Load annotations from a CSV file.

    This function is designed to be extremely robust and handle various CSV formats.

    Args:
        csv_path (str): Path to the CSV file
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        col (int, optional): Column index for labels (0-based). Defaults to 2.

    Returns:
        dict: Dictionary of annotations with frame numbers as keys and labels as values
    """
    annotations = {}

    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"No annotation file found at {csv_path}")
        return annotations

    try:
        # First, try to read the entire CSV file to see what we're working with
        if verbose:
            print(f"Reading CSV file: {csv_path}")

        # Try different approaches to read the CSV file
        try:
            # Approach 1: Read the entire file
            df = pd.read_csv(csv_path, header=None)
        except Exception as e1:
            if verbose:
                print(f"Error reading CSV file with default settings: {str(e1)}")
            try:
                # Approach 2: Try with different separator
                df = pd.read_csv(csv_path, header=None, sep=None, engine='python')
            except Exception as e2:
                if verbose:
                    print(f"Error reading CSV file with flexible separator: {str(e2)}")
                try:
                    # Approach 3: Try with explicit separator
                    df = pd.read_csv(csv_path, header=None, sep=',')
                except Exception as e3:
                    if verbose:
                        print(f"Error reading CSV file with comma separator: {str(e3)}")
                    # Give up
                    print(f"Failed to read CSV file: {csv_path}")
                    return annotations

        # Check if we have enough columns
        num_columns = df.shape[1]
        if verbose:
            print(f"CSV file has {num_columns} columns")

        if num_columns < 1:
            print(f"CSV file has no columns")
            return annotations

        # Determine which columns to use
        frame_col = 0  # Always use the first column for frames

        # If we only have one column, use it for both frames and labels
        if num_columns == 1:
            if verbose:
                print(f"CSV file has only one column, using it for both frames and labels")
            label_col = 0
        else:
            # Otherwise, use the specified column for labels, but make sure it's valid
            label_col = min(col, num_columns - 1)
            if col != label_col and verbose:
                print(f"Warning: Requested label column {col} is out of bounds, using column {label_col} instead")

        if verbose:
            print(f"Using column {frame_col} for frames and column {label_col} for labels")

        # Create annotations dictionary
        # Convert frames and labels to integers and handle any errors
        for _, row in df.iterrows():
            try:
                # Convert frame to integer
                frame = int(float(row[frame_col]))  # Handle both integer and float frame numbers

                # Convert label to integer
                label = row[label_col]
                if isinstance(label, str):
                    try:
                        # Try to convert string to integer
                        label = int(float(label))
                    except (ValueError, TypeError):
                        # If conversion fails, keep the original label
                        if verbose:
                            print(f"Warning: Could not convert label '{label}' to integer, keeping as is")

                annotations[frame] = label
            except (ValueError, TypeError) as e:
                if verbose:
                    print(f"Error converting frame to integer: {row[frame_col]} - {str(e)}")

        if verbose:
            print(f"Loaded {len(annotations)} annotations from {csv_path}")

    except Exception as e:
        print(f"Unexpected error loading annotations from {csv_path}: {str(e)}")

    return annotations

def get_csv_paths(csv_paths=None):
    if csv_paths:
        # If CSV paths are provided as an argument, use them
        return csv_paths

    # Otherwise, check if they're provided as command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, nargs='*', help="Paths to the CSV files")
    parser.add_argument('--advanced', action='store_true', default=False,
                      help="Run advanced analyses")
    args, _ = parser.parse_known_args()

    if args.csv:
        return args.csv
    else:
        # If no CSV paths are provided, open a file dialog
        root = tk.Tk()
        root.withdraw()
        csv_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        return csv_paths

def select_ground_truth(csv_paths):
    # Handle the case where csv_paths is a single string
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    # Filter out CSV paths that don't exist
    csv_paths = [csv_path for csv_path in csv_paths if os.path.exists(csv_path)]

    # If no CSV paths exist, return None, []
    if not csv_paths:
        print("No valid CSV files found.")
        return None, []

    ground_truth_paths = [csv_path for csv_path in csv_paths if 'preds' not in os.path.dirname(csv_path)]
    other_csv_paths = [csv_path for csv_path in csv_paths if 'preds' in os.path.dirname(csv_path)]

    if len(ground_truth_paths) > 1:
        print("Multiple ground truth files found. Please select one:")
        for i, path in enumerate(ground_truth_paths, 1):
            print(f"{i}. {path}")
        print("0. None")

        while True:
            try:
                choice = int(input("Enter the number of the ground truth file to use (0 for none): "))
                if 0 <= choice <= len(ground_truth_paths):
                    break
                else:
                    print("Invalid choice. Please enter a number between 0 and", len(ground_truth_paths))
            except ValueError:
                print("Invalid input. Please enter a number.")

        if choice == 0:
            ground_truth_path = None
            other_csv_paths.extend(ground_truth_paths)
        else:
            ground_truth_path = ground_truth_paths[choice - 1]
            other_csv_paths.extend([path for i, path in enumerate(ground_truth_paths) if i != choice - 1])
    elif len(ground_truth_paths) == 1:
        ground_truth_path = ground_truth_paths[0]
    else:
        ground_truth_path = None

    return ground_truth_path, other_csv_paths

def compute_precision_recall(ground_truth, annotations, labels=[0, 1, 2, 3,4, 5, 6]):
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
    """Analyze a sequence of annotations and return a summary of label durations.

    Args:
        annotations (dict): Dictionary of annotations with frame numbers as keys and labels as values.
                           The labels can be simple values or tuples.

    Returns:
        list: List of tuples (label, duration) representing the sequence of labels and their durations
    """
    if not annotations:
        return []

    # Filter out any non-numeric keys (like 'frame' if the CSV has a header)
    numeric_annotations = {}
    for key, value in annotations.items():
        try:
            # Try to convert the key to an integer
            numeric_key = int(float(key))

            # Handle the case where the value is a tuple
            if isinstance(value, tuple):
                # Try to convert the first element of the tuple to an integer
                try:
                    if isinstance(value[0], str):
                        label_value = int(float(value[0]))
                    else:
                        label_value = int(value[0])
                    print(f"Found tuple value: {value}, converted to {label_value}")
                except (ValueError, TypeError):
                    # If conversion fails, use the original value
                    label_value = value[0]
                    print(f"Found tuple value: {value}, using {label_value} as the label (could not convert to integer)")
            else:
                # Try to convert the value to an integer if it's a string
                if isinstance(value, str):
                    try:
                        label_value = int(float(value))
                    except (ValueError, TypeError):
                        label_value = value
                else:
                    label_value = value

            numeric_annotations[numeric_key] = label_value
        except (ValueError, TypeError):
            # Skip non-numeric keys
            print(f"Skipping non-numeric key: {key}")

    if not numeric_annotations:
        print("No numeric annotations found. Check if your CSV file has a header.")
        return []

    sorted_frames = sorted(numeric_annotations.keys())
    current_label = numeric_annotations[sorted_frames[0]]
    start_frame = sorted_frames[0]
    sequence_summary = []

    for frame in sorted_frames[1:]:
        label = numeric_annotations[frame]
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

    score_matrix = np.zeros((n + 1, m + 1))

    for i in range(1, n + 1):
        score_matrix[i][0] = i * gap_penalty
    for j in range(1, m + 1):
        score_matrix[0][j] = j * gap_penalty

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score_matrix[i - 1][j - 1] + (
                match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty)
            delete = score_matrix[i - 1][j] + gap_penalty
            insert = score_matrix[i][j - 1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)

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
            'missing_indices': [[int(x), int(y)] for x,y in all_missing],
            'missmatch_indices': [[int(x), int(y)] for x,y in all_mismatch]
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

def compute_confusion_matrix(ground_truth, annotations, labels=[0, 1, 2, 3, 4, 5, 6]):
    # Convert labels to integers if they're strings
    y_true = []
    y_pred = []

    for frame in ground_truth:
        gt_label = ground_truth[frame]
        pred_label = annotations.get(frame, -1)  # Use -1 for missing frames

        # Convert labels to integers if they're strings
        if isinstance(gt_label, str):
            try:
                gt_label = int(float(gt_label))
            except (ValueError, TypeError):
                # If conversion fails, try to find the label in the labels list
                if gt_label in [str(label) for label in labels]:
                    gt_label = labels[[str(label) for label in labels].index(gt_label)]
                else:
                    # If all else fails, use -1
                    gt_label = -1

        if isinstance(pred_label, str):
            try:
                pred_label = int(float(pred_label))
            except (ValueError, TypeError):
                # If conversion fails, try to find the label in the labels list
                if pred_label in [str(label) for label in labels]:
                    pred_label = labels[[str(label) for label in labels].index(pred_label)]
                else:
                    # If all else fails, use -1
                    pred_label = -1

        y_true.append(gt_label)
        y_pred.append(pred_label)

    # Make sure all labels are integers
    int_labels = []
    for label in labels:
        if isinstance(label, str):
            try:
                int_labels.append(int(float(label)))
            except (ValueError, TypeError):
                # If conversion fails, use the index
                int_labels.append(labels.index(label))
        else:
            int_labels.append(label)

    # Uncomment for debugging
    # print(f"Debug: y_true first 5 elements: {y_true[:5]}")
    # print(f"Debug: y_pred first 5 elements: {y_pred[:5]}")
    # print(f"Debug: labels: {labels}")
    # print(f"Debug: int_labels: {int_labels}")

    return confusion_matrix(y_true, y_pred, labels=int_labels)

def print_confusion_matrix(conf_matrix, labels):
    df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print(df_cm)

def compute_accuracy(ground_truth, annotations):
    total_frames = len(ground_truth)
    correct_frames = sum(1 for frame, gt_label in ground_truth.items()
                         if annotations.get(frame) == gt_label)
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

def colorize_mismatches(seq1, seq2):
    colored_seq1 = []
    colored_seq2 = []
    for char1, char2 in zip(seq1, seq2):
        if char1 != char2 and char1 != '-' and char2 != '-':
            colored_seq1.append(f"\033[91m{char1}\033[0m")  # Red color for mismatches
            colored_seq2.append(f"\033[91m{char2}\033[0m")
        else:
            colored_seq1.append(char1)
            colored_seq2.append(char2)
    return ''.join(colored_seq1), ''.join(colored_seq2)

def plot_segment_lengths(seg_lengths, label_map):
    num_labels = len(label_map)
    num_rows = 2  # Use 2 columns instead of 3 to avoid empty middle columns
    num_cols = (num_labels + num_rows - 1) // num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows))
    axes = axes.flatten()



    for i, (label_index, label_name) in enumerate(label_map.items()):
        all_durations = []
        for label_durations in seg_lengths.values():
            for durations in label_durations.values():
                all_durations.extend(durations)

        # Check if there are any durations to plot
        if not all_durations:
            print("No segment durations to plot.")
            return

        global_min = min(all_durations)
        global_max = np.nanpercentile(all_durations, 95)
        bins = np.linspace(global_min, global_max, 26)

        for key, label_durations in seg_lengths.items():
            if label_index in label_durations:
                durations = label_durations[label_index]
                axes[i].hist(durations, bins=bins, alpha=0.5, label=f"{key}")
        axes[i].set_title(f'Histogram of segment lengths for {label_name}')
        axes[i].set_xlabel('Segment length [frames]')
        axes[i].set_ylabel('Frequency')

    for j in range(len(label_map), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')

    plt.tight_layout()
    plt.show()

def analysis(*paths, col=2, csv_paths=None, advanced=False):
    if not paths and not csv_paths:
        csv_paths = get_csv_paths()
        if not csv_paths:
            return
    elif paths:
        csv_paths = paths
    # If csv_paths is provided as an argument, use it

    ground_truth_path, other_csv_paths = select_ground_truth(csv_paths)
    ground_truth = load_annotations(ground_truth_path, col=col) if ground_truth_path else None

    other_annotations = {}
    for csv_path in other_csv_paths:
        annotations = load_annotations(csv_path, col=col)
        other_annotations[csv_path] = annotations

    # Determine the length of the shortest set of annotations
    min_length = float('inf')
    all_annotations = {}
    for csv_path in csv_paths:
        annotations = load_annotations(csv_path, verbose=False, col=col)
        all_annotations[csv_path] = annotations
        min_length = min(min_length, len(annotations))

    for csv_path in all_annotations:
        all_annotations[csv_path] = {k: v for k, v in list(
            all_annotations[csv_path].items())[:min_length]}

    if ground_truth:
        ground_truth = {k: v for k, v in list(ground_truth.items())[:min_length]}

    print("\n")

    # label_map = {0:'straight', 1:'left cast', 2:'left shallow turn',
    #               3:'left sharp turn', 4:'right cast', 5:'right shallow turn', 6:'right sharp turn'}
    label_map ={0:'straight', 2:'left cast',
                  3:'left turn', 5:'right cast', 6:'right turn'}
    sequences = {}
    seg_lengths = {}
    info = {}
    info['label_map'] = label_map

    for csv_path, annotations in all_annotations.items():
        try:
            # Try to get a nice name for the CSV file
            dirname_parts = os.path.dirname(csv_path).split('/')
            if len(dirname_parts) >= 2:
                nice_name = dirname_parts[-2] + '_' + os.path.basename(csv_path).split('.csv')[0]
            else:
                nice_name = os.path.basename(csv_path).split('.csv')[0]
            print(f"Analyzing {nice_name}:")
        except Exception as e:
            # If anything goes wrong, just use the basename
            print(f"Analyzing {os.path.basename(csv_path)}:")
        sequenced = analyze_sequence(annotations)

        # Check for NaN values, but make sure the values are numeric first
        nan_indices = []
        for i, frame in enumerate(sequenced):
            try:
                if isinstance(frame[0], (int, float)) and np.isnan(frame[0]):
                    nan_indices.append(i)
            except (TypeError, ValueError):
                # If frame[0] can't be checked with isnan, it's not a NaN
                pass

        # Print the indices and corresponding frames
        if len(nan_indices) != 0:
            print("NaN indices found in annotation data, please resolve.")
            for i in nan_indices:
                print(f"NaN found at index {i}: frame = {sequenced[i]}")

        # Convert sequence to numeric arrays, handling non-numeric values
        sequence = []
        seq_counts = []
        for frame in sequenced:
            try:
                # Handle the case where frame[0] is a string that can be converted to a number
                if isinstance(frame[0], str):
                    try:
                        label_value = float(frame[0])
                        sequence.append(int(label_value))
                        seq_counts.append(int(frame[1]))
                        continue
                    except (ValueError, TypeError):
                        pass

                # Handle the case where frame[0] is already a number
                if isinstance(frame[0], (int, float)):
                    sequence.append(int(frame[0]))
                    seq_counts.append(int(frame[1]))
                else:
                    print(f"Warning: Unable to convert value to number: {frame}")
            except (TypeError, ValueError):
                print(f"Warning: Error processing value: {frame}")

        if not sequence:
            print("Warning: No valid sequence data found. Check your annotation format.")
            sequence = np.array([0])  # Default to avoid errors
            seq_counts = np.array([0])
        else:
            sequence = np.array(sequence)
            seq_counts = np.array(seq_counts)
        try:
            # Try to get a nice key for the CSV file
            dirname_parts = os.path.dirname(csv_path).split('/')
            if len(dirname_parts) >= 2:
                key = dirname_parts[-2] + '_' + os.path.basename(csv_path).split('.csv')[0]
            else:
                key = os.path.basename(csv_path).split('.csv')[0]
        except Exception:
            # If anything goes wrong, just use the basename
            key = os.path.basename(csv_path)
        lengths, counts, segment_lengths = compute_segment_stats(sequenced)
        seg_lengths[key] = {label: np.array(durations) for label, durations in segment_lengths.items()}
        sequences[key] = {'sequence': sequence, 'counts': seq_counts,
                            'segments': sum(counts.values())}
        for label in label_map:
            if label in lengths:
                print(f"Average segment length for {label_map[label]}: {lengths[label]:.2f} frames")
                print(f"           Total segments: {counts[label]}")
        print("\n")

        info[csv_path] = {'seg_lengths': seg_lengths, 'sequences': sequences,
              'lengths': lengths, 'counts': counts}

    for csv_path, annotations in other_annotations.items():
        if ground_truth:
            print(f"Comparing {os.path.basename(csv_path).split('.csv')[0]} to the ground truth\n")
            mismatch_frames, mismatch_percent = calculate_mismatches(ground_truth,
                                                                    annotations)
            precision_recall = compute_precision_recall(ground_truth, annotations)
            accuracy = compute_accuracy(ground_truth, annotations)
            conf_matrix = compute_confusion_matrix(ground_truth, annotations, labels=list(label_map.keys()))
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

            info[csv_path]['prec/recall'] = precision_recall
            info[csv_path]['mismatch_percent'] = mismatch_percent
            info[csv_path]['mismatch_frames'] = mismatch_frames
            info[csv_path]['accuracy'] = accuracy
            info[csv_path]['conf_matrix'] = conf_matrix

    comparison_results = compare_sequences(sequences)
    info['comparison_results'] = comparison_results

    for (key1, key2), result in comparison_results.items():
        aggregate_results = merge_intervals(result['missing_indices']
                                            + result['missmatch_indices'])
        colored_seq1, colored_seq2 = colorize_mismatches(result['best_config'][0],
                                                         result['best_config'][1])
        print(f"Comparison between {key1} and {key2}:")
        print(f"  Best Match Score: {result['best_match']:.2f}")
        print(f"  Alignment Score: {result['alignment_score']:.2f}\n")

        print(f"  Annotation sequences:")
        print(f" \t\t {colored_seq1}")
        print(f" \t\t {colored_seq2}\n")

        print(f"  Potential missed annotations in frames:\n{result['missing_indices']}")
        if len(result['missmatch_indices']) != 0:
            print(f"  Potential incorrect annotations in frames:\n{result['missmatch_indices']}\n")

        print(f"  Indices of concern: \n{aggregate_results}\n\n")
    plot_segment_lengths(seg_lengths, label_map)

    # Run advanced analyses if requested
    if advanced:
        try:
            # Define the function to convert annotations to AnnotationData format
            def convert_to_annotation_data(annotations):
                try:
                    # Import the AnnotationData class
                    from src.analysis.data_model import AnnotationData

                    # Create a new AnnotationData object
                    data = AnnotationData()

                    # Add the annotations
                    for frame, label in annotations.items():
                        # Convert label to integer if it's a string
                        if isinstance(label, str):
                            try:
                                label = int(float(label))
                            except (ValueError, TypeError):
                                # If conversion fails, keep the original label
                                pass

                        # Set the annotation using the set_annotation method
                        data.set_annotation(frame, {'label': label})

                    return data
                except ImportError:
                    print("Error: Could not import AnnotationData. Make sure the enhanced analysis tools are installed.")
                    return None

            # Define the function to run advanced analyses
            def run_advanced_analyses(annotations):
                try:
                    # Import the enhanced analysis tools
                    from src.analysis.statistics import StatisticalAnalysis

                    # Convert the annotations to the AnnotationData format
                    data = convert_to_annotation_data(annotations)
                    if data is None:
                        return {}

                    # Create a statistical analysis object
                    stats = StatisticalAnalysis()

                    # Dictionary to store the results
                    results = {}

                    # Run the analyses
                    print("\n=== Advanced Statistical Analyses ===\n")

                    # Basic statistics
                    print("Basic Statistics:")
                    try:
                        basic_stats = stats.basic_statistics(data)
                        print(f"  Total annotations: {basic_stats['total_annotations']}")
                        print(f"  Unique labels: {basic_stats['unique_labels']}")
                        print(f"  Label counts: {basic_stats['label_counts']}")
                        print(f"  Frame range: {basic_stats['frame_min']} - {basic_stats['frame_max']}")
                        print(f"  Total gaps: {basic_stats['total_gaps']}")
                        print(f"  Total gap frames: {basic_stats['total_gap_frames']}")
                        print(f"  Average gap size: {basic_stats['avg_gap_size']:.2f}")
                        print(f"  Maximum gap size: {basic_stats['max_gap_size']}")
                        results['basic_statistics'] = basic_stats
                    except Exception as e:
                        print(f"  Error: {str(e)}")

                    # Change point detection
                    print("\nChange Point Detection:")
                    try:
                        change_points = stats.detect_change_points(data, method='binary_segmentation', penalty='bic')
                        print(f"  Method: {change_points['method']}")
                        print(f"  Penalty: {change_points['penalty']}")
                        print(f"  Number of change points: {len(change_points['change_point_frames'])}")
                        print(f"  Change point frames: {change_points['change_point_frames']}")
                        print("  Segments:")
                        for i, segment in enumerate(change_points['segments']):
                            print(f"    Segment {i+1}:")
                            print(f"      Start frame: {segment['start_frame']}")
                            print(f"      End frame: {segment['end_frame']}")
                            print(f"      Length: {segment['length']}")
                            print(f"      Most common label: {segment.get('most_common_label', 'N/A')}")

                        # If there's a visualization, display it
                        if change_points.get('visualization'):
                            print("  Visualization available (will be displayed in web browser)")

                            # Save the visualization to a temporary file and open it
                            try:
                                import tempfile
                                import webbrowser

                                # Create a temporary HTML file
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                                    html_content = f"""<html>
                                    <head><title>Change Point Detection</title></head>
                                    <body>
                                    <h1>Change Point Detection ({change_points['method']}, {change_points['penalty']})</h1>
                                    <img src="{change_points['visualization']}" />
                                    </body>
                                    </html>"""
                                    f.write(html_content.encode('utf-8'))
                                    temp_path = f.name

                                # Open the HTML file in the default web browser
                                webbrowser.open('file://' + temp_path)
                            except Exception as e:
                                print(f"  Error displaying visualization: {str(e)}")

                        results['change_points'] = change_points
                    except Exception as e:
                        print(f"  Error: {str(e)}")
                        print("  Note: Change point detection requires the 'ruptures' package.")
                        print("  Install it with: conda install -c conda-forge ruptures")

                    # Segment Transition Matrix
                    print("\nSegment Transition Matrix:")
                    try:
                        segment_tm_results = stats.segment_transition_matrix(data)
                        print(f"  Number of segments: {len(segment_tm_results['segments'])}")
                        print(f"  Unique labels: {segment_tm_results['unique_labels']}")
                        print("  Transition matrix:")
                        for i, row in enumerate(segment_tm_results['transition_matrix']):
                            label = segment_tm_results['unique_labels'][i]
                            print(f"    Label {label}: {[round(p, 2) for p in row]}")

                        # If there's a visualization, display it
                        if segment_tm_results.get('visualization'):
                            print("  Visualization available (will be displayed in web browser)")

                            # Save the visualization to a temporary file and open it
                            try:
                                import tempfile
                                import webbrowser
                                import base64

                                # Create a temporary HTML file
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                                    html_content = f"""<html>
                                    <head><title>Segment Transition Matrix</title></head>
                                    <body>
                                    <h1>Segment Transition Matrix</h1>
                                    <img src="{segment_tm_results['visualization']}" />
                                    </body>
                                    </html>"""
                                    f.write(html_content.encode('utf-8'))
                                    temp_path = f.name

                                # Open the HTML file in the default web browser
                                webbrowser.open('file://' + temp_path)
                            except Exception as e:
                                print(f"  Error displaying visualization: {str(e)}")

                        results['segment_transition_matrix'] = segment_tm_results
                    except Exception as e:
                        print(f"  Error: {str(e)}")

                    # Hidden Markov Model analysis
                    print("\nHidden Markov Model Analysis:")
                    try:
                        hmm_results = stats.hmm_analysis(data, n_states=3)
                        print(f"  Number of states: {hmm_results['n_states']}")
                        print("  Most likely labels for each state:")
                        for state, label in hmm_results['most_likely_labels'].items():
                            print(f"    State {state}: {label}")
                        print("  Transition matrix:")
                        for i, row in enumerate(hmm_results['transition_matrix']):
                            print(f"    State {i}: {[round(p, 2) for p in row]}")
                        results['hmm_analysis'] = hmm_results
                    except Exception as e:
                        print(f"  Error: {str(e)}")
                        print("  Note: HMM analysis requires the 'hmmlearn' package.")
                        print("  Install it with: conda install -c conda-forge hmmlearn")

                    # Anomaly detection
                    print("\nAnomaly Detection:")

                    # Z-score method
                    print("\nZ-score Method:")
                    try:
                        anomalies = stats.detect_anomalies(data, method="zscore", threshold=3.0)
                        if not anomalies:
                            print("  No anomalies detected.")
                        else:
                            print(f"  Detected {len(anomalies)} anomalies:")
                            for frame, anomaly in sorted(anomalies.items())[:5]:  # Show only the first 5 anomalies
                                print(f"    Frame {frame}:")
                                print(f"      Label: {anomaly['label']}")
                                print(f"      Duration: {anomaly['duration']}")
                                print(f"      Z-score: {anomaly['zscore']:.2f}")
                        results['anomalies_zscore'] = anomalies
                    except Exception as e:
                        print(f"  Error: {str(e)}")

                    # IQR method
                    print("\nIQR Method:")
                    try:
                        anomalies = stats.detect_anomalies(data, method="iqr", threshold=1.5)
                        if not anomalies:
                            print("  No anomalies detected.")
                        else:
                            print(f"  Detected {len(anomalies)} anomalies:")
                            for frame, anomaly in sorted(anomalies.items())[:5]:  # Show only the first 5 anomalies
                                print(f"    Frame {frame}:")
                                print(f"      Label: {anomaly['label']}")
                                print(f"      Duration: {anomaly['duration']}")
                                print(f"      IQR: {anomaly['iqr']:.2f}")
                        results['anomalies_iqr'] = anomalies
                    except Exception as e:
                        print(f"  Error: {str(e)}")

                    print("\nAdvanced statistical analysis completed successfully!")

                    return results
                except ImportError:
                    print("Error: Could not import StatisticalAnalysis. Make sure the enhanced analysis tools are installed.")
                    return {}

            print("\n=== Running Advanced Analyses ===\n")

            # Run advanced analyses on each annotation set
            for csv_path, annotations in all_annotations.items():
                print(f"\nAdvanced Analysis for {os.path.basename(csv_path).split('.csv')[0]}:")
                advanced_results = run_advanced_analyses(annotations)

                # Add the advanced results to the info dictionary
                if csv_path in info:
                    info[csv_path]['advanced'] = advanced_results
                else:
                    info[csv_path] = {'advanced': advanced_results}
        except ImportError:
            print("\nError: Could not import advanced analysis functions.")
            print("Make sure the enhanced analysis tools are installed.")
        except Exception as e:
            print(f"\nError running advanced analyses: {str(e)}")

    return info

def main(csv_paths=None, advanced=False):
    analysis(csv_paths=csv_paths, advanced=advanced)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, nargs='*', help="Paths to the CSV files")
    parser.add_argument('--advanced', action='store_true', default=False,
                      help="Run advanced analyses")
    args = parser.parse_args()

    # Run the analysis
    main(csv_paths=args.csv, advanced=args.advanced)