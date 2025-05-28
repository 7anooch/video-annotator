import os
import pandas as pd
import argparse
import tkinter as tk
import numpy as np
from tkinter import filedialog
from scipy.ndimage import label

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
    
def get_all_annotations(csv_paths):
    all_annotations = {}
    min_length = float('inf')

    for csv_path in csv_paths:
        annotations = load_annotations(csv_path)
        all_annotations[csv_path] = annotations
        min_length = min(min_length, len(annotations))

    return all_annotations, min_length

def connected_components_1D(arr, dust_size=None):
    lbl, n_features = label(arr)
    indices = np.indices(lbl.shape).T[:, 0]
    cc_dict = {}
    for i in range(1, n_features + 1):
        cc_i = indices[lbl == i]
        if dust_size is not None:
            if np.shape(cc_i)[0] < dust_size:
                continue

        cc_dict[i] = cc_i

    return cc_dict

def analyze_sequence(annotations):
    if not annotations:
        return []

    sorted_frames = sorted(annotations.keys())
    current_label = annotations[sorted_frames[0]]
    start_frame = sorted_frames[0]
    sequence_summary = []
    frame_list = []

    for frame in sorted_frames[1:]:
        label = annotations[frame]
        if label != current_label:
            duration = frame - start_frame
            frame_list.append(frame)
            sequence_summary.append((current_label, duration, frame_list))
            current_label = label
            start_frame = frame
            frame_list = []
        frame_list.append(frame)

    duration = sorted_frames[-1] - start_frame + 1
    sequence_summary.append((current_label, duration, frame_list))

    return sequence_summary

def find_gaps(updated_ground_truth_dict, max_gap_size=30):
    components = analyze_sequence(updated_ground_truth_dict)
    label_cc = {}
    for i, (lbl, dur, fr) in enumerate(components):
        if lbl not in label_cc:
            label_cc[lbl] = []
        label_cc[lbl].append([i, dur, fr])

    gaps = []
    for l in label_cc.keys():
        ind_diff  = np.diff([x[0] for x in label_cc[l]])
        diff2 = np.where(ind_diff == 2)[0]
        for d in diff2:
            gap_start = label_cc[l][d][2][-1]
            gap_end = label_cc[l][d+1][2][0]
            gap_frames = list(range(gap_start, gap_end+1))
            gap_size = len(gap_frames)
            if gap_size < max_gap_size:
                gaps.append((l, gap_size, gap_frames))
                print(f'set segment of length {gap_size} to label:',  l)
                print(gap_frames)

    gaps.sort(key=lambda x: x[1])
    return gaps

def fill_missing_labels(ground_truth_path, ground_truth_ds_path, output_path):
    ground_truth = load_annotations(ground_truth_path)
    ground_truth_ds = load_annotations(ground_truth_ds_path)

    for frame, label in ground_truth.items():
        if int(label) == -1:
            ground_truth[frame] = ground_truth_ds.get(frame, label)

    updated_ground_truth_list = [{'frame': frame, 'label': int(label)} 
                                 for frame, label in sorted(ground_truth.items())]
    updated_ground_truth_dict = {entry['frame']: entry['label'] 
                                 for entry in updated_ground_truth_list}
    
    gaps = find_gaps(updated_ground_truth_dict)

    while gaps != []:
        for l, gap_size, gap_frames in gaps:
            for entry in updated_ground_truth_list:
                if entry['frame'] in gap_frames:
                    entry['label'] = l

            updated_ground_truth_dict = {ent['frame']: ent['label'] 
                                        for ent in updated_ground_truth_list}

            gaps = find_gaps(updated_ground_truth_dict)

    df_updated_gt = pd.DataFrame(updated_ground_truth_list)
    df_updated_gt.to_csv(output_path, index=False)
    print(f"\nUpdated ground truth saved to {output_path}")

def main():
    csv_paths = get_csv_paths()
    if csv_paths:
        ground_truth_path = None
        ground_truth_ds_path = None

        for path in csv_paths:
            if 'ground_truth_DS.csv' in path:
                ground_truth_ds_path = path

            elif 'ground_truth.csv' in path:
                ground_truth_path = path

        if not ground_truth_path or not ground_truth_ds_path:
                    print("Error: Could not identify ground_truth and ground_truth_DS annotations.")
                    return
        
        output_path = ground_truth_path.replace('ground_truth', 
                                                'ground_truth_patched')
        fill_missing_labels(ground_truth_path, ground_truth_ds_path, output_path)
        
    else:
        print("No CSV file selected.")

if __name__ == "__main__":
    main()