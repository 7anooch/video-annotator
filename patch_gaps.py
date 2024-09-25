import os
import pandas as pd
import argparse
import tkinter as tk
from tkinter import filedialog

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

def fill_missing_labels(ground_truth_path, ground_truth_ds_path, output_path):
    ground_truth = load_annotations(ground_truth_path)
    ground_truth_ds = load_annotations(ground_truth_ds_path)

    for frame, label in ground_truth.items():
        if int(label) == -1:
            ground_truth[frame] = ground_truth_ds.get(frame, label)

    updated_ground_truth_list = [{'frame': frame, 'label': int(label)} 
                                 for frame, label in sorted(ground_truth.items())]
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