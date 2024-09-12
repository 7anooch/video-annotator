import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import tkinter as tk
from tkinter import filedialog
import matplotlib.patches as mpatches

def plot_ethogram(ax, annotations, title):
    # Define the color mapping for annotations
    color_mapping = {
        "0": "red",
        "1": "black",
        "2": "blue"
    }
    # Prepare data for plotting
    frames = sorted(annotations.keys())
    behaviors = [annotations[frame] for frame in frames]
    colors = [color_mapping.get(str(int(behavior)), "gray") for behavior in behaviors]

    # Create the plot
    ax.vlines(frames, ymin=0, ymax=1, colors=colors, linewidth=2)
    ax.set_yticks([])
    ax.set_xlabel("Frame Number")
    ax.set_title(title)

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

if __name__ == "__main__":
    csv_paths = get_csv_paths()
    if csv_paths:
        all_annotations = {}
        min_length = float('inf')

        for csv_path in csv_paths:
            annotations = load_annotations(csv_path)
            all_annotations[csv_path] = annotations
            min_length = min(min_length, len(annotations))

        num_files = len(csv_paths)
        fig, axes = plt.subplots(num_files, 1, figsize=(10, 3 * num_files)) 

        if num_files == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot

        for ax, csv_path in zip(axes, csv_paths):
            annotations = all_annotations[csv_path]
            capped_annotations = {frame: annotations[frame] for frame
                                   in sorted(annotations.keys())[:min_length]}
            plot_ethogram(ax, capped_annotations, title=os.path.basename(csv_path))

        labels = ['stop', 'run', 'turn']
        color_mapping = {
        "0": "red",
        "1": "black",
        "2": "blue"
        }
        # Create a legend
        legend_patches = [mpatches.Patch(color=color_mapping[str(i)], label=label) for i, label in enumerate(labels)]
        fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.show()
    else:
        print("No CSV file selected.")