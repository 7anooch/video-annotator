import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import tkinter as tk
from tkinter import filedialog
import matplotlib.patches as mpatches

def plot_ethogram(annotations, title):
    # Define the color mapping for annotations
    color_mapping = {
        "0": "red",
        "1": "green",
        "2": "blue"
    }
    # Prepare data for plotting
    frames = sorted(annotations.keys())
    behaviors = [annotations[frame] for frame in frames]
    colors = [color_mapping.get(str(behavior), "black") for behavior in behaviors]

    # Plot the ethogram
    plt.figure(figsize=(12, 2))  # Adjust the figure size to be bigger vertically
    plt.vlines(frames, ymin=0, ymax=1, colors=colors, linewidth=2)
    plt.yticks([])
    plt.xlabel("Frame Number")
    plt.title(title)

    # Create a legend
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_mapping.items()]
    plt.legend(handles=legend_patches, loc='upper right')

    plt.show()

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
    parser.add_argument('--paths', type=str, nargs='*', help="Paths to the CSV files")
    args = parser.parse_args()

    if args.paths:
        return args.paths
    else:
        root = tk.Tk()
        root.withdraw()
        csv_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        return csv_paths

if __name__ == "__main__":
    csv_paths = get_csv_paths()
    if csv_paths:
        all_annotations = {}
        for csv_path in csv_paths:
            annotations = load_annotations(csv_path)
            all_annotations[csv_path] = annotations
            plot_ethogram(annotations, title=os.path.basename(csv_path))
        # print(all_annotations)
    else:
        print("No CSV file selected.")