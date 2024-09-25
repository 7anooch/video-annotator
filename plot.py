import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import tkinter as tk
from tkinter import filedialog
import matplotlib.patches as mpatches

def plot_ethogram(ax, annotations, title, show_legend=True, show_xlabel=True):
    color_mapping, label_list, _ = get_color_mappings_and_labels(annotations)

    frames = sorted(annotations.keys())
    behaviors = [annotations[frame] for frame in frames] # default to neon yellow for unknown behaviors
    colors = [color_mapping.get(str(int(behavior)), "#FFFF00") for behavior in behaviors]

    ax.vlines(frames, ymin=0, ymax=1, colors=colors, linewidth=2)
    ax.set_yticks([])
    if show_xlabel:
        ax.set_xlabel("Frame Number")
    ax.set_title(title)
    ax.set_xlim(frames[0], frames[-1])

    if show_legend:
        legend_patches = [mpatches.Patch(color=color_mapping[str(key)], label=label) 
                    for key, label in zip(color_mapping.keys(), label_list)]
        ax.legend(handles=legend_patches, loc='upper right')

def get_color_mappings_and_labels(annotations):
    unique_labels = set(annotations.values())
    unique_labels.discard(-1)

    # Determine the annotation type based on the unique labels
    if unique_labels == {0, 1, 2}:
        annotation_type = "ethogram"
    elif unique_labels == {3, 4, 5}:
        annotation_type = "confidence"
    elif unique_labels == {0, 1}:
        annotation_type = "mismatch"
    else:
        raise ValueError("Unknown annotation type based on labels: {}".format(unique_labels))

    color_mappings = {
        "ethogram": {
            "0": "red",
            "1": "black",
            "2": "blue"
        },
        "confidence": {
            "3": "black",
            "4": "gray",
            "5": "white"
        },
        "mismatch": {
            "0": "black",
            "1": "white",
        }
    }
    labels = {
        "ethogram": ['stop', 'run', 'turn'],
        "confidence": ['low', 'medium', 'high'],
        "mismatch": ['mismatch', 'match']
    }

    return color_mappings[annotation_type], labels[annotation_type], annotation_type

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
        all_ethogram = True

        for csv_path in csv_paths:
            annotations = load_annotations(csv_path)
            all_annotations[csv_path] = annotations
            min_length = min(min_length, len(annotations))
            _, _, annotation_type = get_color_mappings_and_labels(annotations)
            if annotation_type != "ethogram":
                all_ethogram = False

        num_files = len(csv_paths)
        fig, axes = plt.subplots(num_files, 1, figsize=(10, 3 * num_files)) 

        if num_files == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot

        plot_count = 0    
        for ax, csv_path in zip(axes, csv_paths):
            plot_count += 1
            annotations = all_annotations[csv_path]
            capped_annotations = {frame: annotations[frame] for frame
                                   in sorted(annotations.keys())[:min_length]}
            if plot_count == num_files:
                plot_ethogram(ax, capped_annotations, title=os.path.basename(csv_path), 
                              show_legend=not all_ethogram)
            else:
                plot_ethogram(ax, capped_annotations, title=os.path.basename(csv_path), 
                              show_legend=not all_ethogram, show_xlabel=False)
            

        if all_ethogram:
            labels = ['stop', 'run', 'turn']
            color_mapping = {
            "0": "red",
            "1": "black",
            "2": "blue"
            }
            # Create a legend
            legend_patches = [mpatches.Patch(color=color_mapping[str(i)], 
                                            label=label) for i, label in enumerate(labels)]
            fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.show()
    else:
        print("No CSV file selected.")