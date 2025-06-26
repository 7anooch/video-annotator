import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import tkinter as tk
from tkinter import filedialog
import matplotlib.patches as mpatches

def plot_ethogram(ax, annotations, title, show_legend=True, show_xlabel=True, pc = False):
    color_mapping, label_list, _ = get_color_mappings_and_labels(annotations)

    frames = sorted(annotations.keys())
    behaviors = [annotations[frame] for frame in frames] # default to neon yellow for unknown behaviors
    colors = [color_mapping.get(str(int(behavior)), "#FFFF00")
               if not np.isnan(behavior) else "#FFFF00" for behavior in behaviors]


    # Fill the region for each frame
    for i, frame in enumerate(frames[:-1]):  # Exclude the last frame to avoid out-of-bounds
        ax.fill_between(
            [frame, frames[i + 1]],  # x range for the current frame
            0,  # ymin
            1,  # ymax
            color=colors[i],  # Color for the current behavior
            step="pre"  # Step style to create sharp transitions
        )
    # ax.vlines(frames, ymin=0, ymax=1, colors=colors, linewidth=2)
    ax.set_yticks([])
    if show_xlabel:
        if pc == True:
            ax.set_xlabel("Peristaltic Cycle")
        else:
            ax.set_xlabel("Frame Number")
    ax.set_title(title)
    ax.set_xlim(frames[0], frames[-1])

    if show_legend:
        legend_patches = [mpatches.Patch(color=color_mapping[str(key)], label=label) 
                    for key, label in zip(color_mapping.keys(), label_list)]
        ax.legend(handles=legend_patches, loc='upper right')

def get_color_mappings_and_labels(annotations):
    unique_labels = set(annotations.values())
    # unique_labels.discard(-1)

    unique_labels = {label for label in unique_labels if not np.isnan(label)}

    # Determine the annotation type based on the unique labels
    if unique_labels == {3, 4, 5}:
        annotation_type = "confidence"
    elif unique_labels == {0, 1}:
        annotation_type = "stop"
    elif unique_labels == {0, 2, 3, 4}:
        annotation_type = "directionless"
    elif unique_labels.issubset({0, 1, -1}):
        annotation_type = "acceptance"
    elif unique_labels.issubset({0, 1, 2, 3, 4, 5, 6, 7,8,9,11}):
        annotation_type = "ethogram"
    else:
        print(unique_labels)
        annotation_type = "ethogram"
        # raise ValueError("Unknown annotation type based on labels: {}".format(unique_labels))

    color_mappings = {

            "ethogram": {
            "0": "black",
            "1": "#CC79A7",    # Purple
            "2": "#56B4E9",     # Sky Blue
            "3": "#009E73",     # Green
            "4": "#CC79A7",    # Purple
            "5": "#E69F00",     # Orange
            "6": "#D55E00",     # Vermilion
            "7": "#8B0000",    # Dark Red (distinct from others)
            "8": "#8B0000",    # Dark Red (distinct from others)
            "9": "#0057B8",   # Deep Blue (vivid, high contrast)
            # "9": "#00CED1",    # Dark Turquoise (bright cyan, high contrast)
            # "10": "#CC79A7",   # Purple (placeholder)
            "11": "#0057B8",   # Deep Blue (vivid, high contrast)
        },
        "directionless": {
            "0": "black",
            "2": "#56B4E9",     # Sky Blue
            "3": "#009E73",     # Green
            "4": "#E57B2F"    # Orange

        },
        "confidence": {
            "3": "black",
            "4": "gray",
            "5": "white"
        },
        "mismatch": {
            "0": "black",
            "1": "white",
        },
        "acceptance": {
            "0": 'black',
            "1": 'green',
            '-1': 'red'
        },
        "stop": {"0": "black",
                "1": "green"}
    }
    labels = {
        # "ethogram": ['straight', 'left cast', 'left turn',
        #              'left sharp turn', 'right cast', 'right turn', 'l cast / r turn', 
        #              'r cast / l turn', 'l cast/turn', 'r cast/turn'],
        "ethogram": ['straight', 'left cast', 'left turn', 'right cast', 
                'right turn', 'opposite cast/turn', 'opposite cast/turn', 
                'same side cast/turn', 'same side cast/turn'],
        "directionless": ['straight', 'cast', 'turn', 'cast + turn'],
        "confidence": ['low', 'medium', 'high'],
        "mismatch": ['mismatch', 'match'],
        "acceptance": ['N/A', 'accept', 'reject'],
        "stop": ['stop', 'run']
    }

    return color_mappings[annotation_type], labels[annotation_type], annotation_type

def get_all_labels(csv_list, col = 2):
    all_annotations = []
    if isinstance(col, (list, tuple)) and len(col) == 2:
        print(f'Merging two columns, {col}')
        for file in csv_list:
            df = pd.read_csv(file, header=None)
            col1 = df[col[0]].values
            col2 = df[col[1]].values
            merged =  np.zeros(len(col1))

            special_case = ((col1 == 2) & (col2 == 6)) | ((col1 == 6) & (col2 == 2))
            special_case2 = ((col1 == 3) & (col2 == 2)) | ((col1 == 2) & (col2 == 3))
            merged[special_case] = 7  # Special case: set merged to 7
            merged[special_case2] = 9  # Special case: set merged to 9
            non_special = ~special_case & ~special_case2
            merged[non_special] = col1[non_special] + col2[non_special]
            print(np.unique(merged))
            
            all_annotations.append(merged)
        all_annotations = np.concatenate(all_annotations)
        return all_annotations
    else:
        for file in csv_list:
            df = pd.read_csv(file, header=None)
            all_annotations.append(df[col].values)
        all_annotations = np.concatenate(all_annotations)
    return all_annotations

def load_annotations(csv_path, col = 2):
    annotations = {}
    # if 'Persistalsis' in csv_path:
        # print(col)
    if os.path.exists(csv_path):
        if isinstance(col, (list, tuple)) and len(col) == 2:
            df = pd.read_csv(csv_path, header=None)
            col1 = df[col[0]].values
            # print(f"col1: {np.unique(col1)}")
            col2 = df[col[1]].values
            # print(f"col2: {np.unique(col2)}")
            merged =  np.zeros(len(col1))
            col1 = np.nan_to_num(col1, nan=0)
            col2 = np.nan_to_num(col2, nan=0)
            # print(np.sum(np.logical_and(col1 != 0, col2 != 0)))
            c1 = np.where(col1 != 0)
            c2 = np.where(col2 != 0)
            # print('intersection', np.intersect1d(c1, c2))

            # special_case = ((col1 == 2) & (col2 == 6)) | ((col1 == 6) & (col2 == 2))
            special_case2 = ((col1 == 3) & (col2 == 2)) | ((col1 == 2) & (col2 == 3))
            # merged[special_case] = 7  # Special case: set merged to 7
            # merged[special_case2] = 9  # Special case: set merged to 9
            merged[special_case2] = 11
            non_special = ~special_case2
            merged[non_special] = col1[non_special] + col2[non_special]
            annotations = {index+1: label for index, label in enumerate(merged)}
            # print(np.unique(list(annotations.values())))
        else:
            df = pd.read_csv(csv_path, header=None, usecols=[0,col], names=['frame', 'label'])
            annotations = {row['frame']: row['label'] for _, row in df.iterrows()}
        print(f"Loaded annotations from {csv_path}")
    else:
        print(f"No annotation file found at {csv_path}")
    return annotations

def get_csv_paths():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--csv', type=str, nargs='*', help="Paths to the CSV files")
    # args = parser.parse_args()

    # if args.csv:
    #     return args.csv
    # else:
    root = tk.Tk()
    root.withdraw()
    csv_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
    return csv_paths
    
def gen_figure(*paths, use_cols, return_fig=False, pc = False, titles=None):
    if not paths:
        csv_paths = get_csv_paths()
        if not csv_paths:
            print("No CSV file selected.")
            return
    else:
        csv_paths = paths

    all_annotations = {}
    min_length = float('inf')
    all_ethogram = True

    for csv_path in csv_paths:
        annotations = load_annotations(csv_path, use_cols)
        all_annotations[csv_path] = annotations
        min_length = min(min_length, len(annotations))
        _, _, annotation_type = get_color_mappings_and_labels(annotations)

        if annotation_type != "ethogram":
            all_ethogram = False

    num_files = len(csv_paths)
    fig, axes = plt.subplots(num_files, 1, figsize=(16, 3 * num_files)) 

    if num_files == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot

    used_labels = []
    plot_count = 0

    for ax, csv_path in zip(axes, csv_paths):
        plot_count += 1
        annotations = all_annotations[csv_path]
        capped_annotations = {frame: annotations[frame] for frame 
                              in sorted(annotations.keys(), key=int)[:min_length]}
        # capped_annotations = {frame: annotations[frame] for frame
        #                         in sorted(annotations.keys())[:min_length]}
        used_labels.extend(list(capped_annotations.values()))

        if titles is not None and len(titles) == num_files:
            title = titles[plot_count - 1]
        else:
            title = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))

        if plot_count == num_files:
            plot_ethogram(ax, capped_annotations, title=title, 
                            show_legend=not all_ethogram, pc=pc)
        else:
            plot_ethogram(ax, capped_annotations, title=title, 
                            show_legend=not all_ethogram, show_xlabel=False, pc=pc)
    
    used_labels = np.unique(used_labels)
    print(f"Used labels: {used_labels}")
        
    if all_ethogram:
        # labels = ['straight', '', 'left cast', 'left turn', '', 'right cast', 
        #           'right turn', 'l cast / r turn', 'r cast / l turn', 'l cast/turn', '', 'r cast/turn']
        labels = ['straight', '', 'left cast', 'left turn', '', 'right cast', 
                  'right turn', 'opposite side cast/turn', 'opposite side cast/turn', 'same side cast/turn', '', 'same side cast/turn']
        
        color_mapping = {
            "0": "black",
            "1": "#CC79A7",    # Purple
            "2": "#56B4E9",     # Sky Blue
            "3": "#009E73",     # Green
            "4": "#CC79A7",    # Purple
            "5": "#E69F00",     # Orange
            "6": "#D55E00",     # Vermilion
            "7": "#8B0000",    # Dark Red (distinct from others)
            "8": "#8B0000",    # Dark Red (distinct from others)
            "9": "#0057B8",   # Deep Blue (vivid, high contrast)
            # "9": "#00CED1",    # Dark Turquoise (bright cyan, high contrast)
            "10": "#CC79A7",   # Purple (placeholder)
            "11": "#0057B8",
        }

        # Create a legend
        legend_patches = [mpatches.Patch(color=color_mapping[str(i)], 
                                        label=label) for i, label in enumerate(labels) if i in used_labels]
        fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cols', type=str, default="2")
    args = parser.parse_args()

    use_cols = [int(col) for col in args.use_cols.split(",")]
    if len(use_cols) == 1:
        use_cols = use_cols[0]

    gen_figure(use_cols=use_cols)

if __name__ == "__main__":
    main()