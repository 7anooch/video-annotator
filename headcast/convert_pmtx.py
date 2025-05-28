import scipy.io
import csv
import os
import tkinter as tk
import argparse
from tkinter import filedialog
import numpy as np
from glob import glob
from headcast.headcast_funcs import get_labels

parser = argparse.ArgumentParser(description="Convert kinData from MATLAB file to CSV files.")
parser.add_argument("--matfile", type=str, help="Path to the MATLAB file", default=None)
parser.add_argument("--outputdir", type=str, help="Path to the output directory", default=None)

INVERT_LEFT_RIGHT = False
OUTWARD_ONLY = False  # If True, only outward casts are considered 

args = parser.parse_args()

def find_subdirs(parent_dir):
    pattern = os.path.join(parent_dir, '**', 'all_supmtx.mat')
    return [os.path.dirname(file_path) for file_path in glob(pattern, recursive=True)]

# Load the MATLAB file
mat_file_path = args.matfile
if not mat_file_path or not os.path.exists(mat_file_path):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    directory_path = filedialog.askdirectory(
            title="Select Directory Containing MATLAB Files"
        )
    dirs = find_subdirs(directory_path)
    if len(dirs) == 0:
        print("No file selected. Exiting.")
        exit()
else:
    dirs = [os.path.dirname(mat_file_path)]

for directory in dirs:
    mat_file_path = os.path.join(directory, 'all_supmtx.mat')
    if not os.path.exists(mat_file_path):
        print(f"File not found: {mat_file_path}")
    data = scipy.io.loadmat(mat_file_path)
    kin_data = data['allSupmtx']

    # output to same directory as the MATLAB file
    output_dir = os.path.dirname(mat_file_path)

    for n in range(kin_data.shape[1]):
        if kin_data[0, n].size == 0:
            continue
        trial = kin_data[0, n]

        cols = get_labels(trial, 
                        outward_only=OUTWARD_ONLY,
                        invert_left_right=INVERT_LEFT_RIGHT)
        mode_data, mode_cast, mode_turn, mode_accept = cols

        path_name  = f'trial_{n+1}_pc.csv'
        if OUTWARD_ONLY:
            path_name = path_name.replace('.csv', '_outward.csv')
        if INVERT_LEFT_RIGHT:
            path_name = path_name.replace('.csv', '_invert.csv')

        output_csv_path = os.path.join(output_dir, path_name)

        with open(output_csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Write frame number and mode value
            for index, mode_value in enumerate(mode_data, start=1):
                writer.writerow([index, 0, mode_value, mode_cast[index-1], mode_turn[index-1], mode_accept[index-1]])

        if n % 8 == 0 or n == kin_data.shape[1] - 1:
            print(f"Exported trial {n+1} to {output_csv_path}")