import scipy.io
import csv
import os
import tkinter as tk
import argparse
from tkinter import filedialog
import numpy as np
from glob import glob

parser = argparse.ArgumentParser(description="Convert kinData from MATLAB file to CSV files (raw, no manipulation).")
parser.add_argument("--matfile", type=str, help="Path to the MATLAB file", default=None)
parser.add_argument("--outputdir", type=str, help="Path to the output directory", default=None)
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
        continue
    data = scipy.io.loadmat(mat_file_path)
    kin_data = data['allSupmtx']

    output_dir = os.path.dirname(mat_file_path)

    for n in range(kin_data.shape[1]):
        if kin_data[0, n].size == 0:
            continue
        trial = kin_data[0, n]

        path_name = f'trial_{n+1}_raw.csv'
        output_csv_path = os.path.join(output_dir, path_name)

        # Write the entire trial matrix as-is to CSV
        with open(output_csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for row in trial:
                writer.writerow(row)

        if n % 8 == 0 or n == kin_data.shape[1] - 1:
            print(f"Exported trial {n+1} to {output_csv_path}")