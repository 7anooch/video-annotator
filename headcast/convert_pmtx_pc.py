import scipy.io
import csv
import os
import tkinter as tk
import argparse
from tkinter import filedialog
import numpy as np
from headcast.headcast_funcs import get_labels

parser = argparse.ArgumentParser(description="Convert kinData from MATLAB file to CSV files.")
parser.add_argument("--matfile", type=str, help="Path to the MATLAB file", default=None)
parser.add_argument("--outputdir", type=str, help="Path to the output directory", default=None)

INVERT_LEFT_RIGHT = False
OUTWARD_ONLY = False  # If True, only outward casts are considered

args = parser.parse_args()

mat_file_path = args.matfile
if not mat_file_path or not os.path.exists(mat_file_path):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    mat_dir_path = filedialog.askdirectory(
        title="Select MATLAB file dir",

    )
    mat_file_path = os.path.join(mat_dir_path, 'kinData_final.mat')
    print(mat_file_path)
    if not mat_file_path or not os.path.exists(mat_file_path):
        print("No valid .mat file in chosen directory.")
        exit()

data = scipy.io.loadmat(mat_file_path)
kin_data = data['kinData']

output_dir = args.outputdir
if not output_dir:
    output_dir = filedialog.askdirectory(title="Select Output Directory")
if not output_dir:
    print("No output directory selected. Exiting.")
    exit()

for n in range(kin_data.shape[1]):
    trial_data = kin_data[0, n][0,0]
    try:
        trial = trial_data['pmtx']
    except Exception as e:
        print(f"Skipping trial {n+1}: {e}")
        continue

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
        for index, mode_value in enumerate(mode_data, start=1):
            writer.writerow([index, 0, mode_value, mode_cast[index-1], 
                            mode_turn[index-1], mode_accept[index-1]])

    if n % 8 == 0 or n == kin_data.shape[1] - 1:
        print(f"Exported trial {n+1} to {output_csv_path}")