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
args = parser.parse_args()

mat_file_path = args.matfile
if not mat_file_path or not os.path.exists(mat_file_path):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    mat_file_path = filedialog.askdirectory(
        title="Select MATLAB file dir",

    )
    mat_file_path = os.path.join(mat_file_path, 'kinData_final.mat')
    print(mat_file_path)
    if not mat_file_path:
        print("No file selected. Exiting.")
        exit()

data = scipy.io.loadmat(mat_file_path)
kin_data = data['kinData']

output_dir = os.path.dirname(mat_file_path)

for n in range(kin_data.shape[1]):
    trial_data = kin_data[0, n][0,0]
    try:
        trial = trial_data['pmtx']
    except Exception as e:
        print(f"Skipping trial {n+1}: {e}")
        continue

    path_name = f'trial_{n+1}_raw.csv'
    output_csv_path = os.path.join(output_dir, path_name)

    # Write the entire trial matrix as-is to CSV
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in trial:
            writer.writerow(row)

    if n % 8 == 0 or n == kin_data.shape[1] - 1:
        print(f"Exported trial {n+1} to {output_csv_path}")