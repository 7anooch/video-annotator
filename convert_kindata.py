import scipy.io
import csv
import os
import tkinter as tk
import argparse
from tkinter import filedialog
import numpy as np

parser = argparse.ArgumentParser(description="Convert kinData from MATLAB file to CSV files.")
parser.add_argument("--matfile", type=str, help="Path to the MATLAB file", default=None)
parser.add_argument("--outputdir", type=str, help="Path to the output directory", default=None)
parser.add_argument("--translate_labels", action='store_true', default=False)

args = parser.parse_args()

# Load the MATLAB file
mat_file_path = args.matfile
if not mat_file_path or not os.path.exists(mat_file_path):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    mat_file_path = filedialog.askopenfilename(
        title="Select MATLAB File",
        filetypes=[("MATLAB Files", "*.mat")]
    )
    if not mat_file_path:
        print("No file selected. Exiting.")
        exit()

data = scipy.io.loadmat(mat_file_path)

# Access the kinData structure
kin_data = data['kinData']

# Create a directory to save the CSV files
output_dir = args.outputdir
if not output_dir:
    output_dir = filedialog.askdirectory(
    title="Select Output Directory")

if not output_dir:
    print("No output directory selected. Exiting.")
    exit()

# Loop through each trial in kinData
for n in range(kin_data.shape[1]):
    trial_data = kin_data[0, n][0,0]  
    mode_data = trial_data['mode'].ravel()
    # print(np.unique(mode_data))

    # Create a CSV file for this trial
    if args.translate_labels == True:
        output_csv_path = os.path.join(output_dir, f'trial_{n+1}_trans.csv')
    else:
        output_csv_path = os.path.join(output_dir, f'trial_{n+1}.csv')
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write frame number and mode value
        for frame_index, mode_value in enumerate(mode_data, start=1):
            if args.translate_labels == True:
                if mode_value == 1:
                    mode_value = 0
                elif mode_value == 4:
                    mode_value = 3
                elif mode_value == 7:
                    mode_value = 6 


            writer.writerow([frame_index, 0, mode_value])

    # print(f"Exported trial {n+1} to {output_csv_path}")