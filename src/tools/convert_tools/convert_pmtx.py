import scipy.io
import csv
import os
import tkinter as tk
import argparse
from tkinter import filedialog
import numpy as np
from glob import glob

parser = argparse.ArgumentParser(description="Convert kinData from MATLAB file to CSV files.")
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
    # mat_file_path = filedialog.askopenfilename(
    #     title="Select MATLAB File",
    #     filetypes=[("MATLAB Files", "*.mat")]
    # )
    # if not mat_file_path:
    #     print("No file selected. Exiting.")
    #     exit()


for directory in dirs:
    mat_file_path = os.path.join(directory, 'all_supmtx.mat')
    if not os.path.exists(mat_file_path):
        print(f"File not found: {mat_file_path}")
    data = scipy.io.loadmat(mat_file_path)

    # kin_data = data['kinData']
    kin_data = data['allSupmtx']

    # Create a directory to save the CSV files
    output_dir = os.path.dirname(mat_file_path)
    # output_dir = args.outputdir
    # if not output_dir:
    #     output_dir = filedialog.askdirectory(
    #     title="Select Output Directory")

    # if not output_dir:
    #     print("No output directory selected. Exiting.")
    #     exit()


    for n in range(kin_data.shape[1]):
        if kin_data[0, n].size == 0:
            continue
        trial = kin_data[0, n]
        # trial_data = kin_data[0, n][0,0]
        # try:
        #     trial = trial_data['peristalsisMatrix']
        # except:
        #     continue

        selected_columns = trial[:, [4, 5, 11, 12, 23]]
        mode_data = np.zeros(selected_columns.shape[0])
        mode_cast = np.zeros(selected_columns.shape[0])
        mode_turn = np.zeros(selected_columns.shape[0])
        mode_accept = np.zeros(selected_columns.shape[0])


        right_cast = np.logical_and(selected_columns[:, 3] == -1, selected_columns[:, 2] == 1)
        left_cast = np.logical_and(selected_columns[:, 3] == 1, selected_columns[:, 2] == 1)

        right_turn = np.logical_and(selected_columns[:, 1] == -1, selected_columns[:, 0] == 1)
        left_turn = np.logical_and(selected_columns[:, 1] == 1, selected_columns[:, 0] == 1)

        cast_only = np.logical_and(selected_columns[:, 0] == 0, selected_columns[:, 2] == 1)
        turn_only = np.logical_and(selected_columns[:, 0] == 1, selected_columns[:, 2] == 0)

        accept = (selected_columns[:, 4] == 1)
        reject = (selected_columns[:, 4] == -1)

        mode_data[np.logical_and(right_cast, cast_only)] = 5
        mode_data[np.logical_and(left_cast, cast_only)] = 2
        mode_data[np.logical_and(right_turn, turn_only)] = 6
        mode_data[np.logical_and(left_turn, turn_only)] = 3

        mode_turn[right_turn] = 6
        mode_turn[left_turn] = 3

        mode_cast[right_cast] = 5
        mode_cast[left_cast] = 2

        mode_accept[accept] = 1
        mode_accept[reject] = -1

        output_csv_path = os.path.join(output_dir, f'trial_{n+1}_pc.csv')

        with open(output_csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Write frame number and mode value
            for index, mode_value in enumerate(mode_data, start=1):
                writer.writerow([index, 0, mode_value, mode_cast[index-1], mode_turn[index-1], mode_accept[index-1]])

        if n % 8 == 0 or n == kin_data.shape[1] - 1:
            print(f"Exported trial {n+1} to {output_csv_path}")