import os
import pandas as pd
import numpy as np

def annotate_frame(app, label, frame_number):
    global annotations
    annotations[frame_number] = label
    save_annotations(app.video_path, annotations)
    app.update_annotations_listbox()


def save_annotations(video_path, annotations):
    max_frame = max(annotations.keys(), default=0)
    all_frames = list(range(int(max_frame) + 1))
    labels = [annotations.get(frame, np.nan) for frame in all_frames]
    df = pd.DataFrame({'frame': all_frames, 'label': labels})  # Ensure correct column names
    
    # Extract the video file name without the extension
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_file_name = f"{base_name}_annotation.csv"
    
    # Get the directory of the video file
    video_dir = os.path.dirname(video_path)
    
    # Construct the full path for the CSV file
    csv_file_path = os.path.join(video_dir, csv_file_name)
    
    df.to_csv(csv_file_path, index=False)
    print(f"Annotations saved to {csv_file_path}")