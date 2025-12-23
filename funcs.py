import os
import pandas as pd
import numpy as np
import cv2

def save_annotations(annotations, output_csv_path, start_frame_offset=0):

    min_frame = min(annotations.keys(), default=start_frame_offset)
    max_frame = max(annotations.keys(), default=start_frame_offset)
    min_frame = max(min_frame, start_frame_offset)

    all_frames = list(range(int(min_frame), int(max_frame) + 1))
    labels = [annotations.get(frame, np.nan) for frame in all_frames]
    df = pd.DataFrame({'frame': all_frames, 'label': labels})
    
    df.to_csv(output_csv_path, index=False)

def get_csv_file_path(video_path, output_csv_name=None, peristalsis_mode=False):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if peristalsis_mode:
        if output_csv_name:
            output_csv_name = output_csv_name.rstrip()
            if not output_csv_name.endswith('.csv'):
                output_csv_name += '.csv'
            csv_file_name = output_csv_name
        else:
            csv_file_name = f"{base_name}_perisannot.csv"
    else:
        if output_csv_name:
            output_csv_name = output_csv_name.rstrip()
            if not output_csv_name.endswith('.csv'):
                output_csv_name += '.csv'
            csv_file_name = output_csv_name
        else:
            csv_file_name = f"{base_name}_annotation.csv"
    
    video_dir = os.path.dirname(video_path)
    csv_file_path = os.path.join(video_dir, csv_file_name)
    
    return csv_file_path

def get_common_substring(strs):
    if not strs:
        return ""
    
    shortest_str = min(strs, key=len)
    
    def is_common_substring(length):
        for i in range(len(shortest_str) - length + 1):
            substr = shortest_str[i:i + length]
            if all(substr in s for s in strs):
                return substr
        return None
    
    low, high = 0, len(shortest_str)
    result = ""
    while low <= high:
        mid = (low + high) // 2
        substr = is_common_substring(mid)
        if substr:
            result = substr
            low = mid + 1
        else:
            high = mid - 1
    
    return result

def format_frames_and_ranges(frames):
    if not frames:
        return ""
    
    frames = sorted(frames)
    ranges = []
    start = frames[0]
    end = frames[0]

    for i in range(1, len(frames)):
        if frames[i] == end + 1:
            end = frames[i]
        else:
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{end}")
            start = frames[i]
            end = frames[i]
    
    if start == end:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{end}")
    
    return ", ".join(ranges)

def resize_frame(frame, target_width=1200):
    height, width = frame.shape[:2]
    scaling_factor = target_width / float(width)
    return cv2.resize(frame, None, fx=scaling_factor, 
                    fy=scaling_factor, interpolation=cv2.INTER_AREA)

def save_peristalsis_annotations(annotations, output_csv_path, start_frame_offset, total_frames, active_label_column='label'):
    """
    Save peristalsis annotations to CSV file.
    Creates a DataFrame with all frames and saves labels to the specified column.
    Handles multiple label columns (label, label1, label2, etc.)
    """
    # Create frame list for all frames
    frames = list(range(start_frame_offset, start_frame_offset + total_frames))
    
    # Load existing CSV if it exists
    if os.path.exists(output_csv_path):
        df = pd.read_csv(output_csv_path)
        # Ensure all frames are present
        existing_frames = set(df['frame'].tolist())
        missing_frames = [f for f in frames if f not in existing_frames]
        if missing_frames:
            # Add missing frames with NaN for all label columns
            missing_df = pd.DataFrame({'frame': missing_frames})
            for col in df.columns:
                if col != 'frame' and col.startswith('label'):
                    missing_df[col] = 0
            df = pd.concat([df, missing_df], ignore_index=True)
            df = df.sort_values('frame').reset_index(drop=True)
    else:
        # Create new DataFrame with all frames
        df = pd.DataFrame({'frame': frames})
    
    # Create or update the active label column
    label_data = []
    for frame in frames:
        label_data.append(annotations.get(frame, 0))
    
    df[active_label_column] = label_data
    
    # Ensure frame column is first
    cols = ['frame'] + [col for col in df.columns if col != 'frame']
    df = df[cols]
    
    df.to_csv(output_csv_path, index=False)
