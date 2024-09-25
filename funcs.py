import os
import pandas as pd
import numpy as np

def save_annotations(annotations, output_csv_path):
    max_frame = max(annotations.keys(), default=0)
    all_frames = list(range(int(max_frame) + 1))
    labels = [annotations.get(frame, np.nan) for frame in all_frames]
    df = pd.DataFrame({'frame': all_frames, 'label': labels})
    
    df.to_csv(output_csv_path, index=False)

def get_csv_file_path(video_path, output_csv_name=None):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
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