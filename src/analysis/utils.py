#!/usr/bin/env python3
"""
Utility functions for annotation analysis that extend existing functionality.

This module provides utility functions for working with annotation data that
complement and extend the functionality in the existing analysis tools.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, ErrorLevel

# Set up logger
logger = setup_logger('analysis_utils')

@exception_handler
def find_annotation_files(directory: str, recursive: bool = False) -> List[str]:
    """
    Find annotation files in a directory.

    Args:
        directory (str): Directory to search
        recursive (bool, optional): Whether to search recursively. Defaults to False.

    Returns:
        List[str]: List of paths to annotation files
    """
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return []

    # Define supported extensions
    supported_extensions = ['.csv', '.json', '.xlsx', '.xls']

    # Find files
    annotation_files = []

    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in supported_extensions:
                    annotation_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            _, ext = os.path.splitext(file)
            if ext.lower() in supported_extensions:
                annotation_files.append(os.path.join(directory, file))

    logger.info(f"Found {len(annotation_files)} annotation files in {directory}")
    return annotation_files

@exception_handler
def get_video_frame_count(video_path: str) -> int:
    """
    Get the number of frames in a video.

    Args:
        video_path (str): Path to the video file

    Returns:
        int: Number of frames in the video
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return 0

    try:
        import cv2

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video was opened successfully
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return 0

        # Get the frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Release the video capture
        cap.release()

        logger.info(f"Video {video_path} has {frame_count} frames")
        return frame_count
    except Exception as e:
        logger.error(f"Error getting video frame count: {str(e)}")
        return 0

@exception_handler
def get_video_fps(video_path: str) -> float:
    """
    Get the frames per second of a video.

    Args:
        video_path (str): Path to the video file

    Returns:
        float: Frames per second of the video
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return 0.0

    try:
        import cv2

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video was opened successfully
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return 0.0

        # Get the FPS
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Release the video capture
        cap.release()

        logger.info(f"Video {video_path} has {fps} FPS")
        return fps
    except Exception as e:
        logger.error(f"Error getting video FPS: {str(e)}")
        return 0.0

@exception_handler
def frames_to_time(frames: int, fps: float) -> str:
    """
    Convert frames to time string (HH:MM:SS.mmm).

    Args:
        frames (int): Number of frames
        fps (float): Frames per second

    Returns:
        str: Time string
    """
    if fps <= 0:
        logger.warning("Invalid FPS value")
        return "00:00:00.000"

    # Calculate time in seconds
    seconds = frames / fps

    # Calculate hours, minutes, seconds, and milliseconds
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)

    # Format the time string
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    return time_str

@exception_handler
def time_to_frames(time_str: str, fps: float) -> int:
    """
    Convert time string (HH:MM:SS.mmm) to frames.

    Args:
        time_str (str): Time string
        fps (float): Frames per second

    Returns:
        int: Number of frames
    """
    if fps <= 0:
        logger.warning("Invalid FPS value")
        return 0

    try:
        # Parse the time string
        if '.' in time_str:
            time_part, ms_part = time_str.split('.')
            milliseconds = int(ms_part)
        else:
            time_part = time_str
            milliseconds = 0

        # Split the time part
        parts = time_part.split(':')

        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = map(int, parts)
        elif len(parts) == 1:
            hours = 0
            minutes = 0
            seconds = int(parts[0])
        else:
            logger.warning(f"Invalid time format: {time_str}")
            return 0

        # Calculate total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

        # Calculate frames
        frames = int(total_seconds * fps)

        return frames
    except Exception as e:
        logger.error(f"Error converting time to frames: {str(e)}")
        return 0

@exception_handler
def interpolate_annotations(annotations: Dict[int, Dict[str, Any]], method: str = 'nearest') -> Dict[int, Dict[str, Any]]:
    """
    Interpolate missing annotations.

    Args:
        annotations (Dict[int, Dict[str, Any]]): Dictionary of annotations with frame numbers as keys
        method (str, optional): Interpolation method ('nearest', 'linear', or 'fill'). Defaults to 'nearest'.

    Returns:
        Dict[int, Dict[str, Any]]: Dictionary of interpolated annotations
    """
    if not annotations:
        logger.warning("No annotations to interpolate")
        return {}

    # Get sorted frame numbers
    frames = sorted(annotations.keys())

    # Find gaps
    gaps = []
    for i in range(1, len(frames)):
        start_frame = frames[i-1]
        end_frame = frames[i]
        gap_size = end_frame - start_frame - 1

        if gap_size > 0:
            gaps.append((start_frame, end_frame, gap_size))

    # If there are no gaps, return the original annotations
    if not gaps:
        logger.info("No gaps found in annotations")
        return annotations.copy()

    # Create a copy of the annotations
    interpolated = annotations.copy()

    # Interpolate gaps
    for start_frame, end_frame, gap_size in gaps:
        start_annotation = annotations[start_frame]
        end_annotation = annotations[end_frame]

        # Check if both annotations have the same label
        if method == 'fill' or (start_annotation.get('label') == end_annotation.get('label')):
            # Fill the gap with the start annotation
            for frame in range(start_frame + 1, end_frame):
                if method == 'nearest':
                    # Use the nearest annotation
                    if frame - start_frame <= end_frame - frame:
                        interpolated[frame] = start_annotation.copy()
                    else:
                        interpolated[frame] = end_annotation.copy()
                elif method == 'linear':
                    # Only interpolate if both annotations have the same label
                    if start_annotation.get('label') == end_annotation.get('label'):
                        interpolated[frame] = start_annotation.copy()
                elif method == 'fill':
                    # Fill with the start annotation
                    interpolated[frame] = start_annotation.copy()

    logger.info(f"Interpolated {sum(gap[2] for gap in gaps)} frames using {method} method")
    return interpolated

@exception_handler
def smooth_annotations(annotations: Dict[int, Dict[str, Any]], window_size: int = 3) -> Dict[int, Dict[str, Any]]:
    """
    Smooth annotations using a sliding window.

    Args:
        annotations (Dict[int, Dict[str, Any]]): Dictionary of annotations with frame numbers as keys
        window_size (int, optional): Size of the sliding window. Defaults to 3.

    Returns:
        Dict[int, Dict[str, Any]]: Dictionary of smoothed annotations
    """
    if not annotations:
        logger.warning("No annotations to smooth")
        return {}

    if window_size < 3 or window_size % 2 == 0:
        logger.warning(f"Invalid window size: {window_size}. Using window size 3.")
        window_size = 3

    # Get sorted frame numbers
    frames = sorted(annotations.keys())

    # If there are fewer frames than the window size, return the original annotations
    if len(frames) < window_size:
        logger.info(f"Too few annotations ({len(frames)}) for smoothing with window size {window_size}")
        return annotations.copy()

    # Create a copy of the annotations
    smoothed = annotations.copy()

    # Smooth annotations
    half_window = window_size // 2

    for i in range(half_window, len(frames) - half_window):
        center_frame = frames[i]
        window_frames = frames[i-half_window:i+half_window+1]

        # Count labels in the window
        label_counts = {}
        for frame in window_frames:
            annotation = annotations[frame]
            if 'label' in annotation:
                label = annotation['label']
                label_counts[label] = label_counts.get(label, 0) + 1

        # Find the most common label
        if label_counts:
            most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]

            # Update the center annotation if it has a different label
            center_annotation = annotations[center_frame]
            if 'label' in center_annotation and center_annotation['label'] != most_common_label:
                smoothed[center_frame] = center_annotation.copy()
                smoothed[center_frame]['label'] = most_common_label

    logger.info(f"Smoothed annotations using window size {window_size}")
    return smoothed

@exception_handler
def calculate_agreement(annotations1: Dict[int, Dict[str, Any]], annotations2: Dict[int, Dict[str, Any]]) -> float:
    """
    Calculate agreement between two sets of annotations.

    Args:
        annotations1 (Dict[int, Dict[str, Any]]): First set of annotations
        annotations2 (Dict[int, Dict[str, Any]]): Second set of annotations

    Returns:
        float: Agreement rate (0-1)
    """
    if not annotations1 or not annotations2:
        logger.warning("Cannot calculate agreement for empty annotation sets")
        return 0.0

    # Get common frames
    frames1 = set(annotations1.keys())
    frames2 = set(annotations2.keys())
    common_frames = frames1.intersection(frames2)

    # If there are no common frames, return 0
    if not common_frames:
        logger.warning("No common frames between annotation sets")
        return 0.0

    # Count agreements
    agreements = 0

    for frame in common_frames:
        annotation1 = annotations1[frame]
        annotation2 = annotations2[frame]

        if 'label' in annotation1 and 'label' in annotation2:
            if annotation1['label'] == annotation2['label']:
                agreements += 1

    # Calculate agreement rate
    agreement_rate = agreements / len(common_frames)

    logger.info(f"Agreement rate: {agreement_rate:.4f} ({agreements}/{len(common_frames)})")
    return agreement_rate

@exception_handler
def calculate_cohen_kappa(annotations1: Dict[int, Dict[str, Any]], annotations2: Dict[int, Dict[str, Any]]) -> float:
    """
    Calculate Cohen's kappa coefficient between two sets of annotations.

    Args:
        annotations1 (Dict[int, Dict[str, Any]]): First set of annotations
        annotations2 (Dict[int, Dict[str, Any]]): Second set of annotations

    Returns:
        float: Cohen's kappa coefficient
    """
    if not annotations1 or not annotations2:
        logger.warning("Cannot calculate Cohen's kappa for empty annotation sets")
        return 0.0

    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        logger.error("scikit-learn is required for calculating Cohen's kappa")
        return 0.0

    # Get common frames
    frames1 = set(annotations1.keys())
    frames2 = set(annotations2.keys())
    common_frames = sorted(frames1.intersection(frames2))

    # If there are no common frames, return 0
    if not common_frames:
        logger.warning("No common frames between annotation sets")
        return 0.0

    # Create arrays for kappa calculation
    y1 = []
    y2 = []

    for frame in common_frames:
        annotation1 = annotations1[frame]
        annotation2 = annotations2[frame]

        if 'label' in annotation1 and 'label' in annotation2:
            y1.append(annotation1['label'])
            y2.append(annotation2['label'])

    # If there are no valid labels, return 0
    if not y1 or not y2:
        logger.warning("No valid labels for kappa calculation")
        return 0.0

    # Calculate Cohen's kappa
    kappa = cohen_kappa_score(y1, y2)

    logger.info(f"Cohen's kappa: {kappa:.4f}")
    return kappa

@exception_handler
def calculate_fleiss_kappa(annotations_list: List[Dict[int, Dict[str, Any]]]) -> float:
    """
    Calculate Fleiss' kappa coefficient for multiple annotators.
    This extends the existing functionality by supporting multiple annotators.

    Args:
        annotations_list (List[Dict[int, Dict[str, Any]]]): List of annotation sets from different annotators

    Returns:
        float: Fleiss' kappa coefficient
    """
    if not annotations_list or len(annotations_list) < 2:
        logger.warning("Fleiss' kappa requires at least 2 annotation sets")
        return 0.0

    try:
        from statsmodels.stats.inter_rater import fleiss_kappa
    except ImportError:
        logger.error("statsmodels is required for calculating Fleiss' kappa")
        return 0.0

    # Get all frames from all annotation sets
    all_frames = set()
    for annotations in annotations_list:
        all_frames.update(annotations.keys())

    # Get all unique labels from all annotation sets
    all_labels = set()
    for annotations in annotations_list:
        for annotation in annotations.values():
            if 'label' in annotation:
                all_labels.add(annotation['label'])

    # Sort frames and labels for consistent ordering
    all_frames = sorted(all_frames)
    all_labels = sorted(all_labels)

    # Create a mapping from labels to indices
    label_to_index = {label: i for i, label in enumerate(all_labels)}

    # Create a matrix for Fleiss' kappa calculation
    # Each row represents a frame, each column represents a label
    # The value in each cell is the number of annotators who assigned that label to that frame
    matrix = np.zeros((len(all_frames), len(all_labels)))

    # Fill the matrix
    for annotations in annotations_list:
        for i, frame in enumerate(all_frames):
            if frame in annotations and 'label' in annotations[frame]:
                label = annotations[frame]['label']
                if label in label_to_index:
                    matrix[i, label_to_index[label]] += 1

    # Calculate Fleiss' kappa
    kappa = fleiss_kappa(matrix)

    logger.info(f"Fleiss' kappa: {kappa:.4f}")
    return kappa

@exception_handler
def calculate_annotation_quality(annotations: Dict[int, Dict[str, Any]], ground_truth: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate various quality metrics for annotations compared to ground truth.
    This extends the existing functionality by providing a comprehensive quality assessment.

    Args:
        annotations (Dict[int, Dict[str, Any]]): Annotations to evaluate
        ground_truth (Dict[int, Dict[str, Any]]): Ground truth annotations

    Returns:
        Dict[str, float]: Dictionary of quality metrics
    """
    if not annotations or not ground_truth:
        logger.warning("Cannot calculate quality metrics for empty annotation sets")
        return {}

    # Get common frames
    frames1 = set(annotations.keys())
    frames2 = set(ground_truth.keys())
    common_frames = frames1.intersection(frames2)

    # Calculate basic metrics
    total_frames = len(frames1.union(frames2))
    common_count = len(common_frames)
    only_in_annotations = len(frames1 - frames2)
    only_in_ground_truth = len(frames2 - frames1)

    # Calculate agreement on common frames
    agreements = 0
    disagreements = 0

    for frame in common_frames:
        annotation = annotations[frame]
        gt_annotation = ground_truth[frame]

        if 'label' in annotation and 'label' in gt_annotation:
            if annotation['label'] == gt_annotation['label']:
                agreements += 1
            else:
                disagreements += 1

    # Calculate metrics
    coverage = common_count / len(frames2) if frames2 else 0
    accuracy = agreements / common_count if common_count > 0 else 0

    # Calculate Cohen's kappa
    kappa = calculate_cohen_kappa(annotations, ground_truth)

    # Calculate precision and recall for each label
    all_labels = set()
    for annotation in ground_truth.values():
        if 'label' in annotation:
            all_labels.add(annotation['label'])

    label_metrics = {}
    for label in all_labels:
        # True positives: frames where both have this label
        tp = sum(1 for frame in common_frames
                if frame in annotations and frame in ground_truth
                and 'label' in annotations[frame] and 'label' in ground_truth[frame]
                and annotations[frame]['label'] == label and ground_truth[frame]['label'] == label)

        # False positives: frames where annotations has this label but ground truth doesn't
        fp = sum(1 for frame in frames1
                if frame in annotations and 'label' in annotations[frame] and annotations[frame]['label'] == label
                and (frame not in ground_truth or 'label' not in ground_truth[frame] or ground_truth[frame]['label'] != label))

        # False negatives: frames where ground truth has this label but annotations doesn't
        fn = sum(1 for frame in frames2
                if frame in ground_truth and 'label' in ground_truth[frame] and ground_truth[frame]['label'] == label
                and (frame not in annotations or 'label' not in annotations[frame] or annotations[frame]['label'] != label))

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        label_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    # Calculate overall metrics
    metrics = {
        'coverage': coverage,
        'accuracy': accuracy,
        'kappa': kappa,
        'missing_frames': only_in_ground_truth,
        'extra_frames': only_in_annotations,
        'label_metrics': label_metrics
    }

    logger.info(f"Calculated annotation quality metrics: coverage={coverage:.4f}, accuracy={accuracy:.4f}, kappa={kappa:.4f}")
    return metrics
