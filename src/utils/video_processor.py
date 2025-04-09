#!/usr/bin/env python3
"""
Video processing utilities for the Video Annotator.

This module provides tools for processing video files, including:
- Video format conversion
- Frame extraction
- Video trimming
- Video metadata extraction
"""

import os
import cv2
import numpy as np
import subprocess
from typing import Dict, List, Tuple, Optional, Union, Any
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, ErrorLevel

# Set up logger
logger = setup_logger('video_processor')

class VideoProcessor:
    """
    Provides video processing capabilities.
    
    Attributes:
        supported_formats (List[str]): List of supported video formats
    """
    
    def __init__(self):
        """Initialize the VideoProcessor."""
        self.logger = setup_logger('video_processor')
        
        # List of supported video formats
        self.supported_formats = [
            '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', 
            '.webm', '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2'
        ]
    
    @exception_handler
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a video file.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Dict[str, Any]: Dictionary containing video information
        """
        # Check if the file exists
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return {}
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Could not open video file: {video_path}")
            return {}
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration in seconds
        duration = total_frames / fps if fps > 0 else 0
        
        # Get video codec
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        # Release the video capture
        cap.release()
        
        # Get file size
        file_size = os.path.getsize(video_path)
        
        # Create the info dictionary
        info = {
            'path': video_path,
            'filename': os.path.basename(video_path),
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'codec': codec,
            'file_size': file_size,
            'file_size_mb': file_size / (1024 * 1024)
        }
        
        self.logger.info(f"Video info: {info}")
        return info
    
    @exception_handler
    def extract_frames(self, video_path: str, output_dir: str, 
                      start_frame: int = 0, end_frame: int = None,
                      step: int = 1, format: str = 'jpg',
                      quality: int = 95) -> List[str]:
        """
        Extract frames from a video file.
        
        Args:
            video_path (str): Path to the video file
            output_dir (str): Directory to save the extracted frames
            start_frame (int, optional): First frame to extract. Defaults to 0.
            end_frame (int, optional): Last frame to extract. Defaults to None (last frame).
            step (int, optional): Step between frames. Defaults to 1.
            format (str, optional): Output image format. Defaults to 'jpg'.
            quality (int, optional): Image quality (0-100). Defaults to 95.
            
        Returns:
            List[str]: List of paths to the extracted frames
        """
        # Check if the video file exists
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Could not open video file: {video_path}")
            return []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set end_frame if not specified
        if end_frame is None or end_frame >= total_frames:
            end_frame = total_frames - 1
        
        # Validate frame range
        if start_frame < 0:
            start_frame = 0
        if end_frame >= total_frames:
            end_frame = total_frames - 1
        if start_frame > end_frame:
            self.logger.error(f"Invalid frame range: {start_frame} to {end_frame}")
            cap.release()
            return []
        
        # Set the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        frame_paths = []
        frame_number = start_frame
        
        while frame_number <= end_frame:
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save the frame
            frame_path = os.path.join(output_dir, f"frame_{frame_number:06d}.{format}")
            
            # Set the image quality
            if format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif format.lower() == 'png':
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 10)])
            else:
                cv2.imwrite(frame_path, frame)
            
            frame_paths.append(frame_path)
            
            # Skip frames according to step
            for _ in range(step):
                ret = cap.grab()
                if not ret:
                    break
                frame_number += 1
            
            # Log progress
            if len(frame_paths) % 100 == 0:
                self.logger.info(f"Extracted {len(frame_paths)} frames")
        
        # Release the video capture
        cap.release()
        
        self.logger.info(f"Extracted {len(frame_paths)} frames from {video_path} to {output_dir}")
        return frame_paths
    
    @exception_handler
    def trim_video(self, input_path: str, output_path: str, 
                  start_frame: int, end_frame: int) -> bool:
        """
        Trim a video to a specific frame range.
        
        Args:
            input_path (str): Path to the input video file
            output_path (str): Path to save the output video file
            start_frame (int): First frame to include
            end_frame (int): Last frame to include
            
        Returns:
            bool: True if trimming was successful, False otherwise
        """
        # Check if the input file exists
        if not os.path.exists(input_path):
            self.logger.error(f"Input video file not found: {input_path}")
            return False
        
        # Get input video info
        input_info = self.get_video_info(input_path)
        if not input_info:
            return False
        
        # Validate frame range
        total_frames = input_info['total_frames']
        if start_frame < 0:
            start_frame = 0
        if end_frame >= total_frames:
            end_frame = total_frames - 1
        if start_frame > end_frame:
            self.logger.error(f"Invalid frame range: {start_frame} to {end_frame}")
            return False
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.logger.error(f"Could not open input video: {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create a VideoWriter for the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Set the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames
        frame_count = 0
        total_frames_to_process = end_frame - start_frame + 1
        
        for _ in range(total_frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Write the frame to the output video
            out.write(frame)
            
            frame_count += 1
            
            # Log progress
            if frame_count % 100 == 0:
                self.logger.info(f"Processed {frame_count}/{total_frames_to_process} frames")
        
        # Release resources
        cap.release()
        out.release()
        
        self.logger.info(f"Trimmed video from frame {start_frame} to {end_frame} and saved to {output_path}")
        return True
    
    @exception_handler
    def is_supported_format(self, file_path: str) -> bool:
        """
        Check if a file is in a supported video format.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            bool: True if the file is in a supported format, False otherwise
        """
        # Get the file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        # Check if the extension is in the list of supported formats
        return ext in self.supported_formats
    
    @exception_handler
    def get_supported_formats(self) -> List[str]:
        """
        Get a list of supported video formats.
        
        Returns:
            List[str]: List of supported video formats
        """
        return self.supported_formats.copy()
    
    @exception_handler
    def get_frame(self, video_path: str, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from a video file.
        
        Args:
            video_path (str): Path to the video file
            frame_number (int): Frame number to get
            
        Returns:
            Optional[np.ndarray]: The frame as a NumPy array, or None if the frame could not be read
        """
        # Check if the video file exists
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return None
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Could not open video file: {video_path}")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate frame number
        if frame_number < 0 or frame_number >= total_frames:
            self.logger.error(f"Invalid frame number: {frame_number}")
            cap.release()
            return None
        
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        
        # Release the video capture
        cap.release()
        
        if not ret:
            self.logger.error(f"Could not read frame {frame_number} from {video_path}")
            return None
        
        return frame
