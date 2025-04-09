import cv2
import numpy as np
import os
from src.utils.logger import setup_logger
from src.utils.error_handling import show_error_message, exception_handler
from src.core.config import Config

class VideoPlayer:
    """
    Handles video loading, playback, and frame navigation.

    Attributes:
        video_path (str): Path to the video file
        cap (cv2.VideoCapture): OpenCV video capture object
        total_frames (int): Total number of frames in the video
        frame_number (int): Current frame number
        frame_cache (dict): Cache of loaded frames
        cache_size (int): Maximum number of frames to cache
    """

    def __init__(self, video_path, config_path='config.json'):
        """
        Initialize the VideoPlayer.

        Args:
            video_path (str): Path to the video file
            config_path (str, optional): Path to the configuration file. Defaults to 'config.json'.
        """
        self.logger = setup_logger('video_player')
        self.video_path = video_path
        self.frame_cache = {}

        # Load configuration
        self.config = Config(config_path)
        video_config = self.config.get_video_config()
        self.cache_size = video_config.get('cache_size', 30)
        self.playing = False
        self.frame_number = 0

        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {self.video_path}")
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                # Use default FPS from config
                ui_config = self.config.get_ui_config()
                self.fps = ui_config.get('default_fps', 30)
            self.logger.info(f"Loaded video with {self.total_frames} frames at {self.fps} fps from {self.video_path}")
        except Exception as e:
            self.logger.error(f"Error loading video: {str(e)}")
            raise

    def __del__(self):
        """Release the video capture when the object is deleted."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    @exception_handler
    def load_frame(self, frame_number):
        """
        Load a specific frame from the video.

        Args:
            frame_number (int): The frame number to load

        Returns:
            numpy.ndarray: The loaded frame as an RGB image
        """
        if frame_number < 0 or frame_number >= self.total_frames:
            self.logger.warning(f"Frame number {frame_number} out of range (0-{self.total_frames-1})")
            return None

        # Check if the frame is in the cache
        if frame_number in self.frame_cache:
            self.logger.debug(f"Frame {frame_number} loaded from cache")
            return self.frame_cache[frame_number]

        # Load the frame from the video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if not ret:
            self.logger.error(f"Failed to load frame {frame_number}")
            return None

        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add to cache
        if len(self.frame_cache) >= self.cache_size:
            # Remove the oldest frame from cache
            oldest_frame = min(self.frame_cache.keys())
            del self.frame_cache[oldest_frame]

        self.frame_cache[frame_number] = frame
        self.logger.debug(f"Frame {frame_number} loaded from video")

        return frame

    @exception_handler
    def next_frame(self):
        """
        Move to the next frame.

        Returns:
            numpy.ndarray: The next frame as an RGB image
        """
        if self.frame_number < self.total_frames - 1:
            self.frame_number += 1
            return self.load_frame(self.frame_number)
        else:
            self.logger.info("Reached the end of the video")
            return None

    @exception_handler
    def prev_frame(self):
        """
        Move to the previous frame.

        Returns:
            numpy.ndarray: The previous frame as an RGB image
        """
        if self.frame_number > 0:
            self.frame_number -= 1
            return self.load_frame(self.frame_number)
        else:
            self.logger.info("Already at the first frame")
            return None

    @exception_handler
    def go_to_frame(self, frame_number):
        """
        Go to a specific frame.

        Args:
            frame_number (int): The frame number to go to

        Returns:
            numpy.ndarray: The specified frame as an RGB image
        """
        if 0 <= frame_number < self.total_frames:
            self.frame_number = frame_number
            return self.load_frame(self.frame_number)
        else:
            self.logger.warning(f"Frame number {frame_number} out of range (0-{self.total_frames-1})")
            return None

    @exception_handler
    def resize_frame(self, frame, target_width=1200):
        """
        Resize a frame to a target width while maintaining aspect ratio.

        Args:
            frame (numpy.ndarray): The frame to resize
            target_width (int, optional): The target width. Defaults to 1200.

        Returns:
            numpy.ndarray: The resized frame
        """
        if frame is None:
            return None

        height, width = frame.shape[:2]
        scaling_factor = target_width / float(width)
        return cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    @property
    def current_frame(self):
        """Get the current frame."""
        return self.load_frame(self.frame_number)

    @property
    def current_frame_number(self):
        """Get the current frame number."""
        return self.frame_number
