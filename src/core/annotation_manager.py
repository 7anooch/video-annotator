import os
import pandas as pd
import numpy as np
from src.utils.logger import setup_logger
from src.utils.error_handling import show_error_message, exception_handler
from src.core.config import Config

class AnnotationManager:
    """
    Manages annotations, saving, and loading.

    Attributes:
        annotation_path (str): Path to the annotation CSV file
        annotations (dict): Dictionary of frame numbers to labels
    """

    def __init__(self, annotation_path, config_path='config.json'):
        """
        Initialize the AnnotationManager.

        Args:
            annotation_path (str): Path to the annotation CSV file
            config_path (str, optional): Path to the configuration file. Defaults to 'config.json'.
        """
        self.logger = setup_logger('annotation_manager')
        self.annotation_path = annotation_path
        self.annotations = {}

        # Load configuration
        self.config = Config(config_path)
        self.annotations_config = self.config.get_annotations_config()
        self.auto_save = self.annotations_config.get('auto_save', True)

        self.load_annotations()

    @exception_handler
    def load_annotations(self):
        """
        Load annotations from the CSV file.

        Returns:
            dict: Dictionary of frame numbers to labels
        """
        if os.path.exists(self.annotation_path):
            try:
                df = pd.read_csv(self.annotation_path)
                self.annotations = {row['frame']: row['label'] for _, row in df.iterrows()}
                self.logger.info(f"Loaded {len(self.annotations)} annotations from {self.annotation_path}")
            except Exception as e:
                self.logger.error(f"Error loading annotations: {str(e)}")
                show_error_message(f"Error loading annotations: {str(e)}")
        else:
            self.logger.info(f"No annotation file found at {self.annotation_path}. Starting with empty annotations.")

        return self.annotations

    @exception_handler
    def save_annotations(self):
        """
        Save annotations to the CSV file.
        """
        try:
            max_frame = max(self.annotations.keys(), default=0)
            all_frames = list(range(int(max_frame) + 1))
            labels = [self.annotations.get(frame, np.nan) for frame in all_frames]
            df = pd.DataFrame({'frame': all_frames, 'label': labels})

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.annotation_path)), exist_ok=True)

            df.to_csv(self.annotation_path, index=False)
            self.logger.info(f"Saved {len(self.annotations)} annotations to {self.annotation_path}")
        except Exception as e:
            self.logger.error(f"Error saving annotations: {str(e)}")
            show_error_message(f"Error saving annotations: {str(e)}")

    @exception_handler
    def annotate_frame(self, frame, label):
        """
        Annotate a single frame.

        Args:
            frame (int): Frame number to annotate
            label (int): Label to assign to the frame

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.annotations[frame] = label
            self.logger.info(f"Annotated frame {frame} with label {label}")
            return True
        except Exception as e:
            self.logger.error(f"Error annotating frame {frame}: {str(e)}")
            return False

    @exception_handler
    def annotate_frame_range(self, start_frame, end_frame, label, callback=None):
        """
        Annotate a range of frames.

        Args:
            start_frame (int): First frame in the range
            end_frame (int): Last frame in the range
            label (int): Label to assign to the frames
            callback (function, optional): Callback function to report progress. Defaults to None.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if start_frame < 0 or end_frame < 0:
                raise ValueError("Frame numbers must be non-negative.")
            if start_frame > end_frame:
                raise ValueError("Start frame must be less than or equal to end frame.")

            for i, frame in enumerate(range(start_frame, end_frame + 1)):
                self.annotations[frame] = label
                if callback and i % 10 == 0:  # Call callback every 10 frames
                    callback(i, end_frame - start_frame + 1)

            self.logger.info(f"Annotated frames {start_frame} to {end_frame} with label {label}")
            return True
        except Exception as e:
            self.logger.error(f"Error annotating frame range {start_frame}-{end_frame}: {str(e)}")
            return False

    @exception_handler
    def get_annotation(self, frame):
        """
        Get the annotation for a specific frame.

        Args:
            frame (int): Frame number

        Returns:
            int or None: The label for the frame, or None if not annotated
        """
        return self.annotations.get(frame, None)

    @exception_handler
    def clear_annotation(self, frame):
        """
        Clear the annotation for a specific frame.

        Args:
            frame (int): Frame number

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if frame in self.annotations:
                del self.annotations[frame]
                self.logger.info(f"Cleared annotation for frame {frame}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error clearing annotation for frame {frame}: {str(e)}")
            return False

    @exception_handler
    def clear_all_annotations(self):
        """
        Clear all annotations.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.annotations.clear()
            self.logger.info("Cleared all annotations")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing all annotations: {str(e)}")
            return False

    @property
    def annotation_count(self):
        """Get the number of annotations."""
        return len(self.annotations)
