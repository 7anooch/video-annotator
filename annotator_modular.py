import tkinter as tk
import os
import argparse
import traceback
from tkinter import filedialog
from src.core.video_player import VideoPlayer
from src.core.annotation_manager import AnnotationManager
from src.ui.ui_controller import UIController
from src.utils.logger import setup_logger
from src.utils.error_handling import show_error_message
from src.core.config import Config
from src.utils.annotation.funcs import get_csv_file_path

def main():
    """Main function to run the application."""
    logger = setup_logger('video_annotator_main')
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Video Annotation Tool")
        parser.add_argument('--csv', type=str, help="Name of the annotation CSV file")
        parser.add_argument('--side_controls', action='store_true', default=False,
                            help="Place controls on the right side")
        parser.add_argument('--video', type=str, help="Path to the video file")
        parser.add_argument('--config', type=str, default='config.json',
                            help="Path to the configuration file")
        args = parser.parse_args()

        # Load configuration
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Get video path
        if args.video and os.path.exists(args.video):
            video_path = args.video
            logger.info(f"Using video file from command line: {video_path}")
        else:
            video_path = filedialog.askopenfilename(filetypes=[("AVI and MP4 files", "*.avi *.mp4")])
            if not video_path:
                logger.warning("No video file selected. Exiting.")
                return
            logger.info(f"Selected video file: {video_path}")

        # Get annotation file name
        if args.csv:
            annotation_file_name = args.csv
            logger.info(f"Using annotation file from command line: {annotation_file_name}")
        else:
            default_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_annotation.csv"
            print(f"Default csv file name: {default_name}")
            annotation_file_name = input("Enter the name of the annotation file (press Enter to use default): ")
            if not annotation_file_name:
                annotation_file_name = None
                logger.info(f"Using default annotation file name: {default_name}")
            else:
                logger.info(f"Using custom annotation file name: {annotation_file_name}")

        # Create the main window
        root = tk.Tk()
        root.title("Video Annot8er")

        # Get the CSV path
        csv_path = get_csv_file_path(video_path, annotation_file_name)
        logger.info(f"Saving annotations in {csv_path}")
        print(f"Saving annotations in {csv_path}")

        # Create the components
        video_player = VideoPlayer(video_path, config_path=args.config)
        annotation_manager = AnnotationManager(csv_path, config_path=args.config)
        ui_controller = UIController(root, video_player, annotation_manager,
                                    config_path=args.config,
                                    controls_right=args.side_controls)

        # Return the components for testing purposes
        return root, video_player, annotation_manager, ui_controller
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        show_error_message(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    print("\nAvailable keybindings: \n")
    print("Left Arrow: Previous Frame")
    print("Right Arrow: Next Frame")
    print("Spacebar: Play/Pause\n")
    print("S: Annotate as Stop")
    print("R: Annotate as Run")
    print("T: Annotate as Turn\n")

    try:
        result = main()
        if result:
            root, _, _, _ = result
            root.mainloop()
    except Exception as e:
        print(f"Error in main application: {str(e)}")
        traceback.print_exc()
