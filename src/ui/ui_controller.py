import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import time
import threading
import queue
import os
from src.utils.logger import setup_logger
from src.utils.error_handling import show_error_message, show_warning_message, show_info_message, exception_handler
from src.core.config import Config
from src.ui.theme_manager import ThemeManager
from src.ui.keyboard_manager import KeyboardManager
from src.ui.status_bar import StatusBar
from src.ui.timeline import Timeline
from src.ui.help_dialog import HelpDialog
from src.ui.video_processing_dialog import VideoProcessingDialog
from src.ui.annotation_export_dialog import AnnotationExportDialog
from src.ui.annotation_import_dialog import AnnotationImportDialog

class UIController:
    """
    Manages UI elements and user interactions.

    Attributes:
        master (tk.Tk): The main Tkinter window
        video_player (VideoPlayer): The video player object
        annotation_manager (AnnotationManager): The annotation manager object
    """

    def __init__(self, master, video_player, annotation_manager, config_path='config.json', controls_right=False):
        """
        Initialize the UIController.

        Args:
            master (tk.Tk): The main Tkinter window
            video_player (VideoPlayer): The video player object
            annotation_manager (AnnotationManager): The annotation manager object
            config_path (str, optional): Path to the configuration file. Defaults to 'config.json'.
            controls_right (bool, optional): Whether to place controls on the right side. Defaults to False.
        """
        self.logger = setup_logger('ui_controller')
        self.master = master
        self.video_player = video_player
        self.annotation_manager = annotation_manager

        # Load configuration
        self.config = Config(config_path)
        ui_config = self.config.get_ui_config()
        self.controls_right = controls_right or ui_config.get('controls_right', False)
        self.window_width = ui_config.get('window_width', 1200)
        self.window_height = ui_config.get('window_height', 800)
        self.theme_name = ui_config.get('theme', 'default')
        self.show_timeline = ui_config.get('show_timeline', True)
        self.show_status_bar = ui_config.get('show_status_bar', True)

        # Set window title and icon
        self.master.title("Video Annotator")
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets', 'icon.png')
        if os.path.exists(icon_path):
            try:
                icon = tk.PhotoImage(file=icon_path)
                self.master.iconphoto(True, icon)
            except Exception as e:
                self.logger.warning(f"Failed to load icon: {str(e)}")

        # Set window size
        self.master.geometry(f"{self.window_width}x{self.window_height}")

        # Initialize theme manager
        self.theme_manager = ThemeManager(self.master, self.theme_name)

        # Initialize keyboard manager
        keyboard_shortcuts = self.config.get_keyboard_shortcuts()
        self.keyboard_manager = KeyboardManager(self.master, keyboard_shortcuts)

        # Get labels from config
        self.labels = self.config.get_labels()

        # Set up UI elements
        self.setup_ui()

        # Initialize the frame queue and processing thread
        self.frame_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Set up key bindings
        self.setup_key_bindings()

        # Load the first frame
        self.load_frame(self.video_player.frame_number)

        # Update the annotations listbox
        self.update_annotations_listbox()

    def setup_ui(self):
        """Set up the UI elements."""
        # Create menu bar
        self.setup_menu()

        # Create a frame for the annotations listbox and scrollbar
        self.annotations_frame = ttk.Frame(self.master)
        self.annotations_frame.grid(row=0, column=0, rowspan=4 if self.controls_right else 5, sticky="ns")

        self.annotations_listbox = tk.Listbox(self.annotations_frame, width=20)
        self.annotations_listbox.pack(side="left", fill="y")
        self.annotations_listbox.bind('<<ListboxSelect>>', self.on_annotation_select)

        self.scrollbar = ttk.Scrollbar(self.annotations_frame, orient="vertical",
                                      command=self.annotations_listbox.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.annotations_listbox.config(yscrollcommand=self.scrollbar.set)

        # Create a frame for the video
        self.video_frame = ttk.Frame(self.master)
        self.video_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")

        # Create timeline if enabled
        if self.show_timeline:
            self.timeline_frame = ttk.Frame(self.master)
            self.timeline_frame.grid(row=4, column=1, sticky="ew")
            self.timeline = Timeline(self.timeline_frame, self.video_player, self.annotation_manager, self.theme_manager)
            self.timeline.pack(fill=tk.X, expand=True)

        # Create status bar if enabled
        if self.show_status_bar:
            self.status_bar_frame = ttk.Frame(self.master)
            self.status_bar_frame.grid(row=5, column=0, columnspan=3, sticky="ew")
            self.status_bar = StatusBar(self.status_bar_frame, self.theme_manager)
            self.status_bar.pack(fill=tk.X, expand=True)
            self.status_bar.set_status("Ready")

        self.label = ttk.Label(self.video_frame)
        self.label.pack(fill="both", expand=True)

        # Create a frame for the controls
        self.controls_frame = ttk.Frame(self.master)
        if self.controls_right:
            self.controls_frame.grid(row=0, column=2, rowspan=4, sticky="ns")
        else:
            self.controls_frame.grid(row=4, column=1, sticky="ew")

        # Create a label for the current annotation
        self.annotation_label = ttk.Label(self.controls_frame, text="", font=("Arial", 14, "bold"))
        self.annotation_label.grid(row=0, column=0, columnspan=2 if self.controls_right else 3)

        # Create a progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.controls_frame, variable=self.progress_var,
                                           maximum=self.video_player.total_frames)
        self.progress_bar.grid(row=1 if self.controls_right else 0, column=0,
                              columnspan=2 if self.controls_right else 3, sticky="ew")
        self.progress_bar.bind("<Button-1>", self.on_progress_bar_click)

        # Create playback control buttons
        self.prev_button = tk.Button(self.controls_frame, text="Prev", command=self.prev_frame)
        self.prev_button.grid(row=2 if self.controls_right else 1, column=0)

        self.play_button = tk.Button(self.controls_frame, text="Play ", command=self.toggle_play_pause)
        self.play_button.grid(row=2 if self.controls_right else 1, column=1)

        self.next_button = tk.Button(self.controls_frame, text="Next", command=self.next_frame)
        self.next_button.grid(row=2 if self.controls_right else 1, column=2)

        # Create a frame entry
        self.entry_label = ttk.Label(self.controls_frame, text="Go to Frame:")
        self.entry_label.grid(row=3 if self.controls_right else 0,
                             column= 0 if self.controls_right else 3)

        self.entry = ttk.Entry(self.controls_frame, width=10)
        self.entry.grid(row=3 if self.controls_right else 0,
                       column= 1 if self.controls_right else 4)
        self.entry.bind('<Return>', lambda event: self.go_to_frame())

        self.go_button = tk.Button(self.controls_frame, text="Go", command=self.go_to_frame)
        self.go_button.grid(row=3 if self.controls_right else 0,
                           column= 1 if self.controls_right else 5)

        # Create label buttons from config
        label_frame = ttk.LabelFrame(self.controls_frame, text="Labels")
        if self.controls_right:
            label_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        else:
            label_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Create a button for each label in the configuration
        for i, label in enumerate(self.labels):
            button = tk.Button(label_frame, text=label['name'],
                              command=lambda val=label['value']: self.annotate_frame(val),
                              fg=label['color'])
            row = i // 3
            col = i % 3
            button.grid(row=row, column=col, padx=2, pady=2, sticky="ew")

            # Bind the key to the label
            self.master.bind(label['key'], lambda event, val=label['value']: self.annotate_frame(val))

        # Create speed label and dropdown menu
        self.speed_label = ttk.Label(self.controls_frame, text="Playback Speed (fps)")
        self.speed_label.grid(row=6 if self.controls_right else 3, column=0)

        self.speeds = ["1 fps", "5 fps", "10 fps", "20 fps", "30 fps", "60 fps"]
        self.selected_speed = tk.StringVar(value=self.speeds[4])  # Default to "30 fps"
        self.speed_menu = ttk.Combobox(self.controls_frame, textvariable=self.selected_speed,
                                      values=self.speeds, state="readonly", width=10)
        self.speed_menu.grid(row=6 if self.controls_right else 3, column=1)
        self.speed_menu.bind("<<ComboboxSelected>>",
                            lambda event: self.set_speed(self.selected_speed.get()))

        # Create range labeling controls
        self.range_label_frame = ttk.LabelFrame(self.controls_frame, text="Range Labeling")
        self.range_label_frame.grid(row=7 if self.controls_right else 4,
                                   column=0, columnspan=2 if self.controls_right else 6,
                                   sticky="ew", padx=5, pady=5)

        self.start_frame_label = ttk.Label(self.range_label_frame, text="Start Frame:")
        self.start_frame_label.grid(row=0, column=0, padx=5, pady=2)

        self.start_frame_entry = ttk.Entry(self.range_label_frame, width=10)
        self.start_frame_entry.grid(row=0, column=1, padx=5, pady=2)

        self.end_frame_label = ttk.Label(self.range_label_frame, text="End Frame:")
        self.end_frame_label.grid(row=0, column=2, padx=5, pady=2)

        self.end_frame_entry = ttk.Entry(self.range_label_frame, width=10)
        self.end_frame_entry.grid(row=0, column=3, padx=5, pady=2)

        self.range_label_button = tk.Button(self.range_label_frame,
                                           text="Label Range", command=self.label_range)
        self.range_label_button.grid(row=0, column=4, padx=5, pady=2)

        # Set focus to the main window
        self.master.focus_set()

    def setup_menu(self):
        """Set up the menu bar."""
        # Create menu bar
        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)

        # Create File menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open Video", command=self.open_video, accelerator="Ctrl+O")
        self.file_menu.add_command(label="Save Annotations", command=self.save_annotations, accelerator="Ctrl+S")

        # Create Export/Import submenu
        self.export_import_menu = tk.Menu(self.file_menu, tearoff=0)
        self.file_menu.add_cascade(label="Export/Import", menu=self.export_import_menu)
        self.export_import_menu.add_command(label="Export Annotations", command=self.open_export_dialog)
        self.export_import_menu.add_command(label="Import Annotations", command=self.open_import_dialog)

        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.master.quit, accelerator="Ctrl+Q")

        # Create Edit menu
        self.edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edit", menu=self.edit_menu)
        self.edit_menu.add_command(label="Copy Annotation", command=self.copy_annotation, accelerator="Ctrl+C")
        self.edit_menu.add_command(label="Paste Annotation", command=self.paste_annotation, accelerator="Ctrl+V")
        self.edit_menu.add_command(label="Delete Annotation", command=self.delete_annotation, accelerator="Delete")
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Settings", command=self.open_settings, accelerator="F2")

        # Create View menu
        self.view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="View", menu=self.view_menu)

        # Create Tools menu
        self.tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Tools", menu=self.tools_menu)
        self.tools_menu.add_command(label="Video Processing", command=self.open_video_processing)

        # Create theme submenu
        self.theme_menu = tk.Menu(self.view_menu, tearoff=0)
        self.view_menu.add_cascade(label="Theme", menu=self.theme_menu)

        # Add theme options
        self.theme_var = tk.StringVar(value=self.theme_name)
        for theme in self.theme_manager.get_available_themes():
            self.theme_menu.add_radiobutton(label=theme.capitalize(),
                                          variable=self.theme_var,
                                          value=theme,
                                          command=lambda t=theme: self.change_theme(t))

        # Add view options
        self.timeline_var = tk.BooleanVar(value=self.show_timeline)
        self.view_menu.add_checkbutton(label="Show Timeline",
                                     variable=self.timeline_var,
                                     command=self.toggle_timeline,
                                     accelerator="F9")

        self.status_bar_var = tk.BooleanVar(value=self.show_status_bar)
        self.view_menu.add_checkbutton(label="Show Status Bar",
                                     variable=self.status_bar_var,
                                     command=self.toggle_status_bar,
                                     accelerator="F8")

        self.view_menu.add_separator()
        self.view_menu.add_command(label="Fullscreen",
                                 command=self.toggle_fullscreen,
                                 accelerator="F11")

        # Create Help menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="Keyboard Shortcuts", command=self.show_help, accelerator="F1")
        self.help_menu.add_command(label="About", command=self.show_about)

        # Register keyboard shortcuts with the keyboard manager
        self.register_keyboard_shortcuts()

    def register_keyboard_shortcuts(self):
        """Register keyboard shortcuts with the keyboard manager."""
        # Register navigation shortcuts
        self.keyboard_manager.register_callback('play_pause', self.toggle_play_pause)
        self.keyboard_manager.register_callback('next_frame', self.next_frame)
        self.keyboard_manager.register_callback('prev_frame', self.prev_frame)
        self.keyboard_manager.register_callback('next_10_frames', lambda: self.next_frame(10))
        self.keyboard_manager.register_callback('prev_10_frames', lambda: self.prev_frame(10))

        # Register file shortcuts
        self.keyboard_manager.register_callback('save', self.save_annotations)
        self.keyboard_manager.register_callback('quit', self.master.quit)

        # Register view shortcuts
        self.keyboard_manager.register_callback('toggle_fullscreen', self.toggle_fullscreen)
        self.keyboard_manager.register_callback('toggle_timeline', self.toggle_timeline)
        self.keyboard_manager.register_callback('toggle_status_bar', self.toggle_status_bar)

        # Register help shortcuts
        self.keyboard_manager.register_callback('help', self.show_help)
        self.keyboard_manager.register_callback('settings', self.open_settings)
        self.keyboard_manager.register_callback('toggle_theme', self.cycle_theme)

    def setup_key_bindings(self):
        """Set up key bindings."""
        self.master.bind('<Left>', lambda event: self.prev_frame())
        self.master.bind('<Right>', lambda event: self.next_frame())
        self.master.bind('<space>', lambda event: self.toggle_play_pause())

        # Key bindings for labels are set up in setup_ui when creating the label buttons

    @exception_handler
    def load_frame(self, frame_number):
        """
        Load a frame and display it.

        Args:
            frame_number (int): The frame number to load
        """
        frame = self.video_player.load_frame(frame_number)
        if frame is not None:
            # Update the progress bar
            self.progress_var.set(frame_number)

            # Update the entry
            self.update_entry()

            # Get the annotation for this frame
            annotation_value = self.annotation_manager.get_annotation(frame_number)

            # Get the label from the config
            label = self.config.get_label_by_value(annotation_value)

            if label:
                annotation = label['name']
                color = label['color']
            else:
                annotation = ""
                color = "black"
            self.annotation_label.config(text=annotation, fg=color)

            # Display the frame
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)
            self.label.config(image=image)
            self.label.image = image  # Keep a reference to prevent garbage collection

    @exception_handler
    def update_entry(self):
        """Update the frame entry with the current frame number."""
        self.entry.delete(0, tk.END)
        self.entry.insert(0, f"Frame {self.video_player.frame_number}")

    @exception_handler
    def on_progress_bar_click(self, event):
        """
        Handle clicks on the progress bar.

        Args:
            event: The click event
        """
        # Calculate the frame number based on the click position
        fraction = event.x / self.progress_bar.winfo_width()
        frame_number = int(fraction * self.video_player.total_frames)
        self.go_to_frame(frame_number)

    @exception_handler
    def on_annotation_select(self, event):
        """
        Handle selection in the annotations listbox.

        Args:
            event: The selection event
        """
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            self.go_to_frame(index)

    @exception_handler
    def toggle_play_pause(self):
        """Toggle between play and pause states."""
        if self.video_player.playing:
            self.video_player.playing = False
            self.play_button.config(text="Play ")
        else:
            self.video_player.playing = True
            self.play_button.config(text="Pause")
            self.start_time = time.time()  # Track the start time
            self.play_frame_set()

    @exception_handler
    def play_frame_set(self):
        """Play a set of frames."""
        if self.video_player.playing and self.video_player.frame_number < self.video_player.total_frames:
            # Calculate the frame skip factor
            frame_skip_factor = max(1, self.video_player.fps // 10)
            elapsed_time = time.time() - self.start_time
            expected_frame_number = int(elapsed_time * self.video_player.fps)

            # Skip frames to match the expected frame number
            if expected_frame_number > self.video_player.frame_number:
                self.video_player.frame_number = expected_frame_number

            # Load the frame and put it in the queue for processing
            if self.video_player.frame_number % frame_skip_factor == 0:
                self.load_frame(self.video_player.frame_number)

            self.video_player.frame_number += 1
            if self.video_player.fps < 30:
                self.update_listbox_selection()

            # Calculate the delay for the next frame
            delay = int(1000 / self.video_player.fps)
            self.master.after(delay, self.play_frame_set)

    @exception_handler
    def update_listbox_selection(self, frame_jump=25):
        """
        Update the selection in the annotations listbox.

        Args:
            frame_jump (int, optional): Number of frames to jump ahead for scrolling. Defaults to 25.
        """
        # Calculate the listbox index based on the current frame
        index = int(self.video_player.frame_number)
        self.annotations_listbox.selection_clear(0, tk.END)
        self.annotations_listbox.selection_set(index)
        if index + frame_jump < self.video_player.total_frames:
            self.annotations_listbox.see(index + frame_jump)
        else:
            self.annotations_listbox.see(index)

    @exception_handler
    def process_frames(self):
        """Process frames from the queue."""
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            # Process the frame (e.g., display it)
            self.display_frame(frame)
            self.frame_queue.task_done()

    @exception_handler
    def display_frame(self, frame):
        """
        Display a frame.

        Args:
            frame: The frame to display
        """
        if frame is not None:
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)
            self.label.config(image=image)
            self.label.image = image  # Keep a reference to prevent garbage collection

    @exception_handler
    def prev_frame(self):
        """Go to the previous frame."""
        self.video_player.prev_frame()
        self.load_frame(self.video_player.frame_number)

    @exception_handler
    def next_frame(self):
        """Go to the next frame."""
        self.video_player.next_frame()
        self.load_frame(self.video_player.frame_number)

    @exception_handler
    def go_to_frame(self, frame_number=None):
        """
        Go to a specific frame.

        Args:
            frame_number (int, optional): The frame number to go to. If None, uses the entry field. Defaults to None.
        """
        try:
            if frame_number is None:
                frame_number = int(self.entry.get().replace("Frame ", ""))

            if 0 <= frame_number < self.video_player.total_frames:
                self.video_player.frame_number = frame_number
                self.load_frame(self.video_player.frame_number)
            else:
                show_warning_message(f"Frame number {frame_number} out of range (0-{self.video_player.total_frames-1})")
        except ValueError:
            show_warning_message("Invalid frame number. Please enter a valid number.")

    @exception_handler
    def set_speed(self, speed):
        """
        Set the playback speed.

        Args:
            speed (str): The speed in the format "X fps"
        """
        self.video_player.fps = int(speed.split()[0])
        self.logger.info(f"Playback speed set to: {self.video_player.fps} fps")

    @exception_handler
    def annotate_frame(self, label):
        """
        Annotate the current frame.

        Args:
            label (int): The label to assign to the frame
        """
        self.annotation_manager.annotate_frame(self.video_player.frame_number, label)
        self.annotation_manager.save_annotations()
        self.update_annotations_listbox()
        self.next_frame()  # Automatically go to the next frame

    @exception_handler
    def label_range(self):
        """Label a range of frames."""
        try:
            start_frame = int(self.start_frame_entry.get())
            end_frame = int(self.end_frame_entry.get())

            if start_frame < 0 or end_frame >= self.video_player.total_frames or start_frame > end_frame:
                raise ValueError("Invalid frame range")

            # Create a label selection dialog
            label_dialog = tk.Toplevel(self.master)
            label_dialog.title("Select Label")
            label_dialog.geometry("300x200")
            label_dialog.resizable(False, False)

            # Create a label frame
            label_frame = ttk.LabelFrame(label_dialog, text="Select a label for the range")
            label_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Create radio buttons for each label
            selected_label_var = tk.IntVar(value=0)  # Default to first label

            for i, label in enumerate(self.labels):
                rb = ttk.Radiobutton(label_frame, text=label['name'], variable=selected_label_var,
                                    value=label['value'])
                rb.pack(anchor="w", padx=10, pady=5)

            # Create OK and Cancel buttons
            button_frame = ttk.Frame(label_dialog)
            button_frame.pack(fill="x", padx=10, pady=10)

            cancel_button = ttk.Button(button_frame, text="Cancel",
                                     command=label_dialog.destroy)
            cancel_button.pack(side="right", padx=5)

            # Use a list to store the result (to work around Python's scoping rules)
            result = [None]

            def on_ok():
                result[0] = selected_label_var.get()
                label_dialog.destroy()

            ok_button = ttk.Button(button_frame, text="OK", command=on_ok)
            ok_button.pack(side="right", padx=5)

            # Make the dialog modal
            label_dialog.transient(self.master)
            label_dialog.grab_set()
            self.master.wait_window(label_dialog)

            # If the user clicked Cancel, return
            if result[0] is None:
                return

            # Create a progress window
            progress_window = tk.Toplevel(self.master)
            progress_window.title("Labeling Progress")
            progress_label = tk.Label(progress_window, text="Labeling frames...")
            progress_label.pack(pady=10)
            progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
            progress_bar.pack(pady=10)
            progress_bar['maximum'] = end_frame - start_frame + 1

            # Define a callback function for progress updates
            def update_progress(current, total):
                progress_bar['value'] = current + 1
                progress_window.update()

            # Annotate the range
            self.annotation_manager.annotate_frame_range(start_frame, end_frame, result[0], update_progress)
            self.annotation_manager.save_annotations()
            self.update_annotations_listbox()

            # Close the progress window
            progress_window.destroy()

            # Go to the end frame
            self.go_to_frame(end_frame)
        except ValueError as e:
            show_error_message(f"Error: {str(e)}")

    @exception_handler
    def update_annotations_listbox(self):
        """Update the annotations listbox."""
        scroll_position = self.annotations_listbox.yview()

        self.annotations_listbox.delete(0, tk.END)

        # Create a mapping of label values to colors
        color_mapping = {}
        for label in self.labels:
            color_mapping[label['value']] = label['color']

        for frame in range(self.video_player.total_frames):
            label_value = self.annotation_manager.get_annotation(frame)

            if isinstance(label_value, str):
                try:
                    label_value = float(label_value)
                except ValueError:
                    self.annotations_listbox.insert(tk.END, f"Frame {frame}: Invalid label type")
                    continue

            if label_value is not None and not np.isnan(label_value):
                # Get the label from the config
                label = self.config.get_label_by_value(label_value)
                if label:
                    action = label['name'].lower()
                else:
                    action = f"Label {int(label_value)}"
                self.annotations_listbox.insert(tk.END, f"Frame {frame}: {action}")
            else:
                self.annotations_listbox.insert(tk.END, f"Frame {frame}: NaN")

            # Get the color for this label
            if label_value is not None and not np.isnan(label_value) and int(label_value) in color_mapping:
                color = color_mapping[int(label_value)]
            else:
                color = "black"
            self.annotations_listbox.itemconfig(frame, {'fg': color})

        # Restore the scroll position
        self.annotations_listbox.yview_moveto(scroll_position[0])

    @exception_handler
    def save_annotations(self):
        """Save annotations to a CSV file."""
        self.annotation_manager.save_annotations()
        self.logger.info("Annotations saved")

        # Update status bar if available
        if hasattr(self, 'status_bar'):
            self.status_bar.set_status("Annotations saved")

    @exception_handler
    def open_video(self):
        """Open a video file."""
        # This is a placeholder for the menu command
        # The actual implementation would be in the main application
        pass

    @exception_handler
    def copy_annotation(self):
        """Copy the current annotation."""
        # Get the current frame number
        frame_number = self.video_player.frame_number

        # Get the annotation for the current frame
        annotation = self.annotation_manager.get_annotation(frame_number)

        # Store the annotation in a class variable
        self.copied_annotation = annotation

        # Update status bar if available
        if hasattr(self, 'status_bar'):
            if annotation is not None:
                self.status_bar.set_status(f"Copied annotation: {annotation}")
            else:
                self.status_bar.set_status("No annotation to copy")

    @exception_handler
    def paste_annotation(self):
        """Paste the copied annotation to the current frame."""
        # Check if there is a copied annotation
        if not hasattr(self, 'copied_annotation') or self.copied_annotation is None:
            # Update status bar if available
            if hasattr(self, 'status_bar'):
                self.status_bar.set_status("No annotation to paste")
            return

        # Get the current frame number
        frame_number = self.video_player.frame_number

        # Paste the annotation
        self.annotation_manager.set_annotation(frame_number, self.copied_annotation)

        # Update the UI
        self.update_annotations_listbox()

        # Update status bar if available
        if hasattr(self, 'status_bar'):
            self.status_bar.set_status(f"Pasted annotation: {self.copied_annotation}")

    @exception_handler
    def delete_annotation(self):
        """Delete the annotation for the current frame."""
        # Get the current frame number
        frame_number = self.video_player.frame_number

        # Delete the annotation
        self.annotation_manager.delete_annotation(frame_number)

        # Update the UI
        self.update_annotations_listbox()

        # Update status bar if available
        if hasattr(self, 'status_bar'):
            self.status_bar.set_status(f"Deleted annotation for frame {frame_number}")

    @exception_handler
    def open_settings(self):
        """Open the settings dialog."""
        # Import here to avoid circular imports
        from src.ui.config_editor import ConfigEditor

        # Create the settings dialog
        settings_dialog = ConfigEditor(self.master, self.config.config_path)

        # Wait for the dialog to close
        self.master.wait_window(settings_dialog.master)

        # Reload the configuration
        self.config = Config(self.config.config_path)

        # Update the UI
        self.update_ui_from_config()

    @exception_handler
    def update_ui_from_config(self):
        """Update the UI based on the current configuration."""
        # Get the UI configuration
        ui_config = self.config.get_ui_config()

        # Update theme
        theme_name = ui_config.get('theme', 'default')
        if theme_name != self.theme_name:
            self.change_theme(theme_name)

        # Update timeline visibility
        show_timeline = ui_config.get('show_timeline', True)
        if show_timeline != self.show_timeline:
            self.toggle_timeline()

        # Update status bar visibility
        show_status_bar = ui_config.get('show_status_bar', True)
        if show_status_bar != self.show_status_bar:
            self.toggle_status_bar()

    @exception_handler
    def change_theme(self, theme_name):
        """Change the UI theme."""
        # Update the theme manager
        self.theme_manager.apply_theme(theme_name)

        # Update the theme name
        self.theme_name = theme_name

        # Update the theme variable
        if hasattr(self, 'theme_var'):
            self.theme_var.set(theme_name)

        # Update the UI components
        if hasattr(self, 'timeline'):
            self.timeline.update_theme()

        if hasattr(self, 'status_bar'):
            self.status_bar.update_theme()

        # Update status bar if available
        if hasattr(self, 'status_bar'):
            self.status_bar.set_status(f"Theme changed to {theme_name}")

    @exception_handler
    def cycle_theme(self):
        """Cycle through available themes."""
        # Get available themes
        themes = self.theme_manager.get_available_themes()

        # Find the index of the current theme
        current_index = themes.index(self.theme_name)

        # Calculate the index of the next theme
        next_index = (current_index + 1) % len(themes)

        # Change to the next theme
        self.change_theme(themes[next_index])

    @exception_handler
    def toggle_timeline(self):
        """Toggle the visibility of the timeline."""
        # Toggle the timeline visibility flag
        self.show_timeline = not self.show_timeline

        # Update the timeline variable
        if hasattr(self, 'timeline_var'):
            self.timeline_var.set(self.show_timeline)

        # Show or hide the timeline
        if hasattr(self, 'timeline_frame'):
            if self.show_timeline:
                self.timeline_frame.grid(row=4, column=1, sticky="ew")
            else:
                self.timeline_frame.grid_forget()

        # Update status bar if available
        if hasattr(self, 'status_bar'):
            status = "shown" if self.show_timeline else "hidden"
            self.status_bar.set_status(f"Timeline {status}")

    @exception_handler
    def toggle_status_bar(self):
        """Toggle the visibility of the status bar."""
        # Toggle the status bar visibility flag
        self.show_status_bar = not self.show_status_bar

        # Update the status bar variable
        if hasattr(self, 'status_bar_var'):
            self.status_bar_var.set(self.show_status_bar)

        # Show or hide the status bar
        if hasattr(self, 'status_bar_frame'):
            if self.show_status_bar:
                self.status_bar_frame.grid(row=5, column=0, columnspan=3, sticky="ew")
            else:
                self.status_bar_frame.grid_forget()

    @exception_handler
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        # Get the current state
        is_fullscreen = self.master.attributes('-fullscreen')

        # Toggle fullscreen
        self.master.attributes('-fullscreen', not is_fullscreen)

        # Update status bar if available
        if hasattr(self, 'status_bar'):
            status = "exited" if is_fullscreen else "entered"
            self.status_bar.set_status(f"Fullscreen mode {status}")

    @exception_handler
    def show_help(self):
        """Show the help dialog."""
        # Create the help dialog
        help_dialog = HelpDialog(self.master, self.keyboard_manager, self.theme_manager)

        # Make the dialog modal
        help_dialog.transient(self.master)
        help_dialog.grab_set()

        # Wait for the dialog to close
        self.master.wait_window(help_dialog)

    @exception_handler
    def show_about(self):
        """Show the about dialog."""
        # Create a simple about dialog
        about_text = "Video Annotator\n\nVersion 1.0.0\n\n© 2024 Video Annotator Team"
        messagebox.showinfo("About Video Annotator", about_text)

    @exception_handler
    def open_video_processing(self):
        """Open the video processing dialog."""
        # Get the current video path
        video_path = self.video_player.video_path if hasattr(self.video_player, 'video_path') else None

        # Create the video processing dialog
        video_processing_dialog = VideoProcessingDialog(self.master, video_path, self.theme_manager)

        # Make the dialog modal
        video_processing_dialog.transient(self.master)
        video_processing_dialog.grab_set()

        # Wait for the dialog to close
        self.master.wait_window(video_processing_dialog)

    @exception_handler
    def open_export_dialog(self):
        """Open the annotation export dialog."""
        # Get the annotations from the annotation manager
        annotations = self.annotation_manager.get_annotations()

        # Check if there are any annotations
        if not annotations:
            show_error_message("No annotations to export")
            return

        # Create the export dialog
        export_dialog = AnnotationExportDialog(self.master, annotations, self.theme_manager)

        # Make the dialog modal
        export_dialog.transient(self.master)
        export_dialog.grab_set()

        # Wait for the dialog to close
        self.master.wait_window(export_dialog)

    @exception_handler
    def open_import_dialog(self):
        """Open the annotation import dialog."""
        # Create a callback function to handle imported annotations
        def import_callback(annotations):
            # Set the annotations in the annotation manager
            self.annotation_manager.set_annotations(annotations)

            # Update the UI
            self.update_annotations_listbox()

            # Update status bar if available
            if hasattr(self, 'status_bar'):
                self.status_bar.set_status(f"Imported {len(annotations)} annotations")

        # Create the import dialog
        import_dialog = AnnotationImportDialog(self.master, import_callback, self.theme_manager)

        # Make the dialog modal
        import_dialog.transient(self.master)
        import_dialog.grab_set()

        # Wait for the dialog to close
        self.master.wait_window(import_dialog)
