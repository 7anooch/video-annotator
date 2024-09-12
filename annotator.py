import tkinter as tk
from tkinter import ttk, Entry, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import os
import time
import threading
import queue
from funcs import save_annotations, annotate_frame

# Global variables
annotations = {}
frame_counter = 0

class VideoApp:
    def __init__(self, master, video_path):
        self.master = master
        self.master.title("Video Annotation")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Frames in file:", self.total_frames)
        
        self.frame_number = 0
        self.playing = False
        self.fps = 30  # Default fps is 30
        self.screen_width = self.master.winfo_screenwidth()

        # Create a frame for the annotations listbox and scrollbar
        self.annotations_frame = ttk.Frame(self.master)
        self.annotations_frame.grid(row=0, column=0, rowspan=4, sticky="ns")

        self.annotations_listbox = tk.Listbox(self.annotations_frame, width=20)
        self.annotations_listbox.pack(side="left", fill="y")
        self.annotations_listbox.bind('<<ListboxSelect>>', self.on_annotation_select)

        self.scrollbar = ttk.Scrollbar(self.annotations_frame, orient="vertical", command=self.annotations_listbox.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.annotations_listbox.config(yscrollcommand=self.scrollbar.set)

        # Create a frame for the video and controls
        self.video_frame = ttk.Frame(self.master)
        self.video_frame.grid(row=0, column=1, columnspan=7)

        self.label = ttk.Label(self.video_frame)
        self.label.grid(row=0, column=0, columnspan=7)

        self.entry = tk.Entry(self.video_frame)
        self.entry.grid(row=1, column=0)
        self.update_entry()

        self.play_button = ttk.Button(self.video_frame, text="Play", command=self.toggle_play_pause)
        self.play_button.grid(row=1, column=1)
        
        self.prev_button = ttk.Button(self.video_frame, text="Previous Frame", command=self.prev_frame)
        self.prev_button.grid(row=1, column=2)
        
        self.next_button = ttk.Button(self.video_frame, text="Next Frame", command=self.next_frame)
        self.next_button.grid(row=1, column=3)

        # Bind left and right arrow keys to prev_frame
        self.master.bind('<Left>', lambda event: self.prev_frame())
        self.master.bind('<Right>', lambda event: self.next_frame())
        self.master.bind('<space>', lambda event: self.toggle_play_pause())
        
        # Define names for the labels
        label_names = ["Stop", "Run", "Turn"]

        # Create a StringVar to hold the selected label
        self.selected_label = tk.StringVar(value=label_names[0])
        
        # Create label buttons
        for i in range(3):
            button = tk.Button(self.video_frame, text=label_names[i], command=lambda i=i: self.annotate_frame(i))
            button.grid(row=2, column=i)

        # Create radio buttons for selecting the label
        for i, label in enumerate(label_names):
            radio_button = tk.Radiobutton(self.video_frame, text=label, variable=self.selected_label, value=label)
            radio_button.grid(row=4, column=i + 3)

        # Bind keys to annotate_frame method
        self.master.bind('s', lambda event: self.annotate_frame(0))  # Bind 's' key to "Stop"
        self.master.bind('r', lambda event: self.annotate_frame(1))  # Bind 'r' key to "Run"
        self.master.bind('t', lambda event: self.annotate_frame(2))  # Bind 't' key to "Turn"

        # Create speed label and dropdown menu
        self.speed_label = ttk.Label(self.video_frame, text="Playback Speed (fps)")
        self.speed_label.grid(row=3, column=0)
        
        self.speeds = ["1 fps", "5 fps", "10 fps", "20 fps", "30 fps", "60 fps"]
        self.selected_speed = tk.StringVar(value=self.speeds[4])  # Default to "30 fps"
        
        self.speed_menu = tk.OptionMenu(self.video_frame, self.selected_speed, *self.speeds, command=self.set_speed)
        self.speed_menu.grid(row=3, column=1, columnspan=2)

        # Move frame_entry and go_button to rows 1 and 2 on the right
        self.frame_entry = tk.Entry(self.video_frame)
        self.frame_entry.grid(row=1, column=4, columnspan=2)
        self.go_button = ttk.Button(self.video_frame, text="Go to Frame", command=self.go_to_frame)
        self.go_button.grid(row=2, column=4, columnspan=2)

        # Add entries and button for specifying range of frames
        self.start_frame_entry = tk.Entry(self.video_frame)
        self.start_frame_entry.grid(row=3, column=3)
        self.start_frame_entry.insert(0, "Start Frame")

        self.end_frame_entry = tk.Entry(self.video_frame)
        self.end_frame_entry.grid(row=3, column=4)
        self.end_frame_entry.insert(0, "End Frame")

        self.range_label_button = ttk.Button(self.video_frame, text="Label Range", command=self.label_range)
        self.range_label_button.grid(row=3, column=5)

        # Load annotations if CSV file exists
        self.load_annotations()

        # Load the first frame after initializing the progress bar
        self.load_frame(self.frame_number)

        # Initialize the frame queue and processing thread
        self.frame_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.master.mainloop()

    def label_range(self):
        try:
            start_frame = int(self.start_frame_entry.get())
            end_frame = int(self.end_frame_entry.get())
            if start_frame < 0 or end_frame >= self.total_frames or start_frame > end_frame:
                raise ValueError("Invalid frame range")
            selected_label = self.selected_label.get()
            label_mapping = {
                "run": 1,
                "stop": 0,
                "turn": 2
            }

            label = label_mapping.get(selected_label.lower(), np.nan)  # Default to np.nan if the label is not found
            for frame in range(start_frame, end_frame + 1):
                self.annotate_frame(label, frame, save=False)  # Pass save=False to avoid saving in each iteration

            # Save annotations once after labeling the entire range
            save_annotations(self.video_path, annotations)
            self.update_annotations_listbox()
        except ValueError as e:
            print(f"Error: {e}")

    def set_speed(self, speed):
        self.fps = int(speed.split()[0])
        print(f"Playback speed set to: {self.fps} fps")

    def load_annotations(self):
        global annotations
        csv_path = os.path.splitext(self.video_path)[0] + "_annotation.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            annotations = {row['frame']: row['label'] for _, row in df.iterrows()}
            self.update_annotations_listbox()
            print(f"Loaded annotations from {csv_path}")

    def update_entry(self):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, f"Frame {self.frame_number}")

    def resize_frame(self, frame, target_width=1200):
        height, width = frame.shape[:2]
        scaling_factor = target_width / float(width)
        return cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    def load_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            if frame.shape[2] == 3:  # Check if the frame is already in RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            min_width = self.screen_width // 4
            max_width = self.screen_width - 100
            resize_width = (2 * self.screen_width) // 3

            if not (min_width <= frame.shape[1] <= max_width):
                frame = self.resize_frame(frame, resize_width)
            
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)
            self.label.config(image=image)
            self.label.image = image
            self.update_entry()
        else:
            print("Failed to load frame.")

    def toggle_play_pause(self):
        if self.playing:
            self.playing = False
            self.play_button.config(text="Play")
        else:
            self.playing = True
            self.play_button.config(text="Pause")
            self.start_time = time.time()  # Track the start time
            self.play_frame_set()

    def play_frame_set(self):
        if self.playing and self.cap.isOpened():
            # Calculate the frame skip factor
            frame_skip_factor = max(1, self.fps // 10)
            
            # Calculate the expected frame number based on elapsed time and target FPS
            elapsed_time = time.time() - self.start_time
            expected_frame_number = int(elapsed_time * self.fps)
            
            # Skip frames to match the expected frame number
            if expected_frame_number > self.frame_number:
                self.frame_number = expected_frame_number
            
            # Load the frame and put it in the queue for processing
            if self.frame_number % frame_skip_factor == 0:
                frame = self.load_frame(self.frame_number)
                self.frame_queue.put(frame)
            
            self.frame_number += 1
            
            # Calculate the delay for the next frame
            delay = int(1000 / self.fps)
            self.master.after(delay, self.play_frame_set)
        else:
            print("Finished playing video.")

    def process_frames(self):
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            # Process the frame (e.g., display it)
            self.display_frame(frame)
            self.frame_queue.task_done()

    def prev_frame(self):
        self.frame_number = max(0, self.frame_number - 1)
        self.update_entry()
        self.load_frame(self.frame_number)

    def next_frame(self):
        self.frame_number += 1
        self.update_entry()
        self.load_frame(self.frame_number)

    def annotate_frame(self, label, frame=None, save=True):
        global annotations
        if frame is None:
            frame = self.frame_number
        annotations[frame] = label
        if save:
            save_annotations(self.video_path, annotations)
            self.update_annotations_listbox()
        print(f"Annotated frame {frame} with label {label}")
        self.next_frame()  # Automatically go to the next frame

    def update_annotations_listbox(self):
        scroll_position = self.annotations_listbox.yview()

        self.annotations_listbox.delete(0, tk.END)
        color_mapping = {
            0: "red",
            1: "green",
            2: "blue",
        }
        for frame in range(self.total_frames):
            label = annotations.get(frame, np.nan)
            
            if isinstance(label, str):
                try:
                    label = float(label)
                except ValueError:
                    self.annotations_listbox.insert(tk.END, f"Frame {frame}: Invalid label type")
                    continue
            
            if not np.isnan(label):
                if label == 0:
                    action = "stop"
                elif label == 1:
                    action = "run"
                elif label == 2:
                    action = "turn"
                else:
                    action = f"Label {int(label)}"
                self.annotations_listbox.insert(tk.END, f"Frame {frame}: {action}")
            else:
                self.annotations_listbox.insert(tk.END, f"Frame {frame}: NaN")
            
            # Convert label to int for color mapping
            color = color_mapping.get(int(label), "black") if not np.isnan(label) else "black"
            self.annotations_listbox.itemconfig(frame, {'fg': color})
        
            # Restore the scroll position
        self.annotations_listbox.yview_moveto(scroll_position[0])

    def on_annotation_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            self.frame_number = index
            self.load_frame(self.frame_number)

    def go_to_frame(self):
        try:
            frame_number = int(self.frame_entry.get())
            if 0 <= frame_number < self.total_frames:
                self.frame_number = frame_number
                self.load_frame(self.frame_number)
            else:
                print("Frame number out of range.")
        except ValueError:
            print("Invalid frame number.")

def main():
    root = tk.Tk()
    root.title("Video Annotation Tool")

    video_path = tk.filedialog.askopenfilename(filetypes=[("AVI and MP4 files", "*.avi *.mp4")])

    app = VideoApp(root, video_path)

if __name__ == "__main__":
    main()