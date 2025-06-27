import tkinter as tk
from tkinter import ttk
from tracemalloc import start
import cv2
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import os
import time
import threading
import queue
import argparse
from tkinter import filedialog
from funcs import save_annotations, get_csv_file_path, resize_frame

# Global variables
annotations = {}
frame_counter = 0

class VideoApp:
    def __init__(self, master, video_path, annotation_path, controls_right=False, start_frame_offset=0):
        self.master = master
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        self.master.title(f"Video Annot8er - {video_basename}")
        self.annotation_path = annotation_path
        self.start_frame_offset = start_frame_offset
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Frames in file:", self.total_frames)
        
        self.frame_number = self.start_frame_offset
        self.playing = False
        self.fps = 30  # Default fps is 30
        self.screen_width = self.master.winfo_screenwidth()

        # Create a frame for the annotations listbox and scrollbar
        self.annotations_frame = ttk.Frame(self.master)
        self.annotations_frame.grid(row=0, column=0, rowspan=4 if controls_right else 5, sticky="ns")

        self.annotations_listbox = tk.Listbox(self.annotations_frame, width=20)
        self.annotations_listbox.pack(side="left", fill="y")
        self.annotations_listbox.bind('<<ListboxSelect>>', self.on_annotation_select)

        self.scrollbar = ttk.Scrollbar(self.annotations_frame, orient="vertical", 
                                       command=self.annotations_listbox.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.annotations_listbox.config(yscrollcommand=self.scrollbar.set)

        # Create a frame for the video
        self.video_frame = ttk.Frame(self.master)
        self.video_frame.grid(row=1, column=1, rowspan=2 if controls_right else 1, columnspan=9)

        self.label = ttk.Label(self.video_frame)
        self.label.grid(row=0, column=0, columnspan=9)

        self.frame_slider = tk.Scale(self.video_frame, from_=self.start_frame_offset, 
                                     to=self.start_frame_offset + self.total_frames - 1, 
                                     orient=tk.HORIZONTAL, length=800, command=self.on_slider_move)
        self.frame_slider.grid(row=1, column=0, columnspan=9)

        self.annotation_frame = tk.Frame(self.master)
        self.annotation_frame.grid(row=0, column=1, columnspan=9)
        self.annotation_label = tk.Label(self.annotation_frame, text="", font=("Helvetica", 22))
        self.annotation_label.pack()

        # Create a frame for the controls
        self.controls_frame = ttk.Frame(self.master)
        if controls_right:
            self.controls_frame.grid(row=1, column=10, rowspan=11, 
                                     columnspan=2, sticky="ns")
        else:
            self.controls_frame.grid(row=4, column=1, columnspan=9)

        self.entry = tk.Entry(self.controls_frame, width=12 if controls_right else 20)
        self.entry.grid(row=0, column=0)
        self.update_entry()

        self.play_button = tk.Button(self.controls_frame, text="Play", 
                                     command=self.toggle_play_pause, width=6)
        self.play_button.grid(row=0, column=1)
        
        self.prev_button = tk.Button(self.controls_frame, text="Previous Frame", 
                                     command=self.prev_frame)
        self.prev_button.grid(row=1 if controls_right else 0, 
                              column=0 if controls_right else 2)
        
        self.next_button = tk.Button(self.controls_frame, text="Next Frame", 
                                     command=self.next_frame)
        self.next_button.grid(row= 1 if controls_right else 0, 
                              column= 1 if controls_right else 3)

        # Bind left and right arrow keys to prev_frame
        self.master.bind('<Left>', lambda event: self.prev_frame())
        self.master.bind('<Right>', lambda event: self.next_frame())
        self.master.bind('<space>', lambda event: self.on_spacebar_press())

        self.master.focus_set()

        if controls_right:
            empty_row = 2
            self.controls_frame.grid_rowconfigure(empty_row, minsize=20) 
        
        # Define names for the labels
        label_names = ["Straight", "Left Cast", "Left Turn", "Right Cast", "Right Turn"]

        # Create a StringVar to hold the selected label
        self.selected_label = tk.StringVar(value=label_names[0])
        
        # Create label buttons
        for i in range(5):
            button = tk.Button(self.controls_frame, text=label_names[i], 
                               command=lambda i=i: self.annotate_frame([0,2,3,5,6][i]))
            if controls_right:
                row = 3 + (i // 2)
                col = i % 2
                button.grid(row=row, column=col, padx=1, pady=1)
            else:
                button.grid(row=1 , column=i , padx=1, pady=1)

        if controls_right:
            empty_row = 6
            self.controls_frame.grid_rowconfigure(empty_row, minsize=20)
            self.controls_frame.grid_rowconfigure(7, minsize=20) 

        # Create radio buttons for selecting the label
        for i, label in enumerate(label_names):
            radio_button = tk.Radiobutton(self.controls_frame, text=label,
                                           variable=self.selected_label, value=label)
            
            if controls_right:
                row = 11 + (i // 2)
                col = i % 2
                if col == 1:
                    row += 1
                radio_button.grid(row=row, column=col)
            else:
                if i < 3:
                    radio_button.grid(row=3, column=i + 3)
                else:
                    radio_button.grid(row=4, column=i-2 + 3)

        # Bind keys to annotate_frame method
        self.master.bind('s', lambda event: self.annotate_frame(0))  # Bind 's' key to "Straight"
        self.master.bind('a', lambda event: self.annotate_frame(2))  # Bind 'q' key to "Left Cast"
        self.master.bind('z', lambda event: self.annotate_frame(3))  # Bind 'a' key to "Left Turn"
        self.master.bind('d', lambda event: self.annotate_frame(5))  # Bind 'e' key to "Right Cast"
        self.master.bind('x', lambda event: self.annotate_frame(6))  # Bind 'd' key to "Right Turn"


        # Create speed label and dropdown menu
        self.speed_label = ttk.Label(self.controls_frame, text="Playback Speed (fps)")
        self.speed_label.grid(row=6 if controls_right else 2, column=0)
        
        self.speeds = ["1 fps", "5 fps", "10 fps", "20 fps", "30 fps", "60 fps"]
        self.selected_speed = tk.StringVar(value=self.speeds[4])  # Default to "30 fps"
        
        self.speed_menu = tk.OptionMenu(self.controls_frame, self.selected_speed, *self.speeds, command=self.set_speed)
        self.speed_menu.grid(row= 6 if controls_right else 2, column=1, 
                             columnspan= 1 if controls_right else 2)

        # Move frame_entry and go_button to rows 1 and 2 on the right
        self.frame_entry = tk.Entry(self.controls_frame, width=12 if controls_right else 20)
        self.frame_entry.grid(row=7 if controls_right else 0, 
                              column= 0 if controls_right else 4, 
                              columnspan=1 if controls_right else 2)
        self.frame_entry.bind('<Return>', lambda event: self.go_to_frame())
        self.go_button = tk.Button(self.controls_frame, text="Go to Frame", 
                                   command=self.go_to_frame)
        self.go_button.grid(row=7 if controls_right else 0, 
                            column=1 if controls_right else 5, columnspan=1)
        
        if controls_right:
            empty_row = 8
            self.controls_frame.grid_rowconfigure(empty_row, minsize=30) 

        # Add entries and button for specifying range of frames
        self.start_frame_entry = tk.Entry(self.controls_frame, width=12 if controls_right else 20)
        self.start_frame_entry.grid(row=9 if controls_right else 2,
                                    column=0 if controls_right else 3)
        self.start_frame_entry.insert(0, "Start Frame")

        self.end_frame_entry = tk.Entry(self.controls_frame, width=12 if controls_right else 20)
        self.end_frame_entry.grid(row=9 if controls_right else 2, 
                                  column=1 if controls_right else 4)
        self.end_frame_entry.insert(0, "End Frame")

        self.range_label_button = tk.Button(self.controls_frame,
                                             text="Label Range", command=self.label_range)
        self.range_label_button.grid(row=14 if controls_right else 2, 
                                     column=0 if controls_right else 5, 
                                     columnspan=2 if controls_right else 1)

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

    def on_spacebar_press(self):
        self.toggle_play_pause()

    def on_slider_move(self, value):
        self.frame_number = int(value)
        self.load_frame(self.frame_number)

    def label_range(self):
        try:
            start_frame = int(self.start_frame_entry.get())
            end_frame = int(self.end_frame_entry.get())
            if start_frame < self.start_frame_offset or end_frame >= self.start_frame_offset + self.total_frames or start_frame > end_frame:
                raise ValueError("Invalid frame range")
            selected_label = self.selected_label.get()
            label_mapping = {
                "straight": 0,
                "left cast": 2,
                "left turn": 3,
                "right cast": 5,
                "right turn": 6
            }

            label = label_mapping.get(selected_label.lower(), np.nan)  # Default to np.nan if the label is not found
            self.annotate_frame_range(label, start_frame=start_frame, 
                                      end_frame=end_frame, save=True)

            # Save annotations once after labeling the entire range
            # save_annotations(self.video_path, annotations)
            self.go_to_frame(end_frame)
            self.update_annotations_listbox()
        except ValueError as e:
            print(f"Error: {e}")

    def set_speed(self, speed):
        self.fps = int(speed.split()[0])
        print(f"Playback speed set to: {self.fps} fps")

    def load_annotations(self):
        global annotations
        csv_path = self.annotation_path
        # csv_path = os.path.splitext(self.video_path)[0] + "_annotation.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            annotations = {row['frame']: row['label'] for _, row in df.iterrows()}
            self.update_annotations_listbox()
            print(f"Loaded annotations from {csv_path}")

    def update_entry(self):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, f"Frame {self.frame_number}")

    def load_frame(self, frame_number):
        actual_frame_number = frame_number - self.start_frame_offset
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_number)
        ret, frame = self.cap.read()
        if ret:
            if frame.shape[2] == 3:  # Check if the frame is already in RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            min_width = self.screen_width // 4
            max_width = self.screen_width - 100
            resize_width = (2 * self.screen_width) // 3

            if not (min_width <= frame.shape[1] <= max_width):
                frame = resize_frame(frame, resize_width)

            annotation_value = annotations.get(frame_number, "")
            annotation_mapping = {
                0: "Straight",
                2: "Left Cast",
                3: "Left Turn",
                5: "Right Cast",
                6: "Right Turn"
            }
            color_mapping = {
                "Straight": "green",
                "Left Cast": "red",
                "Left Turn": "blue",
                "Right Cast": "red",
                "Right Turn": "blue"
            }
            annotation = annotation_mapping.get(annotation_value, "")
            color = color_mapping.get(annotation, "black")
            self.annotation_label.config(text=annotation, fg=color)
            
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
            self.play_button.config(text="Play ")
        else:
            self.playing = True
            self.play_button.config(text="Pause")
            self.start_time = time.time()  # Track the start time
            self.play_frame_set()

    def play_frame_set(self):
        if self.playing and self.cap.isOpened() and \
            self.frame_number < self.start_frame_offset + self.total_frames:
            # Calculate the frame skip factor
            frame_skip_factor = max(1, self.fps // 10)
            elapsed_time = time.time() - self.start_time
            expected_frame_number = int(elapsed_time * self.fps) + self.start_frame_offset
            
            # Skip frames to match the expected frame number
            if expected_frame_number > self.frame_number:
                self.frame_number = expected_frame_number
            
            # Load the frame and put it in the queue for processing
            if self.frame_number % frame_skip_factor == 0:
                frame = self.load_frame(self.frame_number)
                self.frame_queue.put(frame)
            
            self.frame_number += 1
            if self.fps < 30:
                self.update_listbox_selection()
            
            # Calculate the delay for the next frame
            delay = int(1000 / self.fps)
            self.master.after(delay, self.play_frame_set)
        # else:
        #     print("Finished playing video.")

    def update_listbox_selection(self, frame_jump=25):
        # Calculate the listbox index based on the current frame
        index = int(self.frame_number - self.start_frame_offset)
        self.annotations_listbox.selection_clear(0, tk.END)
        self.annotations_listbox.selection_set(index)
        if index+frame_jump < self.total_frames:
            self.annotations_listbox.see(index+frame_jump)
        else:
            self.annotations_listbox.see(index)

    def process_frames(self):
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            # Process the frame (e.g., display it)
            self.display_frame(frame)
            self.frame_queue.task_done()

    def prev_frame(self):
        self.frame_number = max(self.start_frame_offset, self.frame_number - 1)
        self.update_entry()
        self.load_frame(self.frame_number)

    def next_frame(self):
        if self.frame_number < self.start_frame_offset + self.total_frames - 1:
            self.frame_number += 1
        self.update_entry()
        self.load_frame(self.frame_number)

    def annotate_frame(self, label, frame=None, save=True):
        global annotations
        if frame is None:
            frame = self.frame_number
        annotations[frame] = label
        if save:
            save_annotations(annotations, self.annotation_path)
            self.update_annotations_listbox()
        print(f"Annotated frame {frame} with label {label}")
        self.next_frame()  # Automatically go to the next frame

    def annotate_frame_range(self, label, start_frame=None, end_frame=None, save=True):
        global annotations
        if start_frame is None:
            start_frame = self.frame_number
        if end_frame is None:
            end_frame = self.frame_number

        # Update validation to account for offset
        if start_frame < self.start_frame_offset or end_frame < self.start_frame_offset:
            raise ValueError(f"Frame numbers must be >= {self.start_frame_offset}.")
        if start_frame > end_frame:
            raise ValueError("start frame must be less than or equal to end frame.")
        if start_frame >= self.start_frame_offset + self.total_frames or end_frame >= self.start_frame_offset + self.total_frames:
            raise ValueError(f"Frame numbers must be less than {self.start_frame_offset + self.total_frames}.")
    
        for frame in range(start_frame, end_frame + 1):
            annotations[frame] = label
    
        if save:
            save_annotations(annotations, self.annotation_path)
            self.update_annotations_listbox()
    
        print(f"Annotated frames {start_frame} to {end_frame} with label {label}")
        self.frame_number = end_frame + 1  # Automatically go to the frame after the last annotated frame
        self.update_entry()
        self.load_frame(self.frame_number)

    def update_annotations_listbox(self):
        scroll_position = self.annotations_listbox.yview()

        self.annotations_listbox.delete(0, tk.END)
        color_mapping = {
            0: "green",
            2: "red",
            3: "blue",
            5: "red",
            6: "blue",
        }
        for frame in range(self.start_frame_offset, self.start_frame_offset + self.total_frames):
            label = annotations.get(frame, np.nan)
            
            if isinstance(label, str):
                try:
                    label = float(label)
                except ValueError:
                    self.annotations_listbox.insert(tk.END, f"Frame {frame}: Invalid label type")
                    continue
            
            if not np.isnan(label):
                if label == 0:
                    action = "straight"
                elif label == 2:
                    action = "left cast"
                elif label == 3:
                    action = "left turn"
                elif label == 5:
                    action = "right cast"
                elif label == 6:
                    action = "right turn"
                else:
                    action = f"Label {int(label)}"
                self.annotations_listbox.insert(tk.END, f"Frame {frame}: {action}")
            else:
                self.annotations_listbox.insert(tk.END, f"Frame {frame}: NaN")
            
            # Convert label to int for color mapping
            color = color_mapping.get(int(label), "black") if not np.isnan(label) else "black"
            self.annotations_listbox.itemconfig(frame - self.start_frame_offset, {'fg': color})
        
            # Restore the scroll position
        self.annotations_listbox.yview_moveto(scroll_position[0])

    def on_annotation_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            self.frame_number = index + self.start_frame_offset
            self.load_frame(self.frame_number)

    def go_to_frame(self, last_in_range=None):
        try:
            if last_in_range:
                frame_number = int(last_in_range)
            else:
                frame_number = int(self.frame_entry.get())
            if self.start_frame_offset <= frame_number < self.start_frame_offset + self.total_frames:
                self.frame_number = frame_number
                self.load_frame(self.frame_number)
            else:
                print("Frame number out of range.")
        except ValueError:
            print("Invalid frame number.")

def main():
    video_path = filedialog.askopenfilename(filetypes=[("AVI and MP4 files", "*.avi *.mp4")])
    parser = argparse.ArgumentParser(description="Video Annotation Tool")
    parser.add_argument('--csv', type=str, help="Name of the annotation CSV file")
    parser.add_argument('--side_controls', action='store_true', default=False,
                        help="Place controls on the right side")
    parser.add_argument('--start', type=int, default=0,
                        help="Starting frame number (default: 0)")
    args = parser.parse_args()

    if args.csv:
        annotation_file_name = args.csv
    else:
        print(f"Default csv file name: {os.path.splitext(os.path.basename(video_path))[0]}_annotation.csv")
        annotation_file_name = input("Enter the name of the annotation file (press Enter to use default): ")
        if not annotation_file_name:
            annotation_file_name = None 

    root = tk.Tk()
    root.title("Video Annot8er")

    csv_path = get_csv_file_path(video_path, annotation_file_name)
    print(f"Saving annotations in {csv_path}")
    print(f"Frame numbering starts at: {args.start}")
    app = VideoApp(root, video_path, csv_path, controls_right=args.side_controls, start_frame_offset=args.start)

if __name__ == "__main__":
    print("\nAvailable keybindings: \n")
    print("Left Arrow: Previous Frame")
    print("Right Arrow: Next Frame")
    print("Spacebar: Play/Pause\n")
    print("S: Annotate as Straight")
    print("A: Annotate as Left Cast")
    print("Z: Annotate as Left Turn")
    print("D: Annotate as Right Cast")
    print("X: Annotate as Right Turn\n")

    main()