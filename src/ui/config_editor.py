import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
import json
import os
from src.core.config import Config
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler, show_error_message

class ConfigEditor:
    """
    GUI for editing the configuration file.
    """

    def __init__(self, master, config_path='config.json'):
        """
        Initialize the ConfigEditor.

        Args:
            master (tk.Tk): The main Tkinter window
            config_path (str, optional): Path to the configuration file. Defaults to 'config.json'.
        """
        self.logger = setup_logger('config_editor')
        self.master = master
        self.master.title("Video Annotator Configuration Editor")
        self.master.geometry("800x600")

        # Load configuration
        self.config_path = config_path
        self.config = Config(config_path)

        # Create the main notebook
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Create the profiles tab
        self.profiles_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.profiles_frame, text="Profiles")

        # Create the labels tab
        self.labels_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.labels_frame, text="Labels")

        # Create the UI tab
        self.ui_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ui_frame, text="UI")

        # Create the video tab
        self.video_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.video_frame, text="Video")

        # Create the annotations tab
        self.annotations_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.annotations_frame, text="Annotations")

        # Set up the tabs
        self.setup_profiles_tab()
        self.setup_labels_tab()
        self.setup_ui_tab()
        self.setup_video_tab()
        self.setup_annotations_tab()

        # Create the buttons frame
        self.buttons_frame = ttk.Frame(self.master)
        self.buttons_frame.pack(fill="x", padx=10, pady=10)

        # Create the save button
        self.save_button = ttk.Button(self.buttons_frame, text="Save", command=self.save_config)
        self.save_button.pack(side="right", padx=5)

        # Create the cancel button
        self.cancel_button = ttk.Button(self.buttons_frame, text="Cancel", command=self.master.destroy)
        self.cancel_button.pack(side="right", padx=5)

    def setup_profiles_tab(self):
        """Set up the profiles tab."""
        # Create a frame for the profiles list
        self.profiles_list_frame = ttk.LabelFrame(self.profiles_frame, text="Configuration Profiles")
        self.profiles_list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create a listbox for the profiles
        self.profiles_listbox = tk.Listbox(self.profiles_list_frame, height=10)
        self.profiles_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Create a scrollbar for the listbox
        self.profiles_scrollbar = ttk.Scrollbar(self.profiles_list_frame, orient="vertical",
                                              command=self.profiles_listbox.yview)
        self.profiles_scrollbar.pack(side="right", fill="y")
        self.profiles_listbox.configure(yscrollcommand=self.profiles_scrollbar.set)

        # Populate the listbox
        self.populate_profiles_listbox()

        # Create a frame for the profile buttons
        self.profile_buttons_frame = ttk.Frame(self.profiles_frame)
        self.profile_buttons_frame.pack(fill="x", padx=10, pady=5)

        # Create the create button
        self.create_profile_button = ttk.Button(self.profile_buttons_frame, text="Create",
                                             command=self.create_profile)
        self.create_profile_button.pack(side="left", padx=5)

        # Create the switch button
        self.switch_profile_button = ttk.Button(self.profile_buttons_frame, text="Switch",
                                             command=self.switch_profile)
        self.switch_profile_button.pack(side="left", padx=5)

        # Create the delete button
        self.delete_profile_button = ttk.Button(self.profile_buttons_frame, text="Delete",
                                             command=self.delete_profile)
        self.delete_profile_button.pack(side="left", padx=5)

        # Create the import button
        self.import_profile_button = ttk.Button(self.profile_buttons_frame, text="Import",
                                             command=self.import_profile)
        self.import_profile_button.pack(side="left", padx=5)

        # Create the export button
        self.export_profile_button = ttk.Button(self.profile_buttons_frame, text="Export",
                                             command=self.export_profile)
        self.export_profile_button.pack(side="left", padx=5)

        # Create a frame for the current profile
        self.current_profile_frame = ttk.LabelFrame(self.profiles_frame, text="Current Profile")
        self.current_profile_frame.pack(fill="x", padx=10, pady=10)

        # Create a label for the current profile
        self.current_profile_label = ttk.Label(self.current_profile_frame,
                                            text=f"Current Profile: {self.config.current_profile or 'default'}")
        self.current_profile_label.pack(padx=5, pady=5)

    def setup_labels_tab(self):
        """Set up the labels tab."""
        # Create a frame for the labels list
        self.labels_list_frame = ttk.LabelFrame(self.labels_frame, text="Labels")
        self.labels_list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create a treeview for the labels
        self.labels_tree = ttk.Treeview(self.labels_list_frame, columns=("Name", "Key", "Value", "Color"), show="headings")
        self.labels_tree.heading("Name", text="Name")
        self.labels_tree.heading("Key", text="Key")
        self.labels_tree.heading("Value", text="Value")
        self.labels_tree.heading("Color", text="Color")
        self.labels_tree.column("Name", width=150)
        self.labels_tree.column("Key", width=50)
        self.labels_tree.column("Value", width=50)
        self.labels_tree.column("Color", width=100)
        self.labels_tree.pack(fill="both", expand=True, padx=5, pady=5)

        # Create a scrollbar for the treeview
        self.labels_scrollbar = ttk.Scrollbar(self.labels_list_frame, orient="vertical", command=self.labels_tree.yview)
        self.labels_scrollbar.pack(side="right", fill="y")
        self.labels_tree.configure(yscrollcommand=self.labels_scrollbar.set)

        # Create a frame for the label buttons
        self.label_buttons_frame = ttk.Frame(self.labels_frame)
        self.label_buttons_frame.pack(fill="x", padx=10, pady=5)

        # Create the add button
        self.add_label_button = ttk.Button(self.label_buttons_frame, text="Add", command=self.add_label)
        self.add_label_button.pack(side="left", padx=5)

        # Create the edit button
        self.edit_label_button = ttk.Button(self.label_buttons_frame, text="Edit", command=self.edit_label)
        self.edit_label_button.pack(side="left", padx=5)

        # Create the delete button
        self.delete_label_button = ttk.Button(self.label_buttons_frame, text="Delete", command=self.delete_label)
        self.delete_label_button.pack(side="left", padx=5)

        # Populate the treeview
        self.populate_labels_tree()

    def setup_ui_tab(self):
        """Set up the UI tab."""
        # Create a frame for the UI settings
        self.ui_settings_frame = ttk.LabelFrame(self.ui_frame, text="UI Settings")
        self.ui_settings_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Get the UI config
        ui_config = self.config.get_ui_config()

        # Create the controls right checkbox
        self.controls_right_var = tk.BooleanVar(value=ui_config.get('controls_right', False))
        self.controls_right_check = ttk.Checkbutton(self.ui_settings_frame, text="Controls on Right Side",
                                                  variable=self.controls_right_var)
        self.controls_right_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Create the window width entry
        ttk.Label(self.ui_settings_frame, text="Window Width:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.window_width_var = tk.StringVar(value=str(ui_config.get('window_width', 1200)))
        self.window_width_entry = ttk.Entry(self.ui_settings_frame, textvariable=self.window_width_var, width=10)
        self.window_width_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Create the window height entry
        ttk.Label(self.ui_settings_frame, text="Window Height:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.window_height_var = tk.StringVar(value=str(ui_config.get('window_height', 800)))
        self.window_height_entry = ttk.Entry(self.ui_settings_frame, textvariable=self.window_height_var, width=10)
        self.window_height_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Create the default fps entry
        ttk.Label(self.ui_settings_frame, text="Default FPS:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.default_fps_var = tk.StringVar(value=str(ui_config.get('default_fps', 30)))
        self.default_fps_entry = ttk.Entry(self.ui_settings_frame, textvariable=self.default_fps_var, width=10)
        self.default_fps_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

    def setup_video_tab(self):
        """Set up the video tab."""
        # Create a frame for the video settings
        self.video_settings_frame = ttk.LabelFrame(self.video_frame, text="Video Settings")
        self.video_settings_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Get the video config
        video_config = self.config.get_video_config()

        # Create the cache size entry
        ttk.Label(self.video_settings_frame, text="Cache Size:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cache_size_var = tk.StringVar(value=str(video_config.get('cache_size', 30)))
        self.cache_size_entry = ttk.Entry(self.video_settings_frame, textvariable=self.cache_size_var, width=10)
        self.cache_size_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Create the default playback speed entry
        ttk.Label(self.video_settings_frame, text="Default Playback Speed:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.default_playback_speed_var = tk.StringVar(value=str(video_config.get('default_playback_speed', 30)))
        self.default_playback_speed_entry = ttk.Entry(self.video_settings_frame,
                                                    textvariable=self.default_playback_speed_var, width=10)
        self.default_playback_speed_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    def setup_annotations_tab(self):
        """Set up the annotations tab."""
        # Create a frame for the annotations settings
        self.annotations_settings_frame = ttk.LabelFrame(self.annotations_frame, text="Annotations Settings")
        self.annotations_settings_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Get the annotations config
        annotations_config = self.config.get_annotations_config()

        # Create the auto save checkbox
        self.auto_save_var = tk.BooleanVar(value=annotations_config.get('auto_save', True))
        self.auto_save_check = ttk.Checkbutton(self.annotations_settings_frame, text="Auto Save",
                                             variable=self.auto_save_var)
        self.auto_save_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Create the auto advance checkbox
        self.auto_advance_var = tk.BooleanVar(value=annotations_config.get('auto_advance', True))
        self.auto_advance_check = ttk.Checkbutton(self.annotations_settings_frame, text="Auto Advance",
                                                variable=self.auto_advance_var)
        self.auto_advance_check.grid(row=1, column=0, padx=5, pady=5, sticky="w")

    def populate_labels_tree(self):
        """Populate the labels treeview."""
        # Clear the treeview
        for item in self.labels_tree.get_children():
            self.labels_tree.delete(item)

        # Add the labels to the treeview
        for label in self.config.get_labels():
            self.labels_tree.insert("", "end", values=(label['name'], label['key'], label['value'], label['color']))

    @exception_handler
    def add_label(self):
        """Add a new label."""
        # Create a dialog for adding a label
        dialog = tk.Toplevel(self.master)
        dialog.title("Add Label")
        dialog.geometry("300x200")
        dialog.resizable(False, False)

        # Create a frame for the label fields
        label_frame = ttk.LabelFrame(dialog, text="Label")
        label_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create the name entry
        ttk.Label(label_frame, text="Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        name_var = tk.StringVar()
        name_entry = ttk.Entry(label_frame, textvariable=name_var, width=20)
        name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Create the key entry
        ttk.Label(label_frame, text="Key:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        key_var = tk.StringVar()
        key_entry = ttk.Entry(label_frame, textvariable=key_var, width=5)
        key_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Create the value entry
        ttk.Label(label_frame, text="Value:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        value_var = tk.StringVar()
        value_entry = ttk.Entry(label_frame, textvariable=value_var, width=5)
        value_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Create the color entry
        ttk.Label(label_frame, text="Color:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        color_var = tk.StringVar(value="black")
        color_entry = ttk.Entry(label_frame, textvariable=color_var, width=10)
        color_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Create the color button
        color_button = ttk.Button(label_frame, text="Choose Color",
                                command=lambda: self.choose_color(color_var))
        color_button.grid(row=3, column=2, padx=5, pady=5, sticky="w")

        # Create the buttons frame
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(fill="x", padx=10, pady=10)

        # Create the cancel button
        cancel_button = ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy)
        cancel_button.pack(side="right", padx=5)

        # Create the OK button
        def on_ok():
            try:
                # Validate the inputs
                name = name_var.get().strip()
                key = key_var.get().strip()
                value = int(value_var.get().strip())
                color = color_var.get().strip()

                if not name:
                    raise ValueError("Name cannot be empty")
                if not key:
                    raise ValueError("Key cannot be empty")

                # Add the label to the config
                self.config.add_label(name, key, value, color)

                # Refresh the treeview
                self.populate_labels_tree()

                # Close the dialog
                dialog.destroy()
            except ValueError as e:
                show_error_message(str(e), parent=dialog)

        ok_button = ttk.Button(buttons_frame, text="OK", command=on_ok)
        ok_button.pack(side="right", padx=5)

        # Make the dialog modal
        dialog.transient(self.master)
        dialog.grab_set()
        self.master.wait_window(dialog)

    @exception_handler
    def edit_label(self):
        """Edit the selected label."""
        # Get the selected item
        selection = self.labels_tree.selection()
        if not selection:
            show_error_message("No label selected")
            return

        # Get the label values
        values = self.labels_tree.item(selection[0], "values")
        name, key, value, color = values

        # Create a dialog for editing the label
        dialog = tk.Toplevel(self.master)
        dialog.title("Edit Label")
        dialog.geometry("300x200")
        dialog.resizable(False, False)

        # Create a frame for the label fields
        label_frame = ttk.LabelFrame(dialog, text="Label")
        label_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create the name entry
        ttk.Label(label_frame, text="Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        name_var = tk.StringVar(value=name)
        name_entry = ttk.Entry(label_frame, textvariable=name_var, width=20)
        name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Create the key entry
        ttk.Label(label_frame, text="Key:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        key_var = tk.StringVar(value=key)
        key_entry = ttk.Entry(label_frame, textvariable=key_var, width=5)
        key_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Create the value entry
        ttk.Label(label_frame, text="Value:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        value_var = tk.StringVar(value=value)
        value_entry = ttk.Entry(label_frame, textvariable=value_var, width=5)
        value_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Create the color entry
        ttk.Label(label_frame, text="Color:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        color_var = tk.StringVar(value=color)
        color_entry = ttk.Entry(label_frame, textvariable=color_var, width=10)
        color_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Create the color button
        color_button = ttk.Button(label_frame, text="Choose Color",
                                command=lambda: self.choose_color(color_var))
        color_button.grid(row=3, column=2, padx=5, pady=5, sticky="w")

        # Create the buttons frame
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(fill="x", padx=10, pady=10)

        # Create the cancel button
        cancel_button = ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy)
        cancel_button.pack(side="right", padx=5)

        # Create the OK button
        def on_ok():
            try:
                # Validate the inputs
                new_name = name_var.get().strip()
                new_key = key_var.get().strip()
                new_value = int(value_var.get().strip())
                new_color = color_var.get().strip()

                if not new_name:
                    raise ValueError("Name cannot be empty")
                if not new_key:
                    raise ValueError("Key cannot be empty")

                # Remove the old label
                self.config.remove_label(int(value))

                # Add the new label
                self.config.add_label(new_name, new_key, new_value, new_color)

                # Refresh the treeview
                self.populate_labels_tree()

                # Close the dialog
                dialog.destroy()
            except ValueError as e:
                show_error_message(str(e), parent=dialog)

        ok_button = ttk.Button(buttons_frame, text="OK", command=on_ok)
        ok_button.pack(side="right", padx=5)

        # Make the dialog modal
        dialog.transient(self.master)
        dialog.grab_set()
        self.master.wait_window(dialog)

    @exception_handler
    def delete_label(self):
        """Delete the selected label."""
        # Get the selected item
        selection = self.labels_tree.selection()
        if not selection:
            show_error_message("No label selected")
            return

        # Get the label values
        values = self.labels_tree.item(selection[0], "values")
        name, key, value, color = values

        # Confirm deletion
        if not messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the label '{name}'?"):
            return

        # Remove the label from the config
        self.config.remove_label(int(value))

        # Refresh the treeview
        self.populate_labels_tree()

    def populate_profiles_listbox(self):
        """Populate the profiles listbox."""
        # Clear the listbox
        self.profiles_listbox.delete(0, tk.END)

        # Add the profiles to the listbox
        for profile in self.config.profiles:
            self.profiles_listbox.insert(tk.END, profile)

            # Select the current profile
            if profile == (self.config.current_profile or 'default'):
                self.profiles_listbox.selection_set(tk.END)

    @exception_handler
    def create_profile(self):
        """Create a new profile."""
        # Create a dialog for creating a profile
        dialog = tk.Toplevel(self.master)
        dialog.title("Create Profile")
        dialog.geometry("300x150")
        dialog.resizable(False, False)

        # Create a frame for the profile fields
        profile_frame = ttk.LabelFrame(dialog, text="Profile")
        profile_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create the name entry
        ttk.Label(profile_frame, text="Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        name_var = tk.StringVar()
        name_entry = ttk.Entry(profile_frame, textvariable=name_var, width=20)
        name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Create the base profile combobox
        ttk.Label(profile_frame, text="Base Profile:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        base_var = tk.StringVar(value="default")
        base_combo = ttk.Combobox(profile_frame, textvariable=base_var, values=self.config.profiles, width=15)
        base_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Create the buttons frame
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(fill="x", padx=10, pady=10)

        # Create the cancel button
        cancel_button = ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy)
        cancel_button.pack(side="right", padx=5)

        # Create the OK button
        def on_ok():
            try:
                # Validate the inputs
                name = name_var.get().strip()
                base = base_var.get().strip()

                if not name:
                    raise ValueError("Name cannot be empty")

                # Create the profile
                success = self.config.create_profile(name, base)

                if success:
                    # Refresh the listbox
                    self.populate_profiles_listbox()

                    # Update the current profile label
                    self.current_profile_label.config(text=f"Current Profile: {self.config.current_profile or 'default'}")

                    # Close the dialog
                    dialog.destroy()
                else:
                    show_error_message(f"Failed to create profile '{name}'", parent=dialog)
            except ValueError as e:
                show_error_message(str(e), parent=dialog)

        ok_button = ttk.Button(buttons_frame, text="OK", command=on_ok)
        ok_button.pack(side="right", padx=5)

        # Make the dialog modal
        dialog.transient(self.master)
        dialog.grab_set()
        self.master.wait_window(dialog)

    @exception_handler
    def switch_profile(self):
        """Switch to a different profile."""
        # Get the selected profile
        selection = self.profiles_listbox.curselection()
        if not selection:
            show_error_message("No profile selected")
            return

        profile_name = self.profiles_listbox.get(selection[0])

        # Switch to the profile
        success = self.config.switch_profile(profile_name)

        if success:
            # Update the current profile label
            self.current_profile_label.config(text=f"Current Profile: {self.config.current_profile or 'default'}")

            # Refresh the UI to reflect the new profile
            self.refresh_ui()
        else:
            show_error_message(f"Failed to switch to profile '{profile_name}'")

    @exception_handler
    def delete_profile(self):
        """Delete a profile."""
        # Get the selected profile
        selection = self.profiles_listbox.curselection()
        if not selection:
            show_error_message("No profile selected")
            return

        profile_name = self.profiles_listbox.get(selection[0])

        # Confirm deletion
        if not messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the profile '{profile_name}'?"):
            return

        # Delete the profile
        success = self.config.delete_profile(profile_name)

        if success:
            # Refresh the listbox
            self.populate_profiles_listbox()

            # Update the current profile label
            self.current_profile_label.config(text=f"Current Profile: {self.config.current_profile or 'default'}")

            # Refresh the UI to reflect the new profile
            self.refresh_ui()
        else:
            show_error_message(f"Failed to delete profile '{profile_name}'")

    @exception_handler
    def import_profile(self):
        """Import a profile from a file."""
        # Ask for the file to import
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return

        # Ask for the profile name
        dialog = tk.Toplevel(self.master)
        dialog.title("Import Profile")
        dialog.geometry("300x100")
        dialog.resizable(False, False)

        # Create a frame for the profile fields
        profile_frame = ttk.Frame(dialog)
        profile_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create the name entry
        ttk.Label(profile_frame, text="Profile Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        name_var = tk.StringVar(value=os.path.splitext(os.path.basename(file_path))[0])
        name_entry = ttk.Entry(profile_frame, textvariable=name_var, width=20)
        name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Create the buttons frame
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(fill="x", padx=10, pady=10)

        # Create the cancel button
        cancel_button = ttk.Button(buttons_frame, text="Cancel", command=dialog.destroy)
        cancel_button.pack(side="right", padx=5)

        # Create the OK button
        def on_ok():
            try:
                # Validate the inputs
                name = name_var.get().strip()

                if not name:
                    raise ValueError("Name cannot be empty")

                # Import the profile
                success = self.config.import_profile(file_path, name)

                if success:
                    # Refresh the listbox
                    self.populate_profiles_listbox()

                    # Close the dialog
                    dialog.destroy()
                else:
                    show_error_message(f"Failed to import profile from '{file_path}'", parent=dialog)
            except ValueError as e:
                show_error_message(str(e), parent=dialog)

        ok_button = ttk.Button(buttons_frame, text="OK", command=on_ok)
        ok_button.pack(side="right", padx=5)

        # Make the dialog modal
        dialog.transient(self.master)
        dialog.grab_set()
        self.master.wait_window(dialog)

    @exception_handler
    def export_profile(self):
        """Export a profile to a file."""
        # Get the selected profile
        selection = self.profiles_listbox.curselection()
        if not selection:
            show_error_message("No profile selected")
            return

        profile_name = self.profiles_listbox.get(selection[0])

        # Ask for the file to export to
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                               filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return

        # Export the profile
        success = self.config.export_profile(profile_name, file_path)

        if success:
            messagebox.showinfo("Success", f"Profile '{profile_name}' exported to '{file_path}'")
        else:
            show_error_message(f"Failed to export profile '{profile_name}'")

    def refresh_ui(self):
        """Refresh the UI to reflect the current configuration."""
        # Refresh the labels tab
        self.populate_labels_tree()

        # Refresh the UI tab
        ui_config = self.config.get_ui_config()
        self.controls_right_var.set(ui_config.get('controls_right', False))
        self.window_width_var.set(str(ui_config.get('window_width', 1200)))
        self.window_height_var.set(str(ui_config.get('window_height', 800)))
        self.default_fps_var.set(str(ui_config.get('default_fps', 30)))

        # Refresh the video tab
        video_config = self.config.get_video_config()
        self.cache_size_var.set(str(video_config.get('cache_size', 30)))
        self.default_playback_speed_var.set(str(video_config.get('default_playback_speed', 30)))

        # Refresh the annotations tab
        annotations_config = self.config.get_annotations_config()
        self.auto_save_var.set(annotations_config.get('auto_save', True))
        self.auto_advance_var.set(annotations_config.get('auto_advance', True))

    @exception_handler
    def choose_color(self, color_var):
        """Choose a color."""
        color = colorchooser.askcolor(color_var.get())[1]
        if color:
            color_var.set(color)

    @exception_handler
    def save_config(self):
        """Save the configuration."""
        try:
            # Update the UI config
            ui_config = {
                'controls_right': self.controls_right_var.get(),
                'window_width': int(self.window_width_var.get()),
                'window_height': int(self.window_height_var.get()),
                'default_fps': int(self.default_fps_var.get()),
                'theme': self.config.config.get('ui', {}).get('theme', 'default'),
                'font_size': self.config.config.get('ui', {}).get('font_size', 10),
                'show_frame_number': self.config.config.get('ui', {}).get('show_frame_number', True),
                'show_timeline': self.config.config.get('ui', {}).get('show_timeline', True),
                'show_status_bar': self.config.config.get('ui', {}).get('show_status_bar', True)
            }
            self.config.config['ui'] = ui_config

            # Update the video config
            video_config = {
                'cache_size': int(self.cache_size_var.get()),
                'default_playback_speed': int(self.default_playback_speed_var.get()),
                'frame_step': self.config.config.get('video', {}).get('frame_step', 1),
                'auto_play': self.config.config.get('video', {}).get('auto_play', False),
                'loop_playback': self.config.config.get('video', {}).get('loop_playback', False),
                'show_grid': self.config.config.get('video', {}).get('show_grid', False),
                'grid_size': self.config.config.get('video', {}).get('grid_size', 50),
                'grid_color': self.config.config.get('video', {}).get('grid_color', 'gray')
            }
            self.config.config['video'] = video_config

            # Update the annotations config
            annotations_config = {
                'auto_save': self.auto_save_var.get(),
                'auto_advance': self.auto_advance_var.get(),
                'default_csv_name': self.config.config.get('annotations', {}).get('default_csv_name', '{video_name}_annotation.csv'),
                'backup_interval': self.config.config.get('annotations', {}).get('backup_interval', 300),
                'create_backups': self.config.config.get('annotations', {}).get('create_backups', True),
                'max_backups': self.config.config.get('annotations', {}).get('max_backups', 5)
            }
            self.config.config['annotations'] = annotations_config

            # Preserve keyboard shortcuts
            if 'keyboard_shortcuts' not in self.config.config:
                self.config.config['keyboard_shortcuts'] = self.config.default_config['keyboard_shortcuts']

            # Set the profile name
            self.config.config['profile'] = self.config.current_profile or 'default'

            # Save the config
            self.config.save_config()

            # Show a success message
            messagebox.showinfo("Success", "Configuration saved successfully")

            # Close the window
            self.master.destroy()
        except ValueError as e:
            show_error_message(f"Error saving configuration: {str(e)}")

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = ConfigEditor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
