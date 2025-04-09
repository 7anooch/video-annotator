import json
import os
import logging
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from src.utils.logger import setup_logger
from src.utils.error_handling import exception_handler

class Config:
    """
    Configuration manager for the Video Annotator.

    Handles loading, saving, and accessing configuration settings.
    Supports multiple configuration profiles.

    Attributes:
        config_path (str): Path to the configuration file
        config (Dict[str, Any]): The loaded configuration
        profiles_dir (str): Directory for storing configuration profiles
        current_profile (str): Name of the current profile
        profiles (List[str]): List of available profiles
    """

    def __init__(self, config_path='config.json', profile=None):
        """
        Initialize the Config object.

        Args:
            config_path (str, optional): Path to the configuration file. Defaults to 'config.json'.
            profile (str, optional): Name of the profile to load. Defaults to None (use default profile).
        """
        self.logger = setup_logger('config')
        self.config_path = config_path
        self.profiles_dir = os.path.join(os.path.dirname(os.path.abspath(config_path)), 'profiles')

        # Create profiles directory if it doesn't exist
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir, exist_ok=True)
            self.logger.info(f"Created profiles directory at {self.profiles_dir}")

        self.default_config = {
            'profile': 'default',
            'labels': [
                {'name': 'Stop', 'key': 's', 'value': 0, 'color': 'red'},
                {'name': 'Run', 'key': 'r', 'value': 1, 'color': 'green'},
                {'name': 'Turn', 'key': 't', 'value': 2, 'color': 'blue'}
            ],
            'ui': {
                'controls_right': False,
                'window_width': 1200,
                'window_height': 800,
                'default_fps': 30,
                'theme': 'default',
                'font_size': 10,
                'show_frame_number': True,
                'show_timeline': True,
                'show_status_bar': True
            },
            'video': {
                'cache_size': 30,
                'default_playback_speed': 30,
                'frame_step': 1,
                'auto_play': False,
                'loop_playback': False,
                'show_grid': False,
                'grid_size': 50,
                'grid_color': 'gray'
            },
            'annotations': {
                'auto_save': True,
                'auto_advance': True,
                'default_csv_name': '{video_name}_annotation.csv',
                'backup_interval': 300,  # seconds
                'create_backups': True,
                'max_backups': 5
            },
            'keyboard_shortcuts': {
                'play_pause': 'space',
                'next_frame': 'Right',
                'prev_frame': 'Left',
                'next_10_frames': 'Shift+Right',
                'prev_10_frames': 'Shift+Left',
                'save': 'Control+s',
                'quit': 'Control+q'
            }
        }

        # Get available profiles
        self.profiles = self.get_available_profiles()

        # Set current profile
        self.current_profile = profile

        # Load configuration
        self.config = self.load_config()

    @exception_handler
    def get_available_profiles(self) -> List[str]:
        """
        Get a list of available configuration profiles.

        Returns:
            List[str]: List of profile names
        """
        profiles = ['default']

        if os.path.exists(self.profiles_dir):
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith('.json'):
                    profile_name = os.path.splitext(filename)[0]
                    if profile_name != 'default':  # Default is always available
                        profiles.append(profile_name)

        self.logger.info(f"Found {len(profiles)} configuration profiles: {', '.join(profiles)}")
        return profiles

    @exception_handler
    def get_profile_path(self, profile_name: str) -> str:
        """
        Get the path to a profile configuration file.

        Args:
            profile_name (str): Name of the profile

        Returns:
            str: Path to the profile configuration file
        """
        if profile_name == 'default':
            return self.config_path
        else:
            return os.path.join(self.profiles_dir, f"{profile_name}.json")

    def load_config(self):
        """
        Load the configuration from the file.

        Returns:
            dict: The loaded configuration
        """
        # If a profile is specified, load that profile
        if self.current_profile and self.current_profile in self.profiles:
            profile_path = self.get_profile_path(self.current_profile)
            self.logger.info(f"Loading profile '{self.current_profile}' from {profile_path}")

            if os.path.exists(profile_path):
                try:
                    with open(profile_path, 'r') as f:
                        config = json.load(f)
                    self.logger.info(f"Loaded profile '{self.current_profile}' from {profile_path}")
                    return config
                except Exception as e:
                    self.logger.error(f"Error loading profile '{self.current_profile}': {str(e)}")
                    self.logger.info("Using default configuration")
                    return self.default_config
            else:
                self.logger.info(f"Profile '{self.current_profile}' not found at {profile_path}")
                self.logger.info("Using default configuration")
                return self.default_config

        # Otherwise, load the default configuration
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                self.logger.error(f"Error loading configuration: {str(e)}")
                self.logger.info("Using default configuration")
                return self.default_config
        else:
            self.logger.info(f"No configuration file found at {self.config_path}")
            self.logger.info("Creating default configuration file")
            self.save_config(self.default_config)
            return self.default_config

    @exception_handler
    def save_config(self, config=None, profile=None):
        """
        Save the configuration to the file.

        Args:
            config (Dict[str, Any], optional): The configuration to save. If None, saves the current config. Defaults to None.
            profile (str, optional): The profile to save to. If None, uses the current profile. Defaults to None.
        """
        if config is None:
            config = self.config

        # Determine which profile to save to
        save_profile = profile if profile is not None else self.current_profile

        # If no profile is specified, save to the default config file
        if save_profile is None or save_profile == 'default':
            save_path = self.config_path
        else:
            # Make sure the profiles directory exists
            if not os.path.exists(self.profiles_dir):
                os.makedirs(self.profiles_dir, exist_ok=True)

            save_path = os.path.join(self.profiles_dir, f"{save_profile}.json")

        try:
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=4)
            self.logger.info(f"Saved configuration to {save_path}")

            # If this is a new profile, add it to the list of profiles
            if save_profile not in self.profiles:
                self.profiles.append(save_profile)
                self.logger.info(f"Added new profile: {save_profile}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")

    def get_labels(self):
        """
        Get the labels from the configuration.

        Returns:
            list: The labels
        """
        return self.config.get('labels', self.default_config['labels'])

    def get_ui_config(self):
        """
        Get the UI configuration.

        Returns:
            dict: The UI configuration
        """
        return self.config.get('ui', self.default_config['ui'])

    def get_video_config(self):
        """
        Get the video configuration.

        Returns:
            dict: The video configuration
        """
        return self.config.get('video', self.default_config['video'])

    def get_annotations_config(self):
        """
        Get the annotations configuration.

        Returns:
            dict: The annotations configuration
        """
        return self.config.get('annotations', self.default_config['annotations'])

    def update_config(self, new_config):
        """
        Update the configuration with new values.

        Args:
            new_config (dict): The new configuration values
        """
        self.config.update(new_config)
        self.save_config()

    def update_labels(self, labels):
        """
        Update the labels in the configuration.

        Args:
            labels (list): The new labels
        """
        self.config['labels'] = labels
        self.save_config()

    def add_label(self, name, key, value, color):
        """
        Add a new label to the configuration.

        Args:
            name (str): The name of the label
            key (str): The keyboard shortcut for the label
            value (int): The value of the label
            color (str): The color of the label
        """
        self.config['labels'].append({
            'name': name,
            'key': key,
            'value': value,
            'color': color
        })
        self.save_config()

    def remove_label(self, value):
        """
        Remove a label from the configuration.

        Args:
            value (int): The value of the label to remove
        """
        self.config['labels'] = [label for label in self.config['labels'] if label['value'] != value]
        self.save_config()

    def get_label_by_value(self, value):
        """
        Get a label by its value.

        Args:
            value (int): The value of the label

        Returns:
            dict: The label, or None if not found
        """
        for label in self.get_labels():
            if label['value'] == value:
                return label
        return None

    def get_label_by_key(self, key):
        """
        Get a label by its key.

        Args:
            key (str): The key of the label

        Returns:
            dict: The label, or None if not found
        """
        for label in self.get_labels():
            if label['key'] == key:
                return label
        return None

    @exception_handler
    def create_profile(self, profile_name: str, base_profile: str = 'default') -> bool:
        """
        Create a new configuration profile.

        Args:
            profile_name (str): Name of the new profile
            base_profile (str, optional): Name of the profile to base the new profile on. Defaults to 'default'.

        Returns:
            bool: True if the profile was created successfully, False otherwise
        """
        # Check if the profile already exists
        if profile_name in self.profiles:
            self.logger.warning(f"Profile '{profile_name}' already exists")
            return False

        # Load the base profile
        if base_profile in self.profiles:
            base_path = self.get_profile_path(base_profile)
            if os.path.exists(base_path):
                try:
                    with open(base_path, 'r') as f:
                        base_config = json.load(f)
                except Exception as e:
                    self.logger.error(f"Error loading base profile '{base_profile}': {str(e)}")
                    return False
            else:
                self.logger.warning(f"Base profile '{base_profile}' not found at {base_path}")
                base_config = self.default_config
        else:
            self.logger.warning(f"Base profile '{base_profile}' not found")
            base_config = self.default_config

        # Set the profile name in the config
        base_config['profile'] = profile_name

        # Save the new profile
        self.save_config(base_config, profile_name)

        return True

    @exception_handler
    def delete_profile(self, profile_name: str) -> bool:
        """
        Delete a configuration profile.

        Args:
            profile_name (str): Name of the profile to delete

        Returns:
            bool: True if the profile was deleted successfully, False otherwise
        """
        # Check if the profile exists
        if profile_name not in self.profiles:
            self.logger.warning(f"Profile '{profile_name}' does not exist")
            return False

        # Don't allow deleting the default profile
        if profile_name == 'default':
            self.logger.warning("Cannot delete the default profile")
            return False

        # Delete the profile file
        profile_path = self.get_profile_path(profile_name)
        if os.path.exists(profile_path):
            try:
                os.remove(profile_path)
                self.logger.info(f"Deleted profile '{profile_name}' from {profile_path}")

                # Remove the profile from the list of profiles
                self.profiles.remove(profile_name)

                # If the current profile was deleted, switch to the default profile
                if self.current_profile == profile_name:
                    self.current_profile = 'default'
                    self.config = self.load_config()

                return True
            except Exception as e:
                self.logger.error(f"Error deleting profile '{profile_name}': {str(e)}")
                return False
        else:
            self.logger.warning(f"Profile file '{profile_path}' not found")
            return False

    @exception_handler
    def switch_profile(self, profile_name: str) -> bool:
        """
        Switch to a different configuration profile.

        Args:
            profile_name (str): Name of the profile to switch to

        Returns:
            bool: True if the profile was switched successfully, False otherwise
        """
        # Check if the profile exists
        if profile_name not in self.profiles:
            self.logger.warning(f"Profile '{profile_name}' does not exist")
            return False

        # Switch to the new profile
        self.current_profile = profile_name
        self.config = self.load_config()
        self.logger.info(f"Switched to profile '{profile_name}'")

        return True

    @exception_handler
    def export_profile(self, profile_name: str, export_path: str) -> bool:
        """
        Export a configuration profile to a file.

        Args:
            profile_name (str): Name of the profile to export
            export_path (str): Path to export the profile to

        Returns:
            bool: True if the profile was exported successfully, False otherwise
        """
        # Check if the profile exists
        if profile_name not in self.profiles:
            self.logger.warning(f"Profile '{profile_name}' does not exist")
            return False

        # Get the profile path
        profile_path = self.get_profile_path(profile_name)
        if not os.path.exists(profile_path):
            self.logger.warning(f"Profile file '{profile_path}' not found")
            return False

        # Copy the profile file to the export path
        try:
            shutil.copy2(profile_path, export_path)
            self.logger.info(f"Exported profile '{profile_name}' to {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting profile '{profile_name}': {str(e)}")
            return False

    @exception_handler
    def import_profile(self, import_path: str, profile_name: str = None) -> bool:
        """
        Import a configuration profile from a file.

        Args:
            import_path (str): Path to import the profile from
            profile_name (str, optional): Name to give the imported profile. If None, uses the filename. Defaults to None.

        Returns:
            bool: True if the profile was imported successfully, False otherwise
        """
        # Check if the import file exists
        if not os.path.exists(import_path):
            self.logger.warning(f"Import file '{import_path}' not found")
            return False

        # Determine the profile name
        if profile_name is None:
            profile_name = os.path.splitext(os.path.basename(import_path))[0]

        # Make sure the profiles directory exists
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir, exist_ok=True)

        # Copy the import file to the profiles directory
        profile_path = self.get_profile_path(profile_name)
        try:
            # Load the import file to validate it
            with open(import_path, 'r') as f:
                config = json.load(f)

            # Set the profile name in the config
            config['profile'] = profile_name

            # Save the config to the profile path
            with open(profile_path, 'w') as f:
                json.dump(config, f, indent=4)

            self.logger.info(f"Imported profile '{profile_name}' from {import_path}")

            # Add the profile to the list of profiles if it's not already there
            if profile_name not in self.profiles:
                self.profiles.append(profile_name)

            return True
        except Exception as e:
            self.logger.error(f"Error importing profile '{profile_name}': {str(e)}")
            return False

    @exception_handler
    def get_keyboard_shortcuts(self) -> Dict[str, str]:
        """
        Get the keyboard shortcuts from the configuration.

        Returns:
            Dict[str, str]: The keyboard shortcuts
        """
        return self.config.get('keyboard_shortcuts', self.default_config['keyboard_shortcuts'])

    @exception_handler
    def update_keyboard_shortcuts(self, shortcuts: Dict[str, str]) -> None:
        """
        Update the keyboard shortcuts in the configuration.

        Args:
            shortcuts (Dict[str, str]): The new keyboard shortcuts
        """
        self.config['keyboard_shortcuts'] = shortcuts
        self.save_config()
