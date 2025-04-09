#!/usr/bin/env python3
"""
Setup script for Video Annotator.

This script helps with setting up the environment and running the application.
"""

import os
import sys
import subprocess
import argparse
import platform

def check_conda():
    """Check if conda is installed and available."""
    try:
        subprocess.run(['conda', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def create_environment():
    """Create the conda environment from environment.yml."""
    print("Creating conda environment from environment.yml...")
    try:
        subprocess.run(['conda', 'env', 'create', '-f', 'environment.yml'], check=True)
        print("Environment created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating environment: {e}")
        return False

def update_environment():
    """Update the conda environment from environment.yml."""
    print("Updating conda environment from environment.yml...")
    try:
        subprocess.run(['conda', 'env', 'update', '-f', 'environment.yml'], check=True)
        print("Environment updated successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error updating environment: {e}")
        return False

def run_application(app_name):
    """Run the specified application."""
    print(f"Running {app_name}...")
    
    # Determine the correct python executable
    python_cmd = 'python'
    if platform.system() == 'Windows':
        python_cmd = 'python'
    else:
        python_cmd = 'python3'
    
    try:
        subprocess.run([python_cmd, f"{app_name}.py"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running application: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Setup and run Video Annotator')
    parser.add_argument('--create', action='store_true', help='Create conda environment')
    parser.add_argument('--update', action='store_true', help='Update conda environment')
    parser.add_argument('--run', choices=['annotator', 'annotator_modular', 'config_editor', 'plot', 'analyze', 'gen_ground_truth'], 
                        help='Run the specified application')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Check if conda is installed
    if not check_conda():
        print("Error: conda is not installed or not in PATH.")
        print("Please install conda from https://docs.conda.io/en/latest/miniconda.html")
        return
    
    # Create environment if requested
    if args.create:
        if not create_environment():
            return
    
    # Update environment if requested
    if args.update:
        if not update_environment():
            return
    
    # Run application if requested
    if args.run:
        if not run_application(args.run):
            return

if __name__ == '__main__':
    main()
