#!/bin/bash
# Run script for Video Annotator

# Function to check if conda is installed
check_conda() {
    if command -v conda &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check if the environment exists
check_env() {
    conda env list | grep -q "video-annotator"
    return $?
}

# Function to activate the environment
activate_env() {
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate video-annotator
}

# Function to create the environment
create_env() {
    echo "Creating conda environment..."
    conda env create -f environment.yml
}

# Function to update the environment
update_env() {
    echo "Updating conda environment..."
    conda env update -f environment.yml
}

# Main function
main() {
    # Check if conda is installed
    if ! check_conda; then
        echo "Error: conda is not installed or not in PATH."
        echo "Please install conda from https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    # Check if the environment exists
    if ! check_env; then
        echo "Environment 'video-annotator' does not exist."
        read -p "Do you want to create it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            create_env
        else
            echo "Cannot proceed without the environment."
            exit 1
        fi
    fi

    # Activate the environment
    activate_env

    # Show menu
    echo "Video Annotator"
    echo "---------------"
    echo "1. Run Annotator (Modular)"
    echo "2. Run Annotator (Legacy)"
    echo "3. Run Configuration Editor"
    echo "4. Run Plot Tool"
    echo "5. Run Analysis Tool"
    echo "6. Run Ground Truth Generator"
    echo "7. Run Export Tool"
    echo "8. Run Visualization Tool"
    echo "9. Run Performance Profiler"
    echo "10. Run Patch Gaps Tool"
    echo "11. Update Environment"
    echo "12. Exit"
    echo

    # Get user choice
    read -p "Enter your choice: " choice

    # Execute based on choice
    case $choice in
        1)
            python main.py --mode annotator
            ;;
        2)
            python main.py --mode legacy
            ;;
        3)
            python main.py --mode config
            ;;
        4)
            python main.py --mode plot
            ;;
        5)
            python main.py --mode analyze
            ;;
        6)
            python main.py --mode ground_truth
            ;;
        7)
            python main.py --mode export
            ;;
        8)
            python main.py --mode visualize
            ;;
        9)
            python main.py --mode profile
            ;;
        10)
            python main.py --mode patch_gaps
            ;;
        11)
            update_env
            ;;
        12)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice."
            ;;
    esac
}

# Run the main function
main
