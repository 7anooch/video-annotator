@echo off
REM Run script for Video Annotator on Windows

REM Function to check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda is not installed or not in PATH.
    echo Please install conda from https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM Check if the environment exists
conda env list | findstr "video-annotator" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Environment 'video-annotator' does not exist.
    set /p REPLY=Do you want to create it? (y/n)
    if /i "%REPLY%"=="y" (
        echo Creating conda environment...
        conda env create -f environment.yml
    ) else (
        echo Cannot proceed without the environment.
        exit /b 1
    )
)

REM Activate the environment
call conda activate video-annotator

REM Show menu
echo Video Annotator
echo ---------------
echo 1. Run Annotator (Modular)
echo 2. Run Annotator (Legacy)
echo 3. Run Configuration Editor
echo 4. Run Plot Tool
echo 5. Run Analysis Tool
echo 6. Run Ground Truth Generator
echo 7. Run Export Tool
echo 8. Run Visualization Tool
echo 9. Run Performance Profiler
echo 10. Run Patch Gaps Tool
echo 11. Update Environment
echo 12. Exit
echo.

REM Get user choice
set /p CHOICE=Enter your choice:

REM Execute based on choice
if "%CHOICE%"=="1" (
    python main.py --mode annotator
) else if "%CHOICE%"=="2" (
    python main.py --mode legacy
) else if "%CHOICE%"=="3" (
    python main.py --mode config
) else if "%CHOICE%"=="4" (
    python main.py --mode plot
) else if "%CHOICE%"=="5" (
    python main.py --mode analyze
) else if "%CHOICE%"=="6" (
    python main.py --mode ground_truth
) else if "%CHOICE%"=="7" (
    python main.py --mode export
) else if "%CHOICE%"=="8" (
    python main.py --mode visualize
) else if "%CHOICE%"=="9" (
    python main.py --mode profile
) else if "%CHOICE%"=="10" (
    python main.py --mode patch_gaps
) else if "%CHOICE%"=="11" (
    echo Updating conda environment...
    conda env update -f environment.yml
) else if "%CHOICE%"=="12" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice.
)

pause
