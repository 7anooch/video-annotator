"""
Configuration for the web-based peristalsis annotation tool.

Values are read from environment variables when available, with sensible
defaults for local development. Adjust these as needed for your deployment.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Directory where video files are stored (served via /video/<filename>)
VIDEO_DIR = os.getenv("VIDEO_DIR", str(BASE_DIR / "static" / "videos"))

# Default video file to load in the UI
DEFAULT_VIDEO_NAME = os.getenv("DEFAULT_VIDEO_NAME", "video.mp4")

# Directory where uploaded CSVs are stored (if you use the /upload endpoint)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))

# Optional email configuration (for future /email endpoint)
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")  # where CSVs should be sent


