# Local Setup Instructions

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   cd web-annotator
   pip install -r requirements.txt
   ```
   
   Or if you prefer a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add your video file**:
   - Place your video file in `static/videos/` directory
   - Example: `static/videos/video.mp4`
   - Update `config.py` if your video has a different name:
     ```python
     DEFAULT_VIDEO_NAME = "your_video.mp4"
     ```

3. **Run the Flask app**:
   ```bash
   python app.py
   ```
   
   You should see output like:
   ```
   * Running on http://0.0.0.0:5000
   ```

4. **Open in browser**:
   - Go to: `http://localhost:5000/`
   - NOT `http://0.0.0.0:5000/` (that might give 403)
   - You should see the onboarding modal, then the annotation interface

## Troubleshooting

### 403 Forbidden Error
- Make sure you're accessing `http://localhost:5000/` (not `0.0.0.0`)
- Check that Flask is actually running (look for the "Running on..." message)
- Try a different browser or clear browser cache

### Video Not Found
- Make sure your video file is in `static/videos/`
- Check that `DEFAULT_VIDEO_NAME` in `config.py` matches your filename
- Check file permissions (should be readable)

### Module Not Found Errors
- Make sure you've run `pip install -r requirements.txt`
- If using a virtual environment, make sure it's activated

### Port Already in Use
- If port 5000 is busy, Flask will tell you
- You can change the port by setting environment variable:
  ```bash
  PORT=5001 python app.py
  ```
  Then access at `http://localhost:5001/`

