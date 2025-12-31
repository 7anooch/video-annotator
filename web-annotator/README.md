# Web Peristalsis Annotation Tool

This is a web-based version of the peristalsis annotation tool. It allows
annotators to label **cycle start** frames in a fixed video and export the
results as a `_perisannot.csv` file compatible with the desktop workflow.

## Features

- Frame-accurate video navigation based on server-side OpenCV metadata
- Cycle start labeling (`1`) with single-frame annotations
- Clear single frame or clear a frame range
- Scrollable frame list (click to jump, current frame highlighted)
- Onboarding / instructions modal on first load (with \"Don't show again\")
- CSV download (`frame,label` for all frames `0..frameCount-1`)
- Optional CSV upload endpoint on the server

## Technology Stack

- Backend: Flask (Python)
- Frontend: HTML5, CSS, Vanilla JavaScript
- Video metadata: OpenCV (server-side, same as desktop tool)

## Project Layout

```text
web-annotator/
├── app.py                 # Flask application and routes
├── config.py              # Paths and optional email configuration
├── requirements.txt       # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css      # Layout and visual styling
│   ├── js/
│   │   └── annotator.js   # Client-side annotation logic
│   └── videos/
│       └── video.mp4      # Default video (you provide this)
├── templates/
│   ├── index.html         # Main annotator UI + onboarding modal
│   └── success.html       # Simple \"thank you\" page (optional)
└── uploads/               # Destination for uploaded CSVs (optional)
```

## Setup (Local Development)

1. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   cd web-annotator
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your video file**:

   - Place your video in `static/videos/`, for example `headlocked_part1.mp4`.
   - Update `DEFAULT_VIDEO_NAME` in `config.py` if you change the filename.

4. **Run the app**:

   ```bash
   python app.py
   ```

5. **Open in browser**:

   - Go to `http://localhost:5000/`

## Frame Accuracy Notes

Accurate frame numbers are critical. This web app uses the **same method**
as the desktop tool to get frame count and FPS:

- Server-side: `cv2.VideoCapture(video_path)`
  - `frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))`
  - `fps = cap.get(cv2.CAP_PROP_FPS)`

The browser then:

- Uses `frame = Math.floor(video.currentTime * fps)` to compute frame indices.
- Seeks by setting `video.currentTime = frame / fps`.
- Keeps an integer `currentFrame` in JavaScript and clamps it to
  `[0, frame_count - 1]`.

The CSV output always includes one row per frame:

```csv
frame,label
0,0
1,1
2,0
...
```

matching the desktop `_perisannot.csv` structure.

## Onboarding / Instructions

On first load, users see an onboarding modal with:

- Text instructions
- Optional images or demo video (you can edit `index.html` to add media)
- \"Start Annotating\" button
- \"Don't show this again on this device\" checkbox (stored in `localStorage`)

You can freely edit the instructional content in `templates/index.html` inside
the `#onboarding-modal` block.

## CSV Download and Upload

- **Download**:
  - Click \"Download CSV\".
  - Filename pattern:
    `VIDEO_BASENAME_ANNOTATORNAME_perisannot.csv`
  - Contains `frame,label` for frames `0` to `frame_count - 1`.

- **Upload (optional)**:
  - Click \"Upload to Server\".
  - Sends the CSV plus the annotator name to `/upload`.
  - The server saves it under `uploads/` with a timestamped filename.

## Email Integration (Automatic)

When users click "Upload Annotation Labels", the CSV file is:
1. **Saved** to the `uploads/` directory on the server
2. **Automatically emailed** to you (if configured)

### Setting Up Email

1. **Install SendGrid** (already in requirements.txt):
   ```bash
   pip install sendgrid
   ```

2. **Get a SendGrid API key**:
   - Sign up at [sendgrid.com](https://sendgrid.com) (free tier available)
   - Create an API key in Settings → API Keys
   - Copy the API key

3. **Set environment variables**:
   ```bash
   export SENDGRID_API_KEY="your-api-key-here"
   export RECIPIENT_EMAIL="your-email@example.com"
   ```

   Or in Render.com:
   - Go to your service → Environment
   - Add `SENDGRID_API_KEY` and `RECIPIENT_EMAIL`

4. **That's it!** Uploads will automatically email you the CSV file.

### Accessing Uploaded Files

Uploaded CSV files are saved in `web-annotator/uploads/` with filenames like:
```
video1_john_20251223T150943Z_perisannot.csv
```

You can also access them via API:
- **List all uploads**: `GET /api/uploads`
- **Download a file**: `GET /api/uploads/<filename>`

**Note**: If email is not configured, files are still saved to the server. Email is optional.

## Deployment (Free Options)

You can deploy this using a free tier on:

- **Render.com** (simple GitHub integration, containerized, free tier)
- **Railway.app** (credits with GitHub Education)

Basic steps:

1. Push the `web-annotator` folder to a GitHub repository.
2. Create a new web service in Render/Railway connected to your repo.
3. Set build command:

   ```bash
   pip install -r requirements.txt
   ```

4. Set start command:

   ```bash
   gunicorn app:app
   ```

5. Set environment variables (e.g. `VIDEO_DIR`, `DEFAULT_VIDEO_NAME`).
6. Add your video file either in the repo (`static/videos`) or via persistent
   storage on the platform.

## How to Customize for Your Study

- **Video**: Replace `static/videos/video.mp4`, update `DEFAULT_VIDEO_NAME`.
- **Instructions**: Edit the onboarding modal text in `templates/index.html`.
- **Keybindings**: Adjust logic in `static/js/annotator.js` if you want to
  add or change shortcuts.
- **Peristalsis semantics**: Currently a single label (1 = cycle start).
  If you ever need more labels, you can extend the CSV columns and UI.


