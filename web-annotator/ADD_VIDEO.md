# How to Add Your Video File

## Quick Steps

1. **Place your video file** in the `static/videos/` directory:
   ```bash
   cp /path/to/your/video.mp4 web-annotator/static/videos/video.mp4
   ```

2. **Or manually**:
   - Copy your video file
   - Paste it into `web-annotator/static/videos/`
   - Name it `video.mp4` (or update `DEFAULT_VIDEO_NAME` in `config.py`)

3. **Restart Flask**:
   ```bash
   cd web-annotator
   python app.py
   ```

## Video Format Requirements

- **Format**: MP4 (H.264 video codec recommended for best browser compatibility)
- **File size**: No strict limit, but smaller files load faster
- **Naming**: Default is `video.mp4`, but you can change it in `config.py`

## If Your Video Has a Different Name

Edit `config.py`:
```python
DEFAULT_VIDEO_NAME = "your_video_name.mp4"
```

## Converting Videos to MP4 (if needed)

If you have a video in another format, convert it with ffmpeg:
```bash
ffmpeg -i input_video.avi -c:v libx264 -c:a aac output_video.mp4
```

## Troubleshooting

- **Black screen / 0:00 duration**: Video file is missing or corrupted
- **"Video format not supported"**: Convert to H.264/AAC MP4
- **404 error**: Check that file exists at `static/videos/video.mp4`

