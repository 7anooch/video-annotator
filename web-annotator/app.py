import os
from datetime import datetime

import cv2
from flask import Flask, render_template, send_file, request, jsonify, abort

from config import VIDEO_DIR, DEFAULT_VIDEO_NAME, UPLOAD_DIR, SENDGRID_API_KEY, RECIPIENT_EMAIL


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    This app serves:
    - The main annotation page (index)
    - The video file for playback
    - A JSON API exposing accurate frame metadata using OpenCV
    - Optional CSV upload endpoint
    """
    app = Flask(__name__, static_folder="static", template_folder="templates")

    # Ensure required directories exist
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    @app.route("/")
    def index():
        """
        Landing page: list available videos for annotation.
        """
        # Get list of video files in VIDEO_DIR
        video_files = []
        if os.path.isdir(VIDEO_DIR):
            for filename in os.listdir(VIDEO_DIR):
                filepath = os.path.join(VIDEO_DIR, filename)
                if os.path.isfile(filepath) and filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                    # Get basic info for display
                    try:
                        cap = cv2.VideoCapture(filepath)
                        if cap.isOpened():
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
                            duration = frame_count / fps if fps > 0 else 0.0
                            cap.release()
                            video_files.append({
                                'filename': filename,
                                'frame_count': frame_count,
                                'fps': fps,
                                'duration': duration
                            })
                    except Exception as e:
                        app.logger.warning(f"Could not read video info for {filename}: {e}")
                        # Still include it, just without metadata
                        video_files.append({
                            'filename': filename,
                            'frame_count': None,
                            'fps': None,
                            'duration': None
                        })
        
        # Sort by filename
        video_files.sort(key=lambda x: x['filename'])
        return render_template("index.html", videos=video_files)
    
    @app.route("/annotate/<path:video_filename>")
    def annotate(video_filename: str):
        """
        Annotation interface for a specific video.
        """
        video_path = os.path.join(VIDEO_DIR, video_filename)
        if not os.path.isfile(video_path):
            abort(404, description=f"Video not found: {video_filename}")
        
        app.logger.info(f"Serving annotation page for video: {video_filename}")
        return render_template("annotate.html", video_filename=video_filename)

    @app.route("/video/<path:filename>")
    def serve_video(filename: str):
        """
        Serve the video file with proper headers for streaming.
        
        Supports HTTP range requests (206 Partial Content) which browsers
        use for efficient video playback.
        """
        video_path = os.path.join(VIDEO_DIR, filename)
        if not os.path.isfile(video_path):
            abort(404, description="Video not found")
        
        # Get file size for range request support
        file_size = os.path.getsize(video_path)
        
        # Check for range request header
        range_header = request.headers.get("Range", None)
        
        if range_header:
            # Parse range header (e.g., "bytes=0-1023")
            byte_start = 0
            byte_end = file_size - 1
            
            match = request.headers.get("Range", "").replace("bytes=", "").split("-")
            if match[0]:
                byte_start = int(match[0])
            if len(match) > 1 and match[1]:
                byte_end = int(match[1])
            
            length = byte_end - byte_start + 1
            
            def generate():
                with open(video_path, "rb") as f:
                    f.seek(byte_start)
                    remaining = length
                    while remaining:
                        chunk_size = min(8192, remaining)
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            
            response = app.response_class(
                generate(),
                206,
                {
                    "Content-Type": "video/mp4",
                    "Content-Range": f"bytes {byte_start}-{byte_end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(length),
                },
            )
            return response
        else:
            # No range request - send entire file
            return send_file(
                video_path,
                mimetype="video/mp4",
                as_attachment=False,
            )

    @app.route("/api/video-info/<path:filename>")
    def video_info(filename: str):
        """
        Return accurate video metadata (frame_count, fps, duration) using OpenCV.

        This is CRITICAL for frame-accurate annotation. We intentionally use the
        same method as the desktop tool:
        - frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        - fps = cap.get(cv2.CAP_PROP_FPS)
        """
        video_path = os.path.join(VIDEO_DIR, filename)
        if not os.path.isfile(video_path):
            abort(404, description="Video not found")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            abort(500, description="Failed to open video")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        cap.release()

        if fps <= 0:
            # Fallback to a reasonable default if FPS is not available
            fps = 30.0

        duration = frame_count / fps if fps > 0 else 0.0

        return jsonify(
            {
                "frame_count": frame_count,
                "fps": fps,
                "duration": duration,
                "filename": filename,
            }
        )

    @app.route("/upload", methods=["POST"])
    def upload_csv():
        """
        Optional endpoint to receive annotation CSVs from the client.

        Expects:
        - form field 'annotator_name'
        - form field 'video_filename' (optional, for multi-video support)
        - file field 'file' containing the CSV
        """
        file = request.files.get("file")
        annotator_name = request.form.get("annotator_name", "").strip() or "unknown"
        video_filename = request.form.get("video_filename", "").strip() or "video"

        if not file:
            abort(400, description="No file uploaded")

        # Very basic file type check
        if not file.filename.lower().endswith(".csv"):
            abort(400, description="Only CSV files are allowed")

        # Sanitize names for filename
        safe_name = "".join(c for c in annotator_name if c.isalnum() or c in ("-", "_"))
        # Remove extension from video filename for cleaner CSV name
        base_video_name = os.path.splitext(video_filename)[0]
        safe_video_name = "".join(c for c in base_video_name if c.isalnum() or c in ("-", "_", ".")) or "video"
        
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        target_name = f"{safe_video_name}_{safe_name or 'annotator'}_{timestamp}_perisannot.csv"
        target_path = os.path.join(UPLOAD_DIR, target_name)

        file.save(target_path)
        app.logger.info(f"Saved uploaded CSV: {target_path}")
        
        # Optionally send email if configured
        email_sent = False
        email_error = None
        if SENDGRID_API_KEY and RECIPIENT_EMAIL:
            try:
                email_sent = _send_csv_email(target_path, target_name, annotator_name, video_filename)
                app.logger.info(f"Email sent successfully for {target_name}")
            except Exception as e:
                email_error = str(e)
                app.logger.error(f"Failed to send email: {e}")
                # Don't fail the upload if email fails

        response = {
            "status": "ok", 
            "saved_as": target_name,
            "email_sent": email_sent
        }
        if email_error:
            response["email_error"] = email_error
            
        return jsonify(response)

    def _send_csv_email(file_path: str, filename: str, annotator_name: str, video_filename: str) -> bool:
        """
        Send the uploaded CSV file via email using SendGrid.
        
        Returns True if email was sent successfully, False otherwise.
        """
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail, Attachment, Disposition, FileContent, FileName, FileType
        except ImportError:
            app.logger.warning("SendGrid not installed. Install with: pip install sendgrid")
            return False
        
        if not SENDGRID_API_KEY or not RECIPIENT_EMAIL:
            return False
        
        # Read the CSV file
        with open(file_path, 'rb') as f:
            csv_data = f.read()
        
        import base64
        encoded = base64.b64encode(csv_data).decode('utf-8')
        
        # Create email message
        subject = f"Peristalsis Annotations: {video_filename} from {annotator_name}"
        html_content = f"""
        <p>New peristalsis annotation file uploaded:</p>
        <ul>
            <li><strong>Video:</strong> {video_filename}</li>
            <li><strong>Annotator:</strong> {annotator_name}</li>
            <li><strong>File:</strong> {filename}</li>
        </ul>
        <p>The CSV file is attached to this email.</p>
        """
        
        message = Mail(
            from_email=RECIPIENT_EMAIL,
            to_emails=RECIPIENT_EMAIL,
            subject=subject,
            html_content=html_content,
        )
        
        attachment = Attachment(
            FileContent(encoded),
            FileName(filename),
            FileType("text/csv"),
            Disposition("attachment"),
        )
        message.add_attachment(attachment)
        
        # Send email
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        
        if response.status_code in [200, 202]:
            return True
        else:
            app.logger.error(f"SendGrid returned status {response.status_code}")
            return False
    
    @app.route("/api/uploads", methods=["GET"])
    def list_uploads():
        """
        List all uploaded CSV files (for admin access).
        Returns a list of uploaded files with metadata.
        """
        if not os.path.isdir(UPLOAD_DIR):
            return jsonify({"uploads": []})
        
        uploads = []
        for filename in os.listdir(UPLOAD_DIR):
            filepath = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(filepath) and filename.lower().endswith('.csv'):
                stat = os.stat(filepath)
                uploads.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
        
        # Sort by upload time (newest first)
        uploads.sort(key=lambda x: x["uploaded_at"], reverse=True)
        return jsonify({"uploads": uploads})
    
    @app.route("/api/uploads/<path:filename>", methods=["GET"])
    def download_upload(filename: str):
        """
        Download a specific uploaded CSV file (for admin access).
        """
        # Security: prevent directory traversal
        if '..' in filename or '/' in filename:
            abort(400, description="Invalid filename")
        
        filepath = os.path.join(UPLOAD_DIR, filename)
        if not os.path.isfile(filepath):
            abort(404, description="File not found")
        
        return send_file(filepath, mimetype="text/csv", as_attachment=True)

    return app


app = create_app()


if __name__ == "__main__":
    # For local development only. In production use gunicorn or similar.
    # Default to 5001 to avoid macOS AirPlay Receiver conflict on port 5000
    port = int(os.getenv("PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=True)
    print(f"\n✓ Server running at http://localhost:{port}/")


