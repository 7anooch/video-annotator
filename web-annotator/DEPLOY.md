# Deploying to Render.com - Step by Step Guide

## Prerequisites
- ✅ Render.com account (you have this!)
- ✅ GitHub account
- ✅ Your code pushed to a GitHub repository

## Step 1: Prepare Your GitHub Repository

1. **Make sure your code is committed and pushed to GitHub:**
   ```bash
   cd web-annotator
   git add .
   git commit -m "Ready for deployment"
   git push origin main  # or master, depending on your branch name
   ```

2. **Verify your repository structure:**
   - `app.py` (Flask application)
   - `requirements.txt` (dependencies)
   - `Procfile` (tells Render how to run your app)
   - `static/videos/` (with your video files)
   - `templates/` (HTML templates)
   - `static/css/` and `static/js/` (assets)

## Step 2: Create a New Web Service on Render

1. **Log into Render.com** and go to your dashboard

2. **Click "New +"** → **"Web Service"**

3. **Connect your repository:**
   - If this is your first time, click "Connect account" and authorize Render to access your GitHub
   - Select your repository from the list
   - Click "Connect"

4. **Configure your service:**
   - **Name**: Choose a name (e.g., "peristalsis-annotator")
   - **Region**: Choose closest to you (e.g., "Oregon (US West)")
   - **Branch**: Usually `main` or `master`
   - **Root Directory**: **IMPORTANT** - Set this to `web-annotator` (since your Flask app is in a subdirectory)
   - **Runtime**: Python 3
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```bash
     gunicorn app:app --bind 0.0.0.0:$PORT
     ```
     (Or just use the Procfile - Render will detect it automatically)

5. **Click "Create Web Service"**

## Step 3: Configure Environment Variables

1. **In your Render service dashboard**, go to **"Environment"** tab

2. **Add these environment variables** (if needed):
   - `PORT`: Usually auto-set by Render, but you can leave it
   - `VIDEO_DIR`: `/opt/render/project/src/static/videos` (or just leave default)
   - `DEFAULT_VIDEO_NAME`: Your video filename (e.g., `video1.mp4`)
   - `UPLOAD_DIR`: `/opt/render/project/src/uploads` (or leave default)
   
   **Optional (for email):**
   - `SENDGRID_API_KEY`: Your SendGrid API key
   - `RECIPIENT_EMAIL`: Your email address

3. **Click "Save Changes"**

## Step 4: Deploy

1. **Render will automatically start building** your service
2. **Watch the build logs** - you should see:
   - Installing dependencies from `requirements.txt`
   - Starting gunicorn
3. **Wait for "Your service is live"** message
4. **Your app URL** will be something like: `https://your-app-name.onrender.com`

## Step 5: Verify Everything Works

1. **Visit your app URL** in a browser
2. **Check that:**
   - The landing page loads
   - Videos are listed (if you have multiple)
   - Video playback works
   - Annotations can be created
   - CSV download works

## Troubleshooting

### Build Fails
- Check build logs for errors
- Make sure `requirements.txt` is correct
- Verify Python version compatibility

### Videos Not Loading
- Check that video files are in `static/videos/` in your repo
- Verify `DEFAULT_VIDEO_NAME` matches your actual video filename
- Check Render logs for 404 errors

### App Crashes
- Check the "Logs" tab in Render dashboard
- Common issues:
  - Missing environment variables
  - Port binding issues (should use `$PORT`)
  - File path issues

### Static Files Not Loading
- Make sure Flask's `static_folder` is set correctly in `app.py`
- Check that CSS/JS files are in the right directories

## Important Notes

1. **Free Tier Limitations:**
   - Services spin down after 15 minutes of inactivity
   - First request after spin-down takes ~30 seconds
   - 750 hours/month free (enough for most use cases)

2. **Persistent Storage:**
   - Uploaded CSVs are saved to the filesystem
   - On free tier, files persist but may be lost if you redeploy
   - Consider using email (SendGrid) for important uploads

3. **Video Files:**
   - Videos in your repo will be deployed
   - Keep file sizes reasonable (< 100MB recommended)
   - Multiple videos are supported

## Next Steps

Once deployed:
- Share the URL with your annotators
- Monitor uploads via email (if configured) or API endpoints
- Check Render dashboard for usage and logs

