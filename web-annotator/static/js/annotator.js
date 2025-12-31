// Main client-side logic for the web peristalsis annotation tool.
//
// Responsibilities:
// - Fetch accurate video metadata (frame_count, fps) from the server
// - Provide frame-accurate navigation
// - Manage per-frame annotations (0/1)
// - Render frame list and keep it in sync with video playback
// - Support CSV generation and download
// - Provide onboarding/instructions modal

(() => {
  const config = window.APP_CONFIG || {};
  const videoFilename = config.videoFilename;

  // DOM elements
  let video;
  let playPauseBtn;
  let prevFrameBtn;
  let nextFrameBtn;
  let frameSlider;
  let currentFrameSpan;
  let totalFramesSpan;
  let fpsSpan;
  let durationSpan;
  let currentLabelSpan;
  let clearSelectedBtn;
  let rangeStartInput;
  let rangeEndInput;
  let clearRangeBtn;
  let downloadCsvBtn;
  let uploadCsvBtn;
  let frameListContainer;
  let annotatorNameInput;
  let playbackFpsSelect;

  let onboardingModal;
  let startAnnotatingBtn;
  let hideOnboardingCheckbox;

  // State
  let frameCount = 0;
  let fps = 30;
  let currentFrame = 0;
  let annotations = [];
  let isPlaying = false;
  let cyclePressStartFrame = null;
  let isManuallySeeking = false;

  function $(id) {
    return document.getElementById(id);
  }

  function initDomRefs() {
    video = $("video");
    playPauseBtn = $("play-pause-btn");
    prevFrameBtn = $("prev-frame-btn");
    nextFrameBtn = $("next-frame-btn");
    frameSlider = $("frame-slider");
    currentFrameSpan = $("current-frame");
    totalFramesSpan = $("total-frames");
    fpsSpan = $("fps-display");
    durationSpan = $("duration-display");
    currentLabelSpan = $("current-label-display");
    clearSelectedBtn = $("clear-selected-btn");
    rangeStartInput = $("range-start-input");
    rangeEndInput = $("range-end-input");
    clearRangeBtn = $("clear-range-btn");
    downloadCsvBtn = $("download-csv-btn");
    uploadCsvBtn = $("upload-csv-btn");
    frameListContainer = $("frame-list");
    annotatorNameInput = $("annotator-name-input");
    playbackFpsSelect = $("playback-fps-select");

    onboardingModal = $("onboarding-modal");
    startAnnotatingBtn = $("start-annotating-btn");
    hideOnboardingCheckbox = $("hide-onboarding-checkbox");
  }

  async function fetchVideoInfo() {
    const resp = await fetch(`/api/video-info/${encodeURIComponent(videoFilename)}`);
    if (!resp.ok) {
      throw new Error(`Failed to fetch video info: ${resp.statusText}`);
    }
    return resp.json();
  }

  function setCurrentFrame(newFrame) {
    // Clamp to valid range
    newFrame = Math.max(0, Math.min(frameCount - 1, Math.floor(newFrame)));
    if (newFrame === currentFrame) return; // No change needed
    
    isManuallySeeking = true;
    currentFrame = newFrame;
    const targetTime = currentFrame / fps;
    video.currentTime = targetTime;
    updateFrameDisplay();
    updateFrameSlider();
    
    // Reset flag after a short delay to allow seek to complete
    setTimeout(() => {
      isManuallySeeking = false;
    }, 100);
  }

  function updateFrameDisplay() {
    currentFrameSpan.textContent = String(currentFrame);
    const label = annotations[currentFrame] || 0;
    currentLabelSpan.textContent = String(label);
    highlightFrameRow(currentFrame);
  }

  function updateFrameSlider() {
    frameSlider.value = String(currentFrame);
  }

  function buildFrameList() {
    frameListContainer.innerHTML = "";
    for (let f = 0; f < frameCount; f++) {
      const row = document.createElement("div");
      row.className = "frame-row";
      row.dataset.frameIndex = String(f);

      const left = document.createElement("span");
      left.textContent = `Frame ${f}`;

      const right = document.createElement("span");
      right.className = "frame-label";
      if (annotations[f] === 1) {
        right.textContent = "Cycle End";
        row.classList.add("cycle-start");
      }

      row.appendChild(left);
      row.appendChild(right);

      row.addEventListener("click", () => {
        setCurrentFrame(f);
      });

      frameListContainer.appendChild(row);
    }
    highlightFrameRow(currentFrame);
  }

  function refreshFrameRow(frameIndex) {
    const row = frameListContainer.querySelector(
      `.frame-row[data-frame-index="${frameIndex}"]`
    );
    if (!row) return;
    const labelSpan = row.querySelector(".frame-label");
    const label = annotations[frameIndex] || 0;
    row.classList.toggle("cycle-start", label === 1);
    labelSpan.textContent = label === 1 ? "Cycle End" : "";
  }

  function highlightFrameRow(frameIndex) {
    const rows = frameListContainer.querySelectorAll(".frame-row");
    rows.forEach((row) => row.classList.remove("current"));
    const row = frameListContainer.querySelector(
      `.frame-row[data-frame-index="${frameIndex}"]`
    );
    if (row) {
      row.classList.add("current");
      // Scroll into view if needed
      const containerRect = frameListContainer.getBoundingClientRect();
      const rowRect = row.getBoundingClientRect();
      if (rowRect.top < containerRect.top || rowRect.bottom > containerRect.bottom) {
        row.scrollIntoView({ block: "center" });
      }
    }
  }

  function applyLabelToFrame(frameIndex, label) {
    if (frameIndex < 0 || frameIndex >= frameCount) return;
    annotations[frameIndex] = label;
    if (frameIndex === currentFrame) {
      currentLabelSpan.textContent = String(label);
    }
    refreshFrameRow(frameIndex);
  }

  function handleCycleStartPress() {
    cyclePressStartFrame = currentFrame;
  }

  function handleCycleStartRelease() {
    if (cyclePressStartFrame == null) return;
    applyLabelToFrame(cyclePressStartFrame, 1);
    cyclePressStartFrame = null;
  }

  function clearCurrentFrame() {
    applyLabelToFrame(currentFrame, 0);
  }

  function clearRange() {
    const start = parseInt(rangeStartInput.value, 10);
    const end = parseInt(rangeEndInput.value, 10);
    if (Number.isNaN(start) || Number.isNaN(end)) return;
    const s = Math.max(0, Math.min(frameCount - 1, start));
    const e = Math.max(0, Math.min(frameCount - 1, end));
    if (e < s) return;
    for (let f = s; f <= e; f++) {
      applyLabelToFrame(f, 0);
    }
  }

  function togglePlayPause() {
    if (video.paused) {
      video.play();
      isPlaying = true;
      playPauseBtn.textContent = "Pause";
    } else {
      video.pause();
      isPlaying = false;
      playPauseBtn.textContent = "Play";
    }
  }

  function generateCsv() {
    let csv = "frame,label\n";
    for (let f = 0; f < frameCount; f++) {
      const label = annotations[f] || 0;
      csv += `${f},${label}\n`;
    }
    return csv;
  }

  function downloadCsv() {
    const csv = generateCsv();

    // Basic validation on row count (including header)
    const lineCount = csv.split("\n").filter((l) => l.length > 0).length;
    if (lineCount !== frameCount + 1) {
      console.warn(
        "CSV line count mismatch. Expected",
        frameCount + 1,
        "got",
        lineCount
      );
    }

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const name = annotatorNameInput.value.trim() || "annotator";
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, "_");
    const baseVideoName = (videoFilename || "video").replace(/\.[^/.]+$/, "");
    const filename = `${baseVideoName}_${safeName}_perisannot.csv`;

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async function uploadCsv() {
    const csv = generateCsv();
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const name = annotatorNameInput.value.trim() || "annotator";
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, "_");
    const baseVideoName = (videoFilename || "video").replace(/\.[^/.]+$/, "");

    const formData = new FormData();
    formData.append("annotator_name", safeName);
    formData.append("video_filename", videoFilename || "video");
    formData.append(
      "file",
      new File([blob], `${baseVideoName}_${safeName}_perisannot.csv`, {
        type: "text/csv",
      })
    );

    try {
      const resp = await fetch("/upload", {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || resp.statusText);
      }
      const data = await resp.json();
      alert("Upload successful. Saved as: " + data.saved_as);
    } catch (err) {
      console.error(err);
      alert(
        "Upload failed. You can still use Download CSV and send the file manually."
      );
    }
  }

  function attachEventHandlers() {
    playPauseBtn.addEventListener("click", togglePlayPause);

    prevFrameBtn.addEventListener("click", () => {
      setCurrentFrame(currentFrame - 1);
    });

    nextFrameBtn.addEventListener("click", () => {
      setCurrentFrame(currentFrame + 1);
    });

    frameSlider.addEventListener("input", (e) => {
      const value = parseInt(e.target.value, 10) || 0;
      setCurrentFrame(value);
    });

    clearSelectedBtn.addEventListener("click", clearCurrentFrame);
    clearRangeBtn.addEventListener("click", clearRange);
    downloadCsvBtn.addEventListener("click", downloadCsv);
    if (uploadCsvBtn) {
      uploadCsvBtn.addEventListener("click", uploadCsv);
    }

    if (playbackFpsSelect) {
      playbackFpsSelect.addEventListener("change", () => {
        const desired = parseInt(playbackFpsSelect.value, 10) || 30;
        // Adjust playbackRate relative to the true fps so that effective
        // playback frame rate matches the selected value.
        const rate = fps > 0 ? desired / fps : 1.0;
        video.playbackRate = rate;
      });
    }

    const cycleStartBtn = $("cycle-start-btn");
    cycleStartBtn.addEventListener("click", () => {
      // Single-frame labeling via button
      handleCycleStartPress();
      handleCycleStartRelease();
    });

    // Keyboard shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.target && ["INPUT", "TEXTAREA"].includes(e.target.tagName)) {
        return;
      }
      if (e.code === "Space") {
        e.preventDefault();
        e.stopPropagation();
        togglePlayPause();
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        e.stopPropagation();
        if (currentFrame < frameCount - 1) {
          setCurrentFrame(currentFrame + 1);
        }
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        e.stopPropagation();
        if (currentFrame > 0) {
          setCurrentFrame(currentFrame - 1);
        }
      } else if (e.key === "c" || e.key === "C" || e.key === "m" || e.key === "M") {
        // Cycle end press (C or M for left/right-handed users)
        if (cyclePressStartFrame == null) {
          handleCycleStartPress();
        }
      } else if (e.key === "Backspace") {
        e.preventDefault();
        e.stopPropagation();
        clearCurrentFrame();
      }
    });

    window.addEventListener("keyup", (e) => {
      if (e.key === "c" || e.key === "C" || e.key === "m" || e.key === "M") {
        handleCycleStartRelease();
      }
    });

    // Video error handling
    video.addEventListener("error", (e) => {
      console.error("Video error:", e);
      const error = video.error;
      if (error) {
        let msg = "Video failed to load. ";
        switch (error.code) {
          case error.MEDIA_ERR_ABORTED:
            msg += "Loading aborted.";
            break;
          case error.MEDIA_ERR_NETWORK:
            msg += "Network error.";
            break;
          case error.MEDIA_ERR_DECODE:
            msg += "Video codec not supported or file corrupted.";
            break;
          case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
            msg += "Video format not supported.";
            break;
          default:
            msg += `Error code: ${error.code}`;
        }
        alert(msg + "\n\nPlease check:\n1. Video file exists at static/videos/video.mp4\n2. File is a valid MP4 video\n3. File permissions are correct");
      }
    });

    // Keep currentFrame in sync when video time updates (e.g. during play)
    video.addEventListener("timeupdate", () => {
      // Don't update if we're manually seeking (to prevent conflicts with arrow keys)
      if (isManuallySeeking) return;
      
      if (!isNaN(video.currentTime)) {
        const newFrame = Math.floor(video.currentTime * fps);
        if (newFrame !== currentFrame) {
          currentFrame = Math.max(0, Math.min(frameCount - 1, newFrame));
          updateFrameDisplay();
          updateFrameSlider();
        }
      }
    });

    // Ensure we recalc frame after explicit seeks
    video.addEventListener("seeked", () => {
      if (isManuallySeeking) {
        // If we're manually seeking, ensure we're on the exact frame we want
        const targetFrame = Math.floor(video.currentTime * fps);
        if (Math.abs(targetFrame - currentFrame) > 1) {
          // If there's a significant difference, correct it
          currentFrame = Math.max(0, Math.min(frameCount - 1, targetFrame));
          updateFrameDisplay();
          updateFrameSlider();
        }
        isManuallySeeking = false;
      } else {
        const newFrame = Math.floor(video.currentTime * fps);
        currentFrame = Math.max(0, Math.min(frameCount - 1, newFrame));
        updateFrameDisplay();
        updateFrameSlider();
      }
    });

    // Check if video loaded successfully
    video.addEventListener("loadedmetadata", () => {
      if (video.duration === 0 || isNaN(video.duration)) {
        console.warn("Video duration is 0 or invalid. Video may not have loaded correctly.");
      }
    });
  }

  function initOnboarding() {
    const hideFlag = window.localStorage.getItem("hideOnboarding");
    if (!hideFlag) {
      onboardingModal.classList.remove("hidden");
    } else {
      // Directly initialize annotator if user opted out
      initializeAnnotator();
    }

    startAnnotatingBtn.addEventListener("click", () => {
      onboardingModal.classList.add("hidden");
      if (hideOnboardingCheckbox.checked) {
        window.localStorage.setItem("hideOnboarding", "true");
      }
      initializeAnnotator();
    });
  }

  async function initializeAnnotator() {
    try {
      const info = await fetchVideoInfo();
      frameCount = info.frame_count;
      fps = info.fps || 30;

      annotations = new Array(frameCount).fill(0);

      totalFramesSpan.textContent = String(frameCount - 1);
      fpsSpan.textContent = fps.toFixed(2);
      durationSpan.textContent = info.duration.toFixed(2);

      frameSlider.min = "0";
      frameSlider.max = String(frameCount - 1);
      frameSlider.value = "0";

      // Initialize playback rate to match selected playback FPS (default 30).
      if (playbackFpsSelect) {
        const desired = parseInt(playbackFpsSelect.value, 10) || 30;
        const rate = fps > 0 ? desired / fps : 1.0;
        video.playbackRate = rate;
      }

      // Debug: Check video element state
      console.log("Video element src:", video.src);
      console.log("Video element readyState:", video.readyState);
      console.log("Video element networkState:", video.networkState);
      
      // Wait a bit for video metadata to load, then check
      setTimeout(() => {
        console.log("Video duration after load:", video.duration);
        console.log("Video videoWidth:", video.videoWidth);
        console.log("Video videoHeight:", video.videoHeight);
        if (video.duration === 0 || isNaN(video.duration)) {
          console.warn("⚠️ Video duration is 0 or invalid!");
          console.warn("Video error object:", video.error);
          if (video.error) {
            console.error("Video error code:", video.error.code);
            console.error("Video error message:", video.error.message);
          }
        }
      }, 1000);

      buildFrameList();
      updateFrameDisplay();
      attachEventHandlers();
    } catch (err) {
      console.error(err);
      alert("Failed to initialize video / annotations. See console for details.");
    }
  }

  window.addEventListener("load", () => {
    initDomRefs();
    initOnboarding();
  });
})();


