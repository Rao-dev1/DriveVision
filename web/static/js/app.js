/**
 * DriveVision - AI Pothole Detection & Road Safety Application
 * Client-Side Application Controller
 */

// Application State
const state = {
  currentView: 'home',
  detectMode: 'image', // 'image' | 'video'
  activeSample: 'sample1.jpg',
  isScanning: false,
  scanProgress: 0,
  scanTimer: null,
  detectionData: null,
  currentPotholeIdx: 0,
  videoData: null,
  currentKeyframeIdx: 1,
  videoPlaying: false,
  videoProgress: 28, // percentage
  videoTimer: null
};

// DOM Elements
const views = {
  home: document.getElementById('view-home'),
  detect: document.getElementById('view-detect'),
  analyzing: document.getElementById('view-analyzing'),
  results: document.getElementById('view-results'),
  details: document.getElementById('view-details'),
  video: document.getElementById('view-video'),
  analytics: document.getElementById('view-analytics'),
  about: document.getElementById('view-about')
};

const navLinks = document.querySelectorAll('.nav-link');

// Navigation Function
function switchView(viewName) {
  state.currentView = viewName;

  // Toggle sections
  Object.keys(views).forEach(key => {
    if (views[key]) {
      if (key === viewName) {
        views[key].classList.add('active');
      } else {
        views[key].classList.remove('active');
      }
    }
  });

  // Update Nav links active state
  navLinks.forEach(link => {
    const target = link.getAttribute('data-view');
    if (target === viewName) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Special view triggers
  if (viewName === 'analytics') {
    loadAnalytics();
  }
}

// Attach Nav Clicks
navLinks.forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const target = link.getAttribute('data-view');
    if (target) switchView(target);
  });
});

// Top bar and button listeners
document.getElementById('btn-get-started')?.addEventListener('click', () => {
  setDetectMode('image');
  switchView('detect');
});
document.getElementById('btn-hero-detect')?.addEventListener('click', () => {
  setDetectMode('image');
  switchView('detect');
});
document.getElementById('btn-hero-demo')?.addEventListener('click', () => {
  setDetectMode('video');
  switchView('detect');
});
document.getElementById('btn-impact-start')?.addEventListener('click', () => {
  setDetectMode('image');
  switchView('detect');
});
document.getElementById('btn-new-detection')?.addEventListener('click', () => {
  setDetectMode('image');
  switchView('detect');
});
document.getElementById('btn-back-results')?.addEventListener('click', () => switchView('results'));
document.getElementById('btn-view-details')?.addEventListener('click', () => openPotholeDetails(0));

// Back to Detect and New Detection buttons from Video screen
document.getElementById('btn-video-back')?.addEventListener('click', (e) => {
  e.preventDefault();
  setDetectMode('video');
  switchView('detect');
});
document.getElementById('btn-new-video-detection')?.addEventListener('click', () => {
  setDetectMode('video');
  switchView('detect');
});

// ==========================================
// DETECT TAB: Image vs Video Mode Switcher
// ==========================================
const btnModeImage = document.getElementById('btn-mode-image');
const btnModeVideo = document.getElementById('btn-mode-video');
const panelDetectImage = document.getElementById('panel-detect-image');
const panelDetectVideo = document.getElementById('panel-detect-video');
const detectSideImg = document.getElementById('detect-side-img');

function setDetectMode(mode) {
  state.detectMode = mode;
  if (mode === 'image') {
    btnModeImage?.classList.add('active');
    btnModeVideo?.classList.remove('active');
    if (panelDetectImage) panelDetectImage.style.display = 'block';
    if (panelDetectVideo) panelDetectVideo.style.display = 'none';
    if (detectSideImg) detectSideImg.src = '/static/images/detect_road.jpg';
  } else {
    btnModeVideo?.classList.add('active');
    btnModeImage?.classList.remove('active');
    if (panelDetectVideo) panelDetectVideo.style.display = 'block';
    if (panelDetectImage) panelDetectImage.style.display = 'none';
    if (detectSideImg) detectSideImg.src = '/static/images/scanning_road.jpg';
    loadSampleVideos();
  }
}

btnModeImage?.addEventListener('click', () => setDetectMode('image'));
btnModeVideo?.addEventListener('click', () => setDetectMode('video'));

// ==========================================
// 1. IMAGE UPLOAD & SAMPLES HANDLING
// ==========================================
const dropzoneImage = document.getElementById('dropzone');
const fileInputImage = document.getElementById('file-input');

if (dropzoneImage && fileInputImage) {
  dropzoneImage.addEventListener('click', () => fileInputImage.click());

  dropzoneImage.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzoneImage.classList.add('dragover');
  });

  dropzoneImage.addEventListener('dragleave', () => {
    dropzoneImage.classList.remove('dragover');
  });

  dropzoneImage.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzoneImage.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleImageFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInputImage.addEventListener('change', () => {
    if (fileInputImage.files && fileInputImage.files.length > 0) {
      handleImageFileUpload(fileInputImage.files[0]);
    }
  });
}

function handleImageFileUpload(file) {
  const formData = new FormData();
  formData.append('file', file);

  const reader = new FileReader();
  reader.onload = (e) => {
    const scanImg = document.getElementById('scan-preview-img');
    if (scanImg) scanImg.src = e.target.result;
  };
  reader.readAsDataURL(file);

  startScanAndInference(formData, true, 'image');
}

// Load Image Samples
async function loadSampleImages() {
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    const container = document.getElementById('samples-container');
    if (!container || !data.samples) return;

    container.innerHTML = '';
    data.samples.slice(0, 4).forEach((sample) => {
      const card = document.createElement('div');
      card.className = 'sample-thumb-card';
      card.innerHTML = `
        <img src="${sample.url}" alt="${sample.name}" loading="lazy" />
        <div class="sample-thumb-label">${sample.name.replace(/_jpg.*\.jpg/, '.jpg')}</div>
      `;
      card.addEventListener('click', () => {
        triggerSampleImageInference(sample.name, sample.url);
      });
      container.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load sample images:', err);
  }
}

function triggerSampleImageInference(sampleName, sampleUrl) {
  const scanImg = document.getElementById('scan-preview-img');
  if (scanImg) scanImg.src = sampleUrl;

  const payload = JSON.stringify({ sample_name: sampleName });
  startScanAndInference(payload, false, 'image');
}

// ==========================================
// 2. VIDEO UPLOAD & SAMPLES HANDLING
// ==========================================
const dropzoneVideo = document.getElementById('dropzone-video');
const fileInputVideo = document.getElementById('video-file-input');

if (dropzoneVideo && fileInputVideo) {
  dropzoneVideo.addEventListener('click', () => fileInputVideo.click());

  dropzoneVideo.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzoneVideo.classList.add('dragover');
  });

  dropzoneVideo.addEventListener('dragleave', () => {
    dropzoneVideo.classList.remove('dragover');
  });

  dropzoneVideo.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzoneVideo.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleVideoFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInputVideo.addEventListener('change', () => {
    if (fileInputVideo.files && fileInputVideo.files.length > 0) {
      handleVideoFileUpload(fileInputVideo.files[0]);
    }
  });
}

function handleVideoFileUpload(file) {
  const formData = new FormData();
  formData.append('file', file);

  const scanImg = document.getElementById('scan-preview-img');
  if (scanImg) scanImg.src = '/static/images/scanning_road.jpg';

  startScanAndInference(formData, true, 'video', file.name);
}

// Load Video Samples
async function loadSampleVideos() {
  try {
    const res = await fetch('/api/sample-videos');
    const data = await res.json();
    const container = document.getElementById('sample-videos-container');
    if (!container || !data.videos) return;

    container.innerHTML = '';
    data.videos.forEach((vid) => {
      const card = document.createElement('div');
      card.className = 'sample-video-card';
      card.innerHTML = `
        <img src="${vid.thumbnail}" alt="${vid.name}" loading="lazy" />
        <div class="video-badge-pill">⏱ ${vid.badge}</div>
        <div class="sample-play-overlay">▶</div>
        <div class="sample-thumb-label">${vid.name}</div>
      `;
      card.addEventListener('click', () => {
        triggerSampleVideoInference(vid.filename, vid.thumbnail, vid.name);
      });
      container.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load sample videos:', err);
  }
}

function triggerSampleVideoInference(filename, thumbUrl, videoTitle) {
  const scanImg = document.getElementById('scan-preview-img');
  if (scanImg) scanImg.src = thumbUrl || '/static/images/scanning_road.jpg';

  const payload = JSON.stringify({ sample_video: filename });
  startScanAndInference(payload, false, 'video', videoTitle);
}

// ==========================================
// 3. AI SCANNING HUD ANIMATION & PIPELINE
// ==========================================
function startScanAndInference(requestBody, isFormData, mode = 'image', customTitle = '') {
  switchView('analyzing');
  state.isScanning = true;
  state.scanProgress = 0;

  const scanTitle = document.getElementById('scanning-title');
  const scanDesc = document.getElementById('scanning-desc');
  const progressText = document.getElementById('scan-progress-text');
  const progressBar = document.getElementById('scan-progress-bar');

  // Customize HUD labels based on Image vs Video mode
  if (mode === 'video') {
    if (scanTitle) scanTitle.innerText = 'Analyzing Video Stream...';
    if (scanDesc) scanDesc.innerText = customTitle ? `Scanning ${customTitle} frame by frame with YOLOv8...` : 'Extracting video frames and running YOLOv8 pothole detection...';
    document.getElementById('track-label-1').innerText = 'Extracting Frames';
    document.getElementById('track-label-2').innerText = 'Scanning with YOLOv8';
    document.getElementById('track-label-3').innerText = 'Assessing Severity';
    document.getElementById('track-label-4').innerText = 'Generating Summary';
  } else {
    if (scanTitle) scanTitle.innerText = 'Analyzing Your Road...';
    if (scanDesc) scanDesc.innerText = 'Our AI is scanning the image for potholes. Please wait.';
    document.getElementById('track-label-1').innerText = 'Pre-processing';
    document.getElementById('track-label-2').innerText = 'Detecting Potholes';
    document.getElementById('track-label-3').innerText = 'Assessing Severity';
    document.getElementById('track-label-4').innerText = 'Generating Results';
  }

  // Clear existing timer
  if (state.scanTimer) clearInterval(state.scanTimer);

  state.scanTimer = setInterval(() => {
    if (state.scanProgress < 90) {
      state.scanProgress += Math.floor(Math.random() * 8) + 4;
      if (state.scanProgress > 90) state.scanProgress = 90;
      updateProgressUI(state.scanProgress);
    }
  }, 110);

  function updateProgressUI(pct) {
    if (progressText) progressText.innerText = `${pct}%`;
    if (progressBar) {
      const offset = 220 - (220 * pct) / 100;
      progressBar.style.strokeDashoffset = offset;
    }

    const steps = [
      document.getElementById('track-step-1'),
      document.getElementById('track-step-2'),
      document.getElementById('track-step-3'),
      document.getElementById('track-step-4')
    ];

    steps.forEach(s => s?.classList.remove('active', 'completed'));
    if (pct < 28) {
      steps[0]?.classList.add('active');
    } else if (pct < 60) {
      steps[0]?.classList.add('completed');
      steps[1]?.classList.add('active');
    } else if (pct < 85) {
      steps[0]?.classList.add('completed');
      steps[1]?.classList.add('completed');
      steps[2]?.classList.add('active');
    } else {
      steps[0]?.classList.add('completed');
      steps[1]?.classList.add('completed');
      steps[2]?.classList.add('completed');
      steps[3]?.classList.add('active');
    }
  }

  // Select endpoint based on mode
  const endpoint = mode === 'video' ? '/api/detect-video' : '/api/detect';
  const options = {
    method: 'POST',
    body: requestBody
  };
  if (!isFormData) {
    options.headers = { 'Content-Type': 'application/json' };
  }

  fetch(endpoint, options)
    .then(res => res.json())
    .then(data => {
      clearInterval(state.scanTimer);
      state.scanProgress = 100;
      updateProgressUI(100);

      setTimeout(() => {
        state.isScanning = false;
        if (mode === 'video') {
          renderVideoAnalysis(data);
          switchView('video');
        } else {
          state.detectionData = data;
          renderDetectionResults(data);
          switchView('results');
        }
      }, 500);
    })
    .catch(err => {
      clearInterval(state.scanTimer);
      console.error('Detection request failed:', err);
      alert('Error running AI detection. Please try again.');
      switchView('detect');
    });
}

// ==========================================
// 4. RENDER IMAGE RESULTS & DETAILS
// ==========================================
function renderDetectionResults(data) {
  if (!data || !data.summary) return;

  const resImg = document.getElementById('results-annotated-img');
  if (resImg) resImg.src = data.annotated_image;

  document.getElementById('sum-total').innerText = data.summary.total;
  document.getElementById('sum-high').innerText = data.summary.high;
  document.getElementById('sum-medium').innerText = data.summary.medium;
  document.getElementById('sum-low').innerText = data.summary.low;

  const btnDownload = document.getElementById('btn-download-result');
  if (btnDownload) {
    btnDownload.onclick = () => {
      const a = document.createElement('a');
      a.href = data.annotated_image;
      a.download = `DriveVision_Detected_${Date.now()}.jpg`;
      a.click();
    };
  }

  if (data.pipeline_steps) {
    const steps = data.pipeline_steps;
    document.getElementById('step-img-original')?.setAttribute('src', steps.original);
    document.getElementById('step-img-grayscale')?.setAttribute('src', steps.grayscale);
    document.getElementById('step-img-blur')?.setAttribute('src', steps.blur);
    document.getElementById('step-img-canny')?.setAttribute('src', steps.canny);
    document.getElementById('step-img-contours')?.setAttribute('src', steps.contours);
    document.getElementById('step-img-yolo')?.setAttribute('src', steps.yolo);
  }
}

function openPotholeDetails(idx) {
  if (!state.detectionData || !state.detectionData.potholes || state.detectionData.potholes.length === 0) {
    return;
  }

  const potholes = state.detectionData.potholes;
  if (idx < 0) idx = potholes.length - 1;
  if (idx >= potholes.length) idx = 0;
  state.currentPotholeIdx = idx;

  const current = potholes[idx];

  const cropImg = document.getElementById('detail-crop-img');
  const cropPill = document.getElementById('detail-crop-pill');
  if (cropImg) cropImg.src = current.crop_base64 || state.detectionData.annotated_image;
  if (cropPill) {
    cropPill.innerText = `${current.severity} Severity`;
    cropPill.className = `crop-sev-pill ${current.severity.toLowerCase()}`;
    if (current.severity === 'High') cropPill.style.background = '#ef4444';
    else if (current.severity === 'Medium') cropPill.style.background = '#f59e0b';
    else cropPill.style.background = '#10b981';
  }

  document.getElementById('detail-pagination').innerText = `${idx + 1} / ${potholes.length}`;
  document.getElementById('detail-title').innerText = current.title;
  const sevBadge = document.getElementById('detail-sev-badge');
  if (sevBadge) {
    sevBadge.innerText = current.severity;
    sevBadge.className = `severity-pill-badge ${current.severity.toLowerCase()}`;
  }

  document.getElementById('detail-confidence').innerText = current.confidence.toFixed(2);
  document.getElementById('detail-area').innerText = current.area_px;
  document.getElementById('detail-severity-text').innerText = current.severity;
  document.getElementById('detail-location').innerText = current.location;
  document.getElementById('detail-recommendation').innerText = current.recommendation;

  switchView('details');
}

document.getElementById('btn-crop-prev')?.addEventListener('click', () => {
  openPotholeDetails(state.currentPotholeIdx - 1);
});
document.getElementById('btn-crop-next')?.addEventListener('click', () => {
  openPotholeDetails(state.currentPotholeIdx + 1);
});

// ==========================================
// 5. DYNAMIC VIDEO ANALYSIS & CONTROLS
// ==========================================
function renderVideoAnalysis(data) {
  state.videoData = data;
  state.currentKeyframeIdx = 1; // default to 2nd keyframe (Frame 124 / 450)

  // Update Header & Counter
  const descElem = document.getElementById('video-analysis-desc');
  if (descElem) descElem.innerText = `Potholes detected frame-by-frame in ${data.video_name}`;

  updateVideoTelemetry(data.current_frame, data.total_frames, data.duration, "0:12");

  // Update Summary Sidebar
  document.getElementById('video-sum-total').innerText = data.summary.total;
  document.getElementById('video-sum-high').innerText = data.summary.high;
  document.getElementById('video-sum-med').innerText = data.summary.medium;
  document.getElementById('video-sum-low').innerText = data.summary.low;

  // Update Active Frame View
  const frameImg = document.getElementById('video-active-frame-img');
  if (frameImg) frameImg.src = data.active_image;

  // Populate Keyframes Filmstrip
  const keyframesRow = document.getElementById('video-keyframes-row');
  if (keyframesRow && data.keyframes) {
    keyframesRow.innerHTML = '';
    data.keyframes.forEach((kf, idx) => {
      const thumb = document.createElement('div');
      thumb.className = `keyframe-thumb ${idx === state.currentKeyframeIdx ? 'active' : ''}`;
      thumb.innerHTML = `
        <img src="${kf.image}" alt="Keyframe ${kf.frame_index}" />
        <div class="sample-thumb-label" style="font-size: 10px;">F ${kf.frame_index}</div>
      `;
      thumb.addEventListener('click', () => {
        selectKeyframe(idx);
      });
      keyframesRow.appendChild(thumb);
    });
  }
}

function selectKeyframe(idx) {
  if (!state.videoData || !state.videoData.keyframes || !state.videoData.keyframes[idx]) return;
  state.currentKeyframeIdx = idx;
  const kf = state.videoData.keyframes[idx];

  // Update Frame image
  const frameImg = document.getElementById('video-active-frame-img');
  if (frameImg) frameImg.src = kf.image;

  // Update Telemetry
  updateVideoTelemetry(kf.frame_index, state.videoData.total_frames, state.videoData.duration, kf.timestamp);

  // Update slider
  const timeline = document.getElementById('video-timeline');
  if (timeline) {
    const pct = Math.round((kf.frame_index / state.videoData.total_frames) * 100);
    timeline.value = pct;
  }

  // Update active highlight in filmstrip
  const thumbs = document.querySelectorAll('#video-keyframes-row .keyframe-thumb');
  thumbs.forEach((t, i) => {
    if (i === idx) t.classList.add('active');
    else t.classList.remove('active');
  });
}

function updateVideoTelemetry(currentFrame, totalFrames, duration, curTimestamp) {
  const counter = document.getElementById('frame-counter-val');
  if (counter) counter.innerText = `Frame ${currentFrame} / ${totalFrames}`;

  const timestampDisplay = document.getElementById('video-timestamp');
  if (timestampDisplay) timestampDisplay.innerText = `${curTimestamp} / ${duration}`;
}

// Video Controls
const btnVideoPlay = document.getElementById('btn-video-play');
const videoTimeline = document.getElementById('video-timeline');
const btnVideoPrev = document.getElementById('btn-video-prev-frame');
const btnVideoNext = document.getElementById('btn-video-next-frame');

btnVideoPrev?.addEventListener('click', () => {
  if (!state.videoData || !state.videoData.keyframes) return;
  const prevIdx = (state.currentKeyframeIdx - 1 + state.videoData.keyframes.length) % state.videoData.keyframes.length;
  selectKeyframe(prevIdx);
});

btnVideoNext?.addEventListener('click', () => {
  if (!state.videoData || !state.videoData.keyframes) return;
  const nextIdx = (state.currentKeyframeIdx + 1) % state.videoData.keyframes.length;
  selectKeyframe(nextIdx);
});

if (btnVideoPlay && videoTimeline) {
  btnVideoPlay.addEventListener('click', () => {
    state.videoPlaying = !state.videoPlaying;
    btnVideoPlay.innerHTML = state.videoPlaying ? '⏸' : '▶';

    if (state.videoPlaying) {
      state.videoTimer = setInterval(() => {
        if (!state.videoData || !state.videoData.keyframes) return;
        const nextIdx = (state.currentKeyframeIdx + 1) % state.videoData.keyframes.length;
        selectKeyframe(nextIdx);
      }, 1200);
    } else {
      clearInterval(state.videoTimer);
    }
  });

  videoTimeline.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    if (!state.videoData || !state.videoData.keyframes) return;
    const kfIdx = Math.floor((val / 100) * state.videoData.keyframes.length);
    const safeIdx = Math.min(state.videoData.keyframes.length - 1, Math.max(0, kfIdx));
    selectKeyframe(safeIdx);
  });
}

// ==========================================
// 6. ANALYTICS LOADER
// ==========================================
async function loadAnalytics() {
  try {
    const res = await fetch('/api/analytics');
    const data = await res.json();

    document.getElementById('kpi-total-val').innerText = data.total_potholes;
    document.getElementById('kpi-high-val').innerText = data.high_severity;
    document.getElementById('kpi-med-val').innerText = data.medium_severity;
    document.getElementById('kpi-low-val').innerText = data.low_severity;
    document.getElementById('donut-total-num').innerText = data.total_potholes;
  } catch (err) {
    console.error('Failed to load analytics:', err);
  }
}

// Initial Boot
document.addEventListener('DOMContentLoaded', () => {
  loadSampleImages();
  loadSampleVideos();
});
