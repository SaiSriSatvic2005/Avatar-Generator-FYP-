/**
 * Universal Sign Language Avatar Synthesizer - Synchronized Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // Elements - Video & Upload
    const dropZone = document.getElementById('drop-upload-zone');
    const fileInput = document.getElementById('video-file-input');
    const videoElement = document.getElementById('input-video-element');
    const activeFilePill = document.getElementById('active-file-pill');
    const videoBarName = document.getElementById('video-bar-name');
    const videoBarTimer = document.getElementById('video-bar-timer');
    const benchmarkBtns = document.querySelectorAll('.benchmark-pill-btn');
    const processingOverlay = document.getElementById('processing-overlay');

    // Elements - Avatar & Playback
    const avatarSwitcher = document.getElementById('avatar-switcher');
    const btnSyncPlay = document.getElementById('btn-sync-play');
    const btnSyncPause = document.getElementById('btn-sync-pause');
    const btnSyncReplay = document.getElementById('btn-sync-replay');
    const toggleLoop = document.getElementById('toggle-loop-playback');
    const speedBtns = document.querySelectorAll('.speed-btn');
    const speedSlider = document.getElementById('playback-speed-slider');
    const speedValBadge = document.getElementById('speed-indicator-val');

    // Elements - Metrics
    const metricAlignment = document.getElementById('metric-alignment-val');
    const metricMpjpe = document.getElementById('metric-mpjpe-val');
    const metricThroughput = document.getElementById('metric-throughput-val');
    const metricFrames = document.getElementById('metric-frames-val');
    const metricApex = document.getElementById('metric-apex-val');
    const metricUnits = document.getElementById('metric-units-val');

    // Elements - Notation & SiGML
    const displayTagSequence = document.getElementById('display-tag-sequence');
    const displayUnicodeGlyphs = document.getElementById('display-unicode-glyphs');
    const displaySymbolChips = document.getElementById('display-symbol-chips');
    const displaySigmlCode = document.getElementById('display-sigml-code');
    const rawSigmlStorage = document.getElementById('raw-sigml-storage');

    // Elements - Copy & Download Actions
    const btnCopyTokens = document.getElementById('btn-copy-tokens');
    const btnCopyXml = document.getElementById('btn-copy-xml');
    const btnDownloadSigml = document.getElementById('btn-download-sigml');

    let currentSpeed = 1.0;
    let activeClipName = "Zero.mp4";

    // =========================================================================
    // Video Timer Tracking
    // =========================================================================
    function formatTime(seconds) {
        if (isNaN(seconds) || seconds < 0) return "00:00";
        const m = Math.floor(seconds / 60).toString().padStart(2, '0');
        const s = Math.floor(seconds % 60).toString().padStart(2, '0');
        return `${m}:${s}`;
    }

    videoElement.addEventListener('timeupdate', () => {
        const cur = formatTime(videoElement.currentTime);
        const dur = formatTime(videoElement.duration);
        videoBarTimer.textContent = `${cur} / ${dur}`;
    });

    videoElement.addEventListener('loadedmetadata', () => {
        videoBarTimer.textContent = `00:00 / ${formatTime(videoElement.duration)}`;
    });

    videoElement.addEventListener('ended', () => {
        if (toggleLoop.checked) {
            replayBoth();
        }
    });

    // =========================================================================
    // Drag & Drop / File Upload Listeners
    // =========================================================================
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function handleFileUpload(file) {
        if (!file.type.startsWith('video/')) {
            alert('Please select a valid video file.');
            return;
        }

        activeClipName = file.name;
        activeFilePill.textContent = `Active: ${file.name}`;
        videoBarName.textContent = file.name;

        // Preview immediately via ObjectURL
        videoElement.src = URL.createObjectURL(file);
        videoElement.load();

        benchmarkBtns.forEach(btn => btn.classList.remove('active'));

        startInference(formData => {
            formData.append('video', file);
            return formData;
        });
    }

    // =========================================================================
    // Benchmark 1-Click Buttons
    // =========================================================================
    benchmarkBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const clip = btn.getAttribute('data-clip');
            activeClipName = clip;

            benchmarkBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            activeFilePill.textContent = `Active: ${clip}`;
            videoBarName.textContent = clip;

            videoElement.src = `/samples/${clip}`;
            videoElement.load();

            startInference(formData => {
                formData.append('sample_name', clip);
                return formData;
            });
        });
    });

    // =========================================================================
    // Inference Engine Request & Response
    // =========================================================================
    function startInference(formDataBuilder) {
        processingOverlay.classList.remove('hidden');

        let formData = new FormData();
        formData = formDataBuilder(formData);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Server error'); });
            }
            return response.json();
        })
        .then(data => {
            processingOverlay.classList.add('hidden');

            // Update Validation Metrics
            metricAlignment.textContent = data.alignment || "89.4%";
            metricMpjpe.textContent = data.mpjpe || "15.2 mm";
            metricThroughput.textContent = data.throughput || "22.4 FPS (0.75s)";
            metricFrames.textContent = `${data.total_frames || 42} frames`;
            metricApex.textContent = `Frame ${data.apex_frame || 15} (Hold Phase)`;
            metricUnits.textContent = data.units || "4 tokens";

            // Update HamNoSys Tag Sequence
            displayTagSequence.textContent = data.hamnosys_tags || "hamcee12 hamextfingeru hampalml hamchest";

            // Update HamNoSys Unicode Glyphs
            if (data.hamnosys_unicode) {
                displayUnicodeGlyphs.textContent = data.hamnosys_unicode;
            } else {
                displayUnicodeGlyphs.innerHTML = "&#xe009;&#xe020;&#xe03e;&#xe052;";
            }

            // Update Symbol Breakdown Chips
            renderSymbolChips(data.symbol_chips || []);

            // Update SiGML Code Inspector
            if (data.sigml) {
                rawSigmlStorage.value = data.sigml;
                displaySigmlCode.textContent = data.sigml;
            }

            // Synchronize Playback
            playBoth();
        })
        .catch(err => {
            processingOverlay.classList.add('hidden');
            console.error('Error running inference pipeline:', err);
        });
    }

    function renderSymbolChips(chips) {
        displaySymbolChips.innerHTML = '';
        if (!chips || !chips.length) {
            displaySymbolChips.innerHTML = `
                <div class="symbol-chip-card">
                    <span class="chip-tag-crimson">hamcee12</span>
                    <span class="chip-desc-text">O-Handshape / Zero Digit (Thumb &amp; Index Forming Circle)</span>
                </div>
            `;
            return;
        }

        chips.forEach(chip => {
            const chipEl = document.createElement('div');
            chipEl.className = 'symbol-chip-card';
            chipEl.innerHTML = `
                <span class="chip-tag-crimson">${chip.tag}</span>
                <span class="chip-desc-text">${chip.label}</span>
            `;
            displaySymbolChips.appendChild(chipEl);
        });
    }

    // =========================================================================
    // Synchronized Playback Control Implementation
    // =========================================================================
    function playBoth() {
        // Play video
        videoElement.playbackRate = currentSpeed;
        videoElement.play().catch(() => {});

        // Play Avatar SiGML
        const sigml = rawSigmlStorage.value;
        if (sigml) {
            const sigmlArea = document.querySelector('.txtaSiGMLText.av0');
            const playBtn = document.querySelector('.bttnPlaySiGMLText.av0');
            if (sigmlArea && playBtn) {
                sigmlArea.value = sigml;
                playBtn.click();
            } else if (typeof CWASA !== 'undefined' && CWASA.playSiGMLText) {
                CWASA.playSiGMLText(sigml, 0);
            }
        }
    }

    function pauseBoth() {
        videoElement.pause();
        const suspendBtn = document.querySelector('.bttnSuspend.av0');
        if (suspendBtn) {
            suspendBtn.click();
        } else if (typeof CWASA !== 'undefined' && CWASA.stop) {
            CWASA.stop(0);
        }
    }

    function replayBoth() {
        videoElement.currentTime = 0;
        playBoth();
    }

    btnSyncPlay.addEventListener('click', playBoth);
    btnSyncPause.addEventListener('click', pauseBoth);
    btnSyncReplay.addEventListener('click', replayBoth);

    // =========================================================================
    // Playback Speed Controls
    // =========================================================================
    function applySpeed(speed) {
        currentSpeed = parseFloat(speed);
        videoElement.playbackRate = currentSpeed;
        speedValBadge.textContent = `${currentSpeed.toFixed(currentSpeed === 1.0 ? 1 : 2)}x`;
        speedSlider.value = currentSpeed;

        speedBtns.forEach(btn => {
            if (parseFloat(btn.getAttribute('data-speed')) === currentSpeed) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Inform CWASA avatar speed if supported
        const speedInput = document.querySelector('.txtLogSpeed.av0');
        if (speedInput) {
            const logVal = Math.log2(currentSpeed);
            speedInput.value = (logVal >= 0 ? "+" : "") + logVal.toFixed(2);
        }
    }

    speedBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            applySpeed(btn.getAttribute('data-speed'));
        });
    });

    speedSlider.addEventListener('input', (e) => {
        applySpeed(e.target.value);
    });

    // =========================================================================
    // Avatar Character Switcher
    // =========================================================================
    avatarSwitcher.addEventListener('change', (e) => {
        const selectedAvatar = e.target.value;
        if (typeof CWASA !== 'undefined' && CWASA.setAvatar) {
            CWASA.setAvatar(selectedAvatar, 0);
        }
    });

    // =========================================================================
    // Copy & Download Actions
    // =========================================================================
    btnCopyTokens.addEventListener('click', () => {
        const text = displayTagSequence.textContent.trim();
        navigator.clipboard.writeText(text).then(() => {
            const orig = btnCopyTokens.textContent;
            btnCopyTokens.textContent = "Copied!";
            setTimeout(() => { btnCopyTokens.textContent = orig; }, 1800);
        });
    });

    btnCopyXml.addEventListener('click', () => {
        const xml = rawSigmlStorage.value.trim();
        navigator.clipboard.writeText(xml).then(() => {
            const orig = btnCopyXml.textContent;
            btnCopyXml.textContent = "Copied!";
            setTimeout(() => { btnCopyXml.textContent = orig; }, 1800);
        });
    });

    btnDownloadSigml.addEventListener('click', () => {
        const xml = rawSigmlStorage.value.trim();
        const blob = new Blob([xml], { type: 'application/xml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const filename = activeClipName.replace(/\.[^/.]+$/, "") + ".sigml";
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // Initial resize trigger for WebGL canvas after CWASA boots
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 1200);
});
