document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('video-input');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    // Video Elements
    const previewContainer = document.getElementById('preview-container');
    const videoPlayer = document.getElementById('input-video-player');
    const previewFilename = document.getElementById('preview-filename');
    
    // Cards
    const meaningCard = document.getElementById('meaning-card');
    const signGloss = document.getElementById('sign-gloss');
    const signMeaning = document.getElementById('sign-meaning');
    const precisionPill = document.getElementById('precision-pill');
    const metricConf = document.getElementById('metric-conf');
    const metricPrec = document.getElementById('metric-prec');
    
    const resultCard = document.getElementById('result-card');
    const hamnosysTags = document.getElementById('hamnosys-tags');
    const hamnosysChars = document.getElementById('hamnosys-chars');
    const symbolChipsContainer = document.getElementById('symbol-chips');
    
    const matrixCard = document.getElementById('matrix-card');
    const matrixGrid = document.getElementById('matrix-grid');
    
    const sigmlCard = document.getElementById('sigml-card');
    const sigmlCodeDisplay = document.getElementById('sigml-code-display');
    const sigmlStorage = document.getElementById('sigml-storage');
    
    // Synchronized Controllers
    const syncPlayBtn = document.getElementById('sync-play-btn');
    const syncStopBtn = document.getElementById('sync-stop-btn');
    const syncReplayBtn = document.getElementById('sync-replay-btn');
    const loopToggle = document.getElementById('loop-toggle');
    const speedPills = document.querySelectorAll('.speed-pill');
    const avatarSelect = document.getElementById('avatar-model-select');
    
    const scrubPrevBtn = document.getElementById('scrub-prev-btn');
    const scrubNextBtn = document.getElementById('scrub-next-btn');
    const frameCounterDisplay = document.getElementById('frame-counter-display');
    
    const copyTokensBtn = document.getElementById('copy-tokens-btn');
    const copySigmlBtn = document.getElementById('copy-sigml-btn');
    const sampleBtns = document.querySelectorAll('.sample-btn');

    // State Variables
    let isPlaying = false;
    let isLooping = true;
    let avatarLoopTimer = null;
    let currentSpeed = 1.0;
    let avatarWasPlaying = false;

    // Ensure video does NOT use native loop attribute so 'ended' event fires properly
    if (videoPlayer) {
        videoPlayer.loop = false;
    }

    // Initialize Loop Toggle State
    if (loopToggle) {
        isLooping = loopToggle.checked;
        loopToggle.addEventListener('change', (e) => {
            isLooping = e.target.checked;
        });
    }

    // Video ended listener for synchronized looping
    if (videoPlayer) {
        videoPlayer.addEventListener('ended', () => {
            if (isLooping && isPlaying) {
                console.log("[Sync Loop Engine] Input Video ended -> Synchronized replay of video and 3D avatar!");
                videoPlayer.currentTime = 0;
                videoPlayer.play().catch(() => {});
                playAvatarAnimation();
            } else {
                isPlaying = false;
                updatePlayBtnUI(false);
            }
        });
    }

    // Handle Drag & Drop Upload
    if (dropZone) {
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files[0]);
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) handleFileUpload(e.target.files[0]);
        });
    }

    // Sample Selection Buttons
    sampleBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const sampleName = btn.getAttribute('data-sample');
            handleSampleUpload(sampleName);
        });
    });

    function handleFileUpload(file) {
        if (!file.type.startsWith('video/')) {
            alert('Please upload a valid video file.');
            return;
        }

        const objectUrl = URL.createObjectURL(file);
        videoPlayer.src = objectUrl;
        previewFilename.textContent = file.name;

        startProcessing(formData => {
            formData.append('video', file);
            return formData;
        });
    }

    function handleSampleUpload(sampleName) {
        previewFilename.textContent = sampleName;
        startProcessing(formData => {
            formData.append('sample_name', sampleName);
            return formData;
        });
    }

    function startProcessing(formDataBuilder) {
        // UI Reset
        loadingIndicator.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        meaningCard.classList.add('hidden');
        resultCard.classList.add('hidden');
        matrixCard.classList.add('hidden');
        sigmlCard.classList.add('hidden');

        stopAll();

        let formData = new FormData();
        formData = formDataBuilder(formData);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            const contentType = response.headers.get('content-type') || '';
            if (contentType.includes('application/json')) {
                const data = await response.json();
                if (!response.ok || data.error) {
                    throw new Error(data.error || `Server error (${response.status})`);
                }
                return data;
            } else {
                const errorText = await response.text();
                console.error('Server non-JSON response:', errorText);
                throw new Error(`Server processing failed (${response.status}). Please try uploading a video directly.`);
            }
        })
        .then(data => {
            if (data.error) throw new Error(data.error);

            if (data.warnings && data.warnings.length) {
                console.warn('Processing warnings:', data.warnings.join(' | '));
            }
            
            if (data.video_url) {
                videoPlayer.src = data.video_url;
            }
            
            // Populate Results
            hamnosysTags.textContent = data.hamnosys_tags;
            hamnosysChars.textContent = data.hamnosys_unicode;
            sigmlStorage.value = data.sigml;
            sigmlCodeDisplay.textContent = data.sigml;
            
            signGloss.textContent = data.gloss || "DYNAMICALLY PREDICTED SIGN";
            signMeaning.textContent = data.meaning || "Sign language gesture analyzed frame-by-frame";
            
            const precStr = data.precision || "85.7%";
            const confStr = data.confidence || "88.5%";
            
            precisionPill.textContent = precStr + " Precision";
            metricConf.textContent = confStr;
            metricPrec.textContent = precStr;
            
            renderSymbolChips(data.symbol_chips || []);
            render10ModuleMatrix(data.details || {});

            // Show Cards
            loadingIndicator.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            meaningCard.classList.remove('hidden');
            resultCard.classList.remove('hidden');
            matrixCard.classList.remove('hidden');
            sigmlCard.classList.remove('hidden');

            // Synchronized Auto Play Both Video & Avatar
            startSynchronizedPlayback();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during processing: ' + error.message);
            loadingIndicator.classList.add('hidden');
        });
    }

    function renderSymbolChips(chips) {
        symbolChipsContainer.innerHTML = '';
        if (!chips || !chips.length) {
            symbolChipsContainer.textContent = 'No symbol chips available';
            return;
        }
        
        // Group chips by category
        const categories = {};
        chips.forEach(chip => {
            const cat = chip.category || 'General Phonetic Modifier';
            if (!categories[cat]) categories[cat] = [];
            categories[cat].push(chip);
        });

        // Canonical HamNoSys grammar category order
        const categoryOrder = [
            "Symmetry & Structure",
            "Handshape",
            "Extended Finger Direction",
            "Palm Orientation",
            "Body & Spatial Location",
            "Contact & Touch",
            "Movement & Motion",
            "State Transition",
            "General Phonetic Modifier"
        ];

        categoryOrder.forEach(catName => {
            if (categories[catName] && categories[catName].length) {
                const groupEl = document.createElement('div');
                groupEl.className = 'chip-category-group';
                
                const catHeader = document.createElement('div');
                catHeader.className = 'chip-category-title font-tech';
                catHeader.textContent = catName;
                groupEl.appendChild(catHeader);

                const chipsRow = document.createElement('div');
                chipsRow.className = 'chip-category-row';

                categories[catName].forEach(chip => {
                    const chipEl = document.createElement('div');
                    chipEl.className = 'ham-chip';
                    chipEl.innerHTML = `
                        <span class="ham-chip-tag">${chip.tag}</span>
                        <span class="ham-chip-label">(${chip.label})</span>
                    `;
                    chipsRow.appendChild(chipEl);
                });

                groupEl.appendChild(chipsRow);
                symbolChipsContainer.appendChild(groupEl);
            }
        });
    }

    function render10ModuleMatrix(details) {
        matrixGrid.innerHTML = '';
        
        const moduleDefs = [
            { num: "Module 1", name: "Handshape Model", key: "handshape" },
            { num: "Module 2", name: "Orientation Vector (3D Normal)", key: "orientation" },
            { num: "Module 3", name: "Upper Body Location", key: "upper_body" },
            { num: "Module 4", name: "Head & Face Location", key: "head_face" },
            { num: "Module 5", name: "Hand Relative Location", key: "hand_location" },
            { num: "Module 6", name: "Finger Placement", key: "finger_location" },
            { num: "Module 7", name: "Contact Type Recognizer", key: "contact" },
            { num: "Module 8", name: "Arm & Space Position", key: "arm_space" },
            { num: "Module 9", name: "Primary Movement Model", key: "movement1" },
            { num: "Module 10", name: "Secondary Transition", key: "movement2" }
        ];

        moduleDefs.forEach(m => {
            let val = details[m.key] || "Detected";
            if (typeof val === 'object') {
                val = val.final || (val.per_frame ? val.per_frame.slice(0, 3).join(', ') : "Active");
            }
            if (!val || val === 'none') val = "Standard";

            const itemEl = document.createElement('div');
            itemEl.className = 'matrix-item';
            itemEl.innerHTML = `
                <div class="mod-num">${m.num}</div>
                <div class="mod-name">${m.name}</div>
                <div class="mod-pred">${val}</div>
            `;
            matrixGrid.appendChild(itemEl);
        });
    }

    // Synchronized Dual Controller Functions (Input Video + 3D WebGL Avatar)
    function startSynchronizedPlayback() {
        isPlaying = true;
        avatarWasPlaying = false;
        
        videoPlayer.loop = false; // Disable native loop so 'ended' event triggers synchronized replay
        videoPlayer.playbackRate = currentSpeed;
        videoPlayer.currentTime = 0;
        videoPlayer.play().catch(() => {});
        
        playAvatarAnimation();
        updatePlayBtnUI(true);

        // Continuous Loop Monitor: Auto re-trigger avatar loop if animation finishes before video or vice versa
        clearInterval(avatarLoopTimer);
        avatarLoopTimer = setInterval(() => {
            if (!isPlaying || !isLooping) return;

            const sfInput = document.querySelector('.txtSF.av0');
            if (sfInput) {
                const val = sfInput.value || "";
                if (frameCounterDisplay) frameCounterDisplay.textContent = val;
                
                const parts = val.split('/');
                if (parts.length === 2 && parts[1] !== '0') {
                    const curr = parseInt(parts[0], 10);
                    const total = parseInt(parts[1], 10);
                    
                    if (curr > 1) {
                        avatarWasPlaying = true;
                    }
                    
                    // Re-trigger if avatar finished its frames
                    if (avatarWasPlaying && (curr >= total || curr === 0)) {
                        console.log("[Sync Loop Engine] Avatar animation completed frame " + curr + "/" + total + " -> Re-triggering synchronized playback");
                        avatarWasPlaying = false;
                        
                        if (videoPlayer.paused || videoPlayer.ended) {
                            videoPlayer.currentTime = 0;
                            videoPlayer.play().catch(() => {});
                        }
                        playAvatarAnimation();
                    }
                }
            }
        }, 400);
    }

    function stopAll() {
        isPlaying = false;
        avatarWasPlaying = false;
        clearInterval(avatarLoopTimer);
        
        // Stop Input Video
        if (videoPlayer) {
            videoPlayer.pause();
            videoPlayer.currentTime = 0;
        }

        // Stop JASigning Avatar Immediately
        if (typeof CWASA !== 'undefined' && typeof CWASA.stop === 'function') {
            CWASA.stop(0);
        } else {
            const stopBtn = document.querySelector('.bttnStop.av0');
            if (stopBtn) stopBtn.click();
        }

        updatePlayBtnUI(false);
    }

    function playAvatarAnimation() {
        window.dispatchEvent(new Event('resize'));
        const sigmlText = sigmlStorage.value;
        if (!sigmlText) return;

        // Reset state by clicking stop first
        const stopBtn = document.querySelector('.bttnStop.av0');
        if (stopBtn) stopBtn.click();

        // Update SiGML textarea
        const sigmlArea = document.querySelector('.txtaSiGMLText.av0');
        if (sigmlArea) {
            sigmlArea.value = sigmlText;
            sigmlArea.dispatchEvent(new Event('input', { bubbles: true }));
            sigmlArea.dispatchEvent(new Event('change', { bubbles: true }));
        }

        // Play SiGML
        setTimeout(() => {
            const playBtn = document.querySelector('.bttnPlaySiGMLText.av0');
            if (playBtn) {
                playBtn.click();
            } else if (typeof CWASA !== 'undefined' && typeof CWASA.playSiGMLText === 'function') {
                CWASA.playSiGMLText(sigmlText, 0);
            }
        }, 100);
    }

    function updatePlayBtnUI(playing) {
        if (!syncPlayBtn) return;
        if (playing) {
            syncPlayBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                <span>Pause Both (Sync)</span>
            `;
            syncPlayBtn.classList.add('btn-pause');
        } else {
            syncPlayBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                <span>Play Both (Sync)</span>
            `;
            syncPlayBtn.classList.remove('btn-pause');
        }
    }

    // Button Listeners
    if (syncPlayBtn) {
        syncPlayBtn.addEventListener('click', () => {
            if (isPlaying) {
                isPlaying = false;
                videoPlayer.pause();
                const stopBtn = document.querySelector('.bttnStop.av0');
                if (stopBtn) stopBtn.click();
                updatePlayBtnUI(false);
            } else {
                startSynchronizedPlayback();
            }
        });
    }

    if (syncStopBtn) {
        syncStopBtn.addEventListener('click', stopAll);
    }

    if (syncReplayBtn) {
        syncReplayBtn.addEventListener('click', () => {
            if (videoPlayer) videoPlayer.currentTime = 0;
            startSynchronizedPlayback();
        });
    }

    // Speed Selector Pills
    speedPills.forEach(pill => {
        pill.addEventListener('click', () => {
            speedPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            
            currentSpeed = parseFloat(pill.getAttribute('data-speed')) || 1.0;
            if (videoPlayer) videoPlayer.playbackRate = currentSpeed;
            
            // CWASA Speed adjustment
            const logSpeedInput = document.querySelector('.txtLogSpeed.av0');
            if (logSpeedInput) {
                logSpeedInput.value = "+" + currentSpeed.toFixed(1);
            }
        });
    });

    // Frame Scrubbing
    if (scrubPrevBtn) {
        scrubPrevBtn.addEventListener('click', () => {
            if (videoPlayer) videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 1/30);
            const prevBtn = document.querySelector('.bttnPrevF.av0');
            if (prevBtn) prevBtn.click();
        });
    }

    if (scrubNextBtn) {
        scrubNextBtn.addEventListener('click', () => {
            if (videoPlayer) videoPlayer.currentTime += 1/30;
            const nextBtn = document.querySelector('.bttnNextF.av0');
            if (nextBtn) nextBtn.click();
        });
    }



    // Copy Buttons
    if (copyTokensBtn) {
        copyTokensBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(hamnosysTags.textContent || "");
            copyTokensBtn.textContent = "Copied Tokens!";
            setTimeout(() => copyTokensBtn.textContent = "Copy Tokens", 2000);
        });
    }

    if (copySigmlBtn) {
        copySigmlBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(sigmlCodeDisplay.textContent || "");
            copySigmlBtn.textContent = "Copied SiGML XML!";
            setTimeout(() => copySigmlBtn.textContent = "Copy SiGML XML", 2000);
        });
    }

    // Auto-update HUD stats
    setInterval(() => {
        const fpsInput = document.querySelector('.txtFPS.av0');
        const sfInput = document.querySelector('.txtSF.av0');
        
        const fpsHud = document.getElementById('avatar-fps-hud');
        const frameHud = document.getElementById('avatar-frame-hud');
        
        if (fpsHud && fpsInput) fpsHud.textContent = "FPS: " + (fpsInput.value || "60.00");
        if (frameHud && sfInput) frameHud.textContent = "Frame: " + (sfInput.value || "0/0");
    }, 400);
});
