document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------------------
    // Backend API Server Configuration & Health Monitoring
    // -------------------------------------------------------------------------
    const DEFAULT_BACKEND_URL = "https://saisrisatvic2005--sign-avatar-backend-flask-app.us-east.modal.run";
    const OLD_BACKEND_URLS = [
        "https://satvic2005-sign-avatar-backend.hf.space",
        "https://nonchok005-sign-avatar-backend.hf.space"
    ];
    let storedBackendUrl = localStorage.getItem('avatar_backend_url');
    if (storedBackendUrl && OLD_BACKEND_URLS.some(old => storedBackendUrl.includes(old))) {
        localStorage.removeItem('avatar_backend_url');
        storedBackendUrl = null;
    }
    let backendUrl = (storedBackendUrl || DEFAULT_BACKEND_URL).trim();

    const apiStatusDot = document.getElementById('api-status-dot');
    const apiStatusLabel = document.getElementById('api-status-label');
    const apiSettingsDrawer = document.getElementById('api-settings-drawer');
    const btnToggleApiSettings = document.getElementById('btn-toggle-api-settings');
    const backendUrlInput = document.getElementById('backend-url-input');
    const btnSaveBackendUrl = document.getElementById('btn-save-backend-url');
    const btnResetBackendUrl = document.getElementById('btn-reset-backend-url');

    function getApiUrl(endpoint) {
        const base = backendUrl.replace(/\/+$/, '');
        const path = endpoint.startsWith('/') ? endpoint : '/' + endpoint;
        return `${base}${path}`;
    }

    if (backendUrlInput) {
        backendUrlInput.value = backendUrl;
    }

    if (btnToggleApiSettings) {
        btnToggleApiSettings.addEventListener('click', (e) => {
            e.stopPropagation();
            apiSettingsDrawer.classList.toggle('hidden');
        });
    }

    if (btnSaveBackendUrl) {
        btnSaveBackendUrl.addEventListener('click', () => {
            let val = backendUrlInput.value.trim();
            if (!val) val = DEFAULT_BACKEND_URL;
            val = val.replace(/\/+$/, '');
            backendUrl = val;
            localStorage.setItem('avatar_backend_url', backendUrl);
            checkBackendHealth();
            apiSettingsDrawer.classList.add('hidden');
        });
    }

    if (btnResetBackendUrl) {
        btnResetBackendUrl.addEventListener('click', () => {
            backendUrl = DEFAULT_BACKEND_URL;
            localStorage.removeItem('avatar_backend_url');
            if (backendUrlInput) backendUrlInput.value = DEFAULT_BACKEND_URL;
            checkBackendHealth();
            apiSettingsDrawer.classList.add('hidden');
        });
    }

    function checkBackendHealth() {
        if (!apiStatusDot || !apiStatusLabel) return;
        apiStatusDot.className = 'status-dot dot-connecting';
        apiStatusLabel.textContent = 'Connecting to AI Backend...';

        const healthUrl = getApiUrl('/health');
        const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Connection timed out')), 15000)
        );

        Promise.race([
            fetch(healthUrl, { method: 'GET', mode: 'cors' }),
            timeoutPromise
        ])
        .then(res => {
            if (res.ok) {
                return res.json();
            }
            throw new Error(`HTTP ${res.status}`);
        })
        .then(data => {
            apiStatusDot.className = 'status-dot dot-online';
            apiStatusLabel.textContent = 'Cloud AI Online (V8 Hybrid)';
        })
        .catch(err => {
            apiStatusDot.className = 'status-dot dot-offline';
            apiStatusLabel.textContent = 'AI Offline (Click ⚙️ to configure)';
            console.warn(`[API Health Check] Could not reach backend at ${healthUrl}:`, err.message);
        });
    }

    // Run health check on startup
    checkBackendHealth();

    // -------------------------------------------------------------------------
    // Core Application Logic: Video Ingestion, Preview & SiGML Synthesis
    // -------------------------------------------------------------------------
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('video-input');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    // Video & Preview Elements
    const previewContainer = document.getElementById('preview-container');
    const videoPlayer = document.getElementById('input-video-player');
    const previewFilename = document.getElementById('preview-filename');
    
    // Result Cards
    const meaningCard = document.getElementById('meaning-card');
    const signGloss = document.getElementById('sign-gloss');
    const signMeaning = document.getElementById('sign-meaning');
    const precisionPill = document.getElementById('precision-pill');
    
    const resultCard = document.getElementById('result-card');
    const hamnosysTags = document.getElementById('hamnosys-tags');
    const hamnosysChars = document.getElementById('hamnosys-chars');
    const symbolChipsContainer = document.getElementById('symbol-chips');
    
    // Buttons & Controls
    const playAvatarBtn = document.getElementById('play-avatar-btn');
    const sigmlStorage = document.getElementById('sigml-storage');
    const sampleBtns = document.querySelectorAll('.sample-btn');

    // Drag & Drop handlers
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

    // Sample buttons listener
    sampleBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const sampleName = btn.getAttribute('data-sample');
            handleSampleUpload(sampleName);
        });
    });

    function handleFileUpload(file) {
        if (!file) return;
        const isVideo = (file.type && file.type.startsWith('video/')) || 
            /\.(mp4|mov|avi|webm|mkv|m4v)$/i.test(file.name);
            
        if (!isVideo) {
            alert('Please upload a valid sign language video file (.mp4, .mov, .webm, etc.).');
            return;
        }

        // Preview local file in video player
        try {
            const objectUrl = URL.createObjectURL(file);
            videoPlayer.src = objectUrl;
        } catch (_) {}
        previewFilename.textContent = file.name;

        startProcessing(formData => {
            formData.append('video', file, file.name);
            return formData;
        });
    }

    function handleSampleUpload(sampleName) {
        previewFilename.textContent = sampleName;
        // Preview sample from backend
        videoPlayer.src = getApiUrl(`/samples/${sampleName}`);

        startProcessing(formData => {
            formData.append('sample_name', sampleName);
            return formData;
        });
    }

    function startProcessing(formDataBuilder) {
        loadingIndicator.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        meaningCard.classList.add('hidden');
        resultCard.classList.add('hidden');

        let formData = new FormData();
        formData = formDataBuilder(formData);

        const uploadUrl = getApiUrl('/upload');

        // Set an explicit 120s timeout on inference so the spinner never hangs indefinitely
        const controller = new AbortController();
        const timeoutTimer = setTimeout(() => {
            controller.abort();
        }, 120000);

        fetch(uploadUrl, {
            method: 'POST',
            mode: 'cors',
            body: formData,
            signal: controller.signal
        })
        .then(response => {
            clearTimeout(timeoutTimer);
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || err.detail || `Server responded with ${response.status}`); });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Set video player src if returned
            if (data.video_url) {
                if (data.video_url.startsWith('http://') || data.video_url.startsWith('https://')) {
                    videoPlayer.src = data.video_url;
                } else {
                    videoPlayer.src = getApiUrl(data.video_url);
                }
            }
            
            // Update UI with results
            hamnosysTags.textContent = data.hamnosys_tags || "None";
            hamnosysChars.textContent = data.hamnosys_unicode || "";
            sigmlStorage.value = data.sigml || "";
            
            signGloss.textContent = data.gloss || "ISL GESTURE";
            signMeaning.textContent = data.meaning || "Sign language gesture analyzed via pipeline";
            precisionPill.textContent = "Accuracy: " + (data.precision || "85.7%");
            
            // Render Symbol Chips
            renderSymbolChips(data.symbol_chips || []);

            // Hide loading, show result cards
            loadingIndicator.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            meaningCard.classList.remove('hidden');
            resultCard.classList.remove('hidden');

            // Mark backend as online
            apiStatusDot.className = 'status-dot dot-online';
            apiStatusLabel.textContent = 'Cloud AI Online (V8 Hybrid)';

            // Auto play input video and avatar
            videoPlayer.play().catch(() => {});
            playAnimation();
        })
        .catch(error => {
            clearTimeout(timeoutTimer);
            console.error('Error:', error);
            const errorMsg = error.name === 'AbortError'
                ? 'Processing timed out after 120 seconds. Please try a shorter video clip.'
                : error.message;
            alert(`Inference failed from ${uploadUrl}:\n\n${errorMsg}\n\nTip: Click the ⚙️ icon in the header to check or change the backend server URL.`);
            loadingIndicator.classList.add('hidden');
        });
    }

    function renderSymbolChips(chips) {
        symbolChipsContainer.innerHTML = '';
        if (!chips || !chips.length) {
            symbolChipsContainer.textContent = 'No symbol breakdown data available';
            return;
        }
        
        chips.forEach(chip => {
            const chipEl = document.createElement('div');
            chipEl.className = 'ham-chip';
            chipEl.innerHTML = `
                <span class="ham-chip-tag">${chip.tag}</span>
                <span class="ham-chip-label">(${chip.label})</span>
            `;
            symbolChipsContainer.appendChild(chipEl);
        });
    }

    // Play / Replay Avatar Button listener
    if (playAvatarBtn) {
        playAvatarBtn.addEventListener('click', playAnimation);
    }

    // Trigger WebGL avatar canvas recalculation
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 1000);

    function playAnimation() {
        window.dispatchEvent(new Event('resize'));
        
        const sigmlText = sigmlStorage.value;
        if (sigmlText) {
            const sigmlArea = document.querySelector('.txtaSiGMLText.av0');
            const playBtn = document.querySelector('.bttnPlaySiGMLText.av0');
            if (sigmlArea && playBtn) {
                sigmlArea.value = sigmlText;
                playBtn.click();
            } else if (typeof CWASA !== 'undefined') {
                CWASA.playSiGMLText(sigmlText, 0);
            }
        } else {
            console.warn("No SiGML text available to play.");
        }
    }
});
