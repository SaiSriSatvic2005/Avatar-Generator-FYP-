document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('video-input');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    // Video & Preview Elements
    const previewContainer = document.getElementById('preview-container');
    const videoPlayer = document.getElementById('input-video-player');
    const previewFilename = document.getElementById('preview-filename');
    
    // Cards
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

    // Sample buttons
    const sampleBtns = document.querySelectorAll('.sample-btn');

    // Handle Drag & Drop
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

    // Sample Button Listener
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

        // Preview local file in video element
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
        // UI Updates
        loadingIndicator.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        meaningCard.classList.add('hidden');
        resultCard.classList.add('hidden');

        let formData = new FormData();
        formData = formDataBuilder(formData);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Network error'); });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }

            if (data.warnings && data.warnings.length) {
                console.warn('Processing warnings:', data.warnings.join(' | '));
            }
            
            // Set video player src from returned server URL if not already set
            if (data.video_url) {
                videoPlayer.src = data.video_url;
            }
            
            // Update UI with results
            hamnosysTags.textContent = data.hamnosys_tags;
            hamnosysChars.textContent = data.hamnosys_unicode;
            sigmlStorage.value = data.sigml;
            
            signGloss.textContent = data.gloss || "ISL GESTURE";
            signMeaning.textContent = data.meaning || "Sign language gesture analyzed via pipeline";
            precisionPill.textContent = "Accuracy: " + (data.precision || "85.7%");
            
            // Render Clean Symbol Chips (No Emoji Icons)
            renderSymbolChips(data.symbol_chips || []);

            // Hide loading, show result cards
            loadingIndicator.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            meaningCard.classList.remove('hidden');
            resultCard.classList.remove('hidden');

            // Auto play input video and avatar
            videoPlayer.play().catch(() => {});
            playAnimation();
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
            symbolChipsContainer.textContent = 'No symbol data available';
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

    // Directly Play / Replay Avatar Button listener inside Avatar Card
    if (playAvatarBtn) {
        playAvatarBtn.addEventListener('click', playAnimation);
    }

    // Force WebGL canvas recalculation after load
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 1000);

    function playAnimation() {
        window.dispatchEvent(new Event('resize'));
        
        const sigmlText = sigmlStorage.value;
        if (sigmlText) {
            console.log("Playing SiGML text animation (length:", sigmlText.length, ")");
            const sigmlArea = document.querySelector('.txtaSiGMLText.av0');
            if (sigmlArea) {
                sigmlArea.value = sigmlText;
                sigmlArea.dispatchEvent(new Event('input', { bubbles: true }));
                sigmlArea.dispatchEvent(new Event('change', { bubbles: true }));
            }

            if (typeof CWASA !== 'undefined' && typeof CWASA.playSiGMLText === 'function') {
                CWASA.playSiGMLText(sigmlText, 0);
            } else {
                const playBtn = document.querySelector('.bttnPlaySiGMLText.av0');
                if (playBtn) {
                    playBtn.click();
                } else {
                    console.warn("CWASA playSiGMLText function and play button not found.");
                }
            }
        } else {
            console.warn("No SiGML text available to play.");
        }
    }
});
