// Project Portal Interactive Controller
const SIGN_CATALOG = {
  zero: {
    title: "0 or Zero",
    dataset: "ISL",
    datasetLabel: "ISL • Digit",
    badgeClass: "isl-badge",
    video: "assets/videos/zero_sign.mp4",
    fileName: "0 or Zero.mp4",
    githubPath: "FYP-DataSet/ISL/videos/0 or Zero.mp4",
    githubUrl: "https://github.com/SaiSriSatvic2005/FYP-DataSet/blob/main/ISL/videos/0%20or%20Zero.mp4",
    gloss: "ZERO",
    meaning: "Single-hand sign: all fingertips form an anatomical O-shape meeting the thumb at chest level.",
    hamnosys: "<hamceeall/> <hamextfingeruo/> <hampalml/> <hamchest/>",
    sigml: `<?xml version="1.0" encoding="utf-8"?>
<sigml>
<hns_sign gloss="zero">
  <hamnosys_nonmanual></hamnosys_nonmanual>
  <hamnosys_manual>
    <hamceeall/>
    <hamextfingeruo/>
    <hampalml/>
    <hamchest/>
  </hamnosys_manual>
</hns_sign>
</sigml>`
  },
  one: {
    title: "1 or One",
    dataset: "ISL",
    datasetLabel: "ISL • Digit",
    badgeClass: "isl-badge",
    video: "assets/videos/one_sign.mp4",
    fileName: "1 or One.mp4",
    githubPath: "FYP-DataSet/ISL/videos/1 or One.mp4",
    githubUrl: "https://github.com/SaiSriSatvic2005/FYP-DataSet/blob/main/ISL/videos/1%20or%20One.mp4",
    gloss: "ONE",
    meaning: "Single-hand sign: index finger extended upright with palm facing towards the chest/signer.",
    hamnosys: "<hamfinger2/> <hamthumbacrossmod/> <hamextfingeru/> <hampalml/> <hamchest/>",
    sigml: `<?xml version="1.0" encoding="utf-8"?>
<sigml>
<hns_sign gloss="one">
  <hamnosys_nonmanual></hamnosys_nonmanual>
  <hamnosys_manual>
    <hamfinger2/>
    <hamthumbacrossmod/>
    <hamextfingeru/>
    <hampalml/>
    <hamchest/>
  </hamnosys_manual>
</hns_sign>
</sigml>`
  },
  two: {
    title: "2 or Two",
    dataset: "ISL",
    datasetLabel: "ISL • Digit",
    badgeClass: "isl-badge",
    video: "assets/videos/two_sign.mp4",
    fileName: "2 or Two.mp4",
    githubPath: "FYP-DataSet/ISL/videos/2 or Two.mp4",
    githubUrl: "https://github.com/SaiSriSatvic2005/FYP-DataSet/blob/main/ISL/videos/2%20or%20Two.mp4",
    gloss: "TWO",
    meaning: "Single-hand sign: index and middle finger extended in an upright V-formation with thumb folded.",
    hamnosys: "<hamfinger23/> <hamthumboutmod/> <hamextfingeru/> <hampalmdl/> <hamchest/> <hammover/>",
    sigml: `<?xml version="1.0" encoding="utf-8"?>
<sigml>
  <hns_sign gloss="two">
    <hamnosys_nonmanual>
    </hamnosys_nonmanual>
    <hamnosys_manual>
      <hamfinger23/>
      <hamthumboutmod/>
      <hamextfingeru/>
      <hampalmdl/>
      <hamchest/>
      <hammover/>
    </hamnosys_manual>
  </hns_sign>
</sigml>`
  }
};

let currentSignKey = "zero";

function initPortal() {
  const signSelect = document.getElementById('sign-select');
  if (signSelect) {
    signSelect.addEventListener('change', (e) => {
      switchSign(e.target.value);
    });
  }

  const playBtn = document.getElementById('btn-play-sync');
  if (playBtn) {
    playBtn.addEventListener('click', playSynchronized);
  }

  const speedSelect = document.getElementById('speed-select');
  if (speedSelect) {
    speedSelect.addEventListener('change', (e) => {
      const speedVal = parseFloat(e.target.value);
      const video = document.getElementById('demo-video');
      if (video) video.playbackRate = speedVal;
      
      const speedInput = document.querySelector('.txtLogSpeed.av0');
      if (speedInput) {
        speedInput.value = (speedVal === 0.5 ? "-1.0" : (speedVal === 1.5 ? "+1.5" : "+1.0"));
      }
    });
  }

  switchSign('zero');
}

function switchSign(key) {
  if (!SIGN_CATALOG[key]) return;
  currentSignKey = key;

  const sign = SIGN_CATALOG[key];

  const signSelect = document.getElementById('sign-select');
  if (signSelect && signSelect.value !== key) {
    signSelect.value = key;
  }

  const glossBadge = document.getElementById('demo-gloss');
  if (glossBadge) glossBadge.textContent = sign.gloss;

  const metaDesc = document.getElementById('meta-desc');
  if (metaDesc) metaDesc.textContent = sign.meaning;

  const metaBadge = document.getElementById('meta-dataset-badge');
  if (metaBadge) {
    metaBadge.textContent = sign.datasetLabel;
    metaBadge.className = 'slide-dataset-badge ' + sign.badgeClass;
  }

  const video = document.getElementById('demo-video');
  if (video) {
    video.pause();
    video.muted = true;
    video.src = sign.video;
    video.load();

    const onLoaded = () => {
      video.currentTime = 0.05;
      video.removeEventListener('loadeddata', onLoaded);
    };
    video.addEventListener('loadeddata', onLoaded);
  }

  // 1. Direct navigation to the EXACT .mp4 file in GitHub repository
  const openLink = document.getElementById('btn-open-video-tab');
  if (openLink) {
    openLink.href = sign.githubUrl;
    openLink.target = "_blank";
    openLink.rel = "noopener noreferrer";
    openLink.title = `Navigate to ${sign.githubPath} on GitHub`;
  }

  const openText = document.getElementById('open-video-text');
  if (openText) {
    openText.textContent = `GitHub .mp4 File ↗`;
  }

  // 2. Mention exact destination path clearly in metadata bar
  const metaLink = document.getElementById('meta-file-link');
  if (metaLink) {
    metaLink.href = sign.githubUrl;
    metaLink.target = "_blank";
    metaLink.rel = "noopener noreferrer";
    metaLink.title = `Navigate to ${sign.githubPath} on GitHub`;
    metaLink.innerHTML = `<svg class="icon-svg" viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/></svg> <span>Path: ${sign.githubPath}</span> &nearr;`;
  }

  const txta = document.querySelector('.txtaSiGMLText.av0');
  if (txta) {
    txta.value = sign.sigml;
  }
}

function playSynchronized() {
  const sign = SIGN_CATALOG[currentSignKey];
  if (!sign) return;

  const video = document.getElementById('demo-video');
  if (video) {
    video.currentTime = 0;
    video.play().catch(e => console.log("Video auto-play policy:", e));
  }

  if (typeof CWASA !== 'undefined' && CWASA.playSiGMLText) {
    CWASA.playSiGMLText(sign.sigml, 0);
  } else {
    const playBtn = document.querySelector('.bttnPlaySiGMLText.av0');
    if (playBtn) playBtn.click();
  }
}

window.addEventListener('DOMContentLoaded', initPortal);
