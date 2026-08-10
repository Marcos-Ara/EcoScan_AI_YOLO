const screens = Array.from(document.querySelectorAll('.screen'));
const startScanBtn = document.getElementById('startScanBtn');
const saveBtn = document.getElementById('saveBtn');
const themeSwitch = document.getElementById('themeSwitch');
const notifSwitch = document.getElementById('notifSwitch');
const loginForm = document.getElementById('loginForm');
const goRegisterBtn = document.getElementById('goRegisterBtn');
const logoutBtn = document.getElementById('logoutBtn');

const cameraFeed = document.getElementById('cameraFeed');
const overlay = document.getElementById('overlay');
const cameraFallback = document.getElementById('cameraFallback') || null;
const detName = document.getElementById('detName');
const detType = document.getElementById('detType');
const detBin = document.getElementById('detBin');
const detDest = document.getElementById('detDest');
const detTime = document.getElementById('detTime');
const detFact = document.getElementById('detFact');

const API_BASE = window.ECOSCAN_API_BASE || 'http://127.0.0.1:8000';
const DETECTION_INTERVAL_MS = 140;
const MIN_CONFIDENCE = 0.35;
const MAX_BOXES = 5;

let currentScreen = 'loginScreen';
let stream = null;
let detectionLoopActive = false;
let frameCanvas = null;
let frameContext = null;
let lastPredictions = [];
let lastDetectionData = null;
let savedDetections = loadSavedDetections();

const WASTE_RULES = {
  papel: {
    category: 'Papel',
    bin: '🔵 Azul',
    dest: 'Reciclagem',
    time: '3–6 meses',
    fact: 'Papel e papelão devem estar, de preferência, secos e sem restos de comida.'
  },
  plastico: {
    category: 'Plástico',
    bin: '🔴 Vermelha',
    dest: 'Reciclagem',
    time: 'Varia por material',
    fact: 'Garrafas PET, embalagens e outros plásticos devem ir para a coleta seletiva.'
  },
  vidro: {
    category: 'Vidro',
    bin: '🟢 Verde',
    dest: 'Reciclagem',
    time: 'Muito longo',
    fact: 'Vidro pode ser reciclado repetidamente. O modelo final deverá reconhecer vidro.'
  },
  metal: {
    category: 'Metal',
    bin: '🟡 Amarela',
    dest: 'Reciclagem',
    time: 'Varia por material',
    fact: 'Latas, tampas e outros metais devem ser encaminhados para reciclagem.'
  },
  organico: {
    category: 'Orgânico',
    bin: '🟤 Marrom',
    dest: 'Compostagem',
    time: 'Varia',
    fact: 'Restos de alimentos e cascas podem ser destinados à compostagem.'
  },
  rejeito: {
    category: 'Rejeito',
    bin: '⚫ Cinza/Preta',
    dest: 'Rejeitos',
    time: 'Varia',
    fact: 'O modelo final deverá identificar materiais que não entram na coleta seletiva.'
  }
};

// GreenSorter/YOLOv7 atualmente possui estas quatro classes.
// Não fingimos que vidro, orgânico ou rejeito são detectados até haver pesos treinados para eles.
const GREEN_SORTER_MAP = {
  cardboard: 'papel',
  metal: 'metal',
  rigid_plastic: 'plastico',
  soft_plastic: 'plastico'
};

const CATEGORY_COLORS = {
  Papel: '#2f6ef3',
  Plástico: '#df4b42',
  Vidro: '#43a047',
  Metal: '#d6a800',
  Orgânico: '#8b5a2b',
  Rejeito: '#5b6068',
  Indeterminado: '#43a047'
};

init();

function init() {
  applyTheme(getTheme());

  document.addEventListener('click', handleGlobalClicks);

  loginForm?.addEventListener('submit', (e) => {
    e.preventDefault();
    navigateTo('homeScreen');
  });

  goRegisterBtn?.addEventListener('click', () => navigateTo('homeScreen'));
  startScanBtn?.addEventListener('click', () => navigateTo('cameraScreen'));
  saveBtn?.addEventListener('click', saveCurrentDetection);
  logoutBtn?.addEventListener('click', () => navigateTo('loginScreen'));

  themeSwitch?.addEventListener('click', toggleTheme);
  notifSwitch?.addEventListener('click', () => notifSwitch.classList.toggle('on'));

  window.addEventListener('resize', resizeOverlay);

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopCamera();
  });

  window.addEventListener('beforeunload', stopCamera);

  if (window.lucide) lucide.createIcons();
  renderAll();
}

function handleGlobalClicks(e) {
  const go = e.target.closest('[data-go]');
  if (!go) return;

  const target = go.getAttribute('data-go');
  if (target) navigateTo(target);
}

function navigateTo(screenId) {
  if (screenId === 'cameraScreen') {
    openCamera();
  } else if (currentScreen === 'cameraScreen' && screenId !== 'cameraScreen') {
    stopCamera();
  }

  currentScreen = screenId;
  screens.forEach((screen) => screen.classList.remove('active'));

  const target = document.getElementById(screenId);
  if (target) target.classList.add('active');

  updateNavState(screenId);

  if (screenId === 'historyScreen') renderHistory();
  if (screenId === 'statsScreen') renderStats();
  if (screenId === 'achievementsScreen') renderAchievements();

  if (window.lucide) lucide.createIcons();
}

function updateNavState(screenId) {
  document.querySelectorAll('.nav-item').forEach((btn) => {
    btn.classList.toggle('active', btn.getAttribute('data-go') === screenId);
  });
}

function getTheme() {
  return localStorage.getItem('ecoscan-theme') || 'light';
}

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  localStorage.setItem('ecoscan-theme', theme);
  setThemeSwitchUI();
}

function toggleTheme() {
  applyTheme(getTheme() === 'light' ? 'dark' : 'light');
}

function setThemeSwitchUI() {
  themeSwitch?.classList.toggle('on', getTheme() === 'dark');
}

function loadSavedDetections() {
  try {
    return JSON.parse(localStorage.getItem('ecoscan-detections') || '[]');
  } catch {
    return [];
  }
}

function persistDetections() {
  localStorage.setItem('ecoscan-detections', JSON.stringify(savedDetections));
}

function renderAll() {
  renderHomeStats();
  renderHistory();
  renderStats();
  renderAchievements();
}

function renderHomeStats() {
  document.getElementById('totalCount').textContent = savedDetections.length;
  document.getElementById('recycleCount').textContent =
    savedDetections.filter(d => ['Papel', 'Plástico', 'Vidro', 'Metal'].includes(d.category)).length;
  document.getElementById('organicCount').textContent =
    savedDetections.filter(d => d.category === 'Orgânico').length;
}

function renderHistory() {
  const list = document.getElementById('historyList');
  if (!list) return;

  if (!savedDetections.length) {
    list.innerHTML = '<p class="empty">Nenhuma detecção salva ainda.</p>';
    return;
  }

  list.innerHTML = savedDetections.slice().reverse().map((d) => {
    const date = new Date(d.detectedAt).toLocaleDateString('pt-BR');
    const confidence = Number.isFinite(d.confidence) ? ` • ${(d.confidence * 100).toFixed(0)}%` : '';
    return `
      <article class="info-card">
        <div style="display:flex;justify-content:space-between;gap:16px;align-items:flex-start">
          <div>
            <h3 style="margin-bottom:6px">${escapeHTML(d.name)}</h3>
            <p style="margin:0;color:var(--muted)">
              ${escapeHTML(d.category)} • ${escapeHTML(d.bin)}${confidence}
            </p>
          </div>
          <small style="color:var(--muted)">${date}</small>
        </div>
      </article>
    `;
  }).join('');
}

function renderStats() {
  const counts = {
    Orgânico: 0,
    Papel: 0,
    Plástico: 0,
    Vidro: 0,
    Metal: 0,
    Rejeito: 0
  };

  savedDetections.forEach((d) => {
    if (counts[d.category] !== undefined) counts[d.category]++;
  });

  const max = Math.max(1, ...Object.values(counts));
  const setBar = (id, count) => {
    const el = document.getElementById(id);
    if (el) el.style.width = `${(count / max) * 100}%`;
  };

  const ids = {
    Orgânico: 'sbOrganic',
    Papel: 'sbPaper',
    Plástico: 'sbPlastic',
    Vidro: 'sbGlass',
    Metal: 'sbMetal',
    Rejeito: 'sbReject'
  };

  Object.entries(ids).forEach(([category, id]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = counts[category];
  });

  setBar('barOrganic', counts.Orgânico);
  setBar('barPaper', counts.Papel);
  setBar('barPlastic', counts.Plástico);
  setBar('barGlass', counts.Vidro);
  setBar('barMetal', counts.Metal);
  setBar('barReject', counts.Rejeito);

  const total = savedDetections.length;
  let level = '🌱 Iniciante Verde';
  if (total >= 50) level = '🏆 Mestre Sustentável';
  else if (total >= 25) level = '🌎 Guardião Ambiental';
  else if (total >= 10) level = '♻️ Reciclador';

  document.getElementById('ecoLevel').textContent = level;
  document.getElementById('ecoPoints').textContent = `${total * 10} pontos`;
}

function renderAchievements() {
  const total = savedDetections.length;
  document.getElementById('ach1')?.classList.toggle('locked', total < 1);
  document.getElementById('ach2')?.classList.toggle('locked', total < 10);
  document.getElementById('ach3')?.classList.toggle('locked', total < 25);
  document.getElementById('ach4')?.classList.toggle('locked', total < 50);
}

async function openCamera() {
  if (stream) return;

  try {
    if (!cameraFeed || !overlay) {
      throw new Error('Elementos da câmera não encontrados.');
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Seu navegador não suporta acesso à câmera.');
    }

    if (cameraFallback) cameraFallback.hidden = true;

    setDetectionStatus('🤖 Conectando ao detector YOLO...');

    // Resolução menor reduz o custo do envio de frames sem mudar a qualidade
    // visual da câmera exibida ao usuário.
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    cameraFeed.srcObject = stream;

    await waitForVideoMetadata(cameraFeed, 10000);
    await cameraFeed.play();
    resizeOverlay();

    // Tenta zoom 1.5x somente quando o dispositivo suportar.
    try {
      const track = stream.getVideoTracks()[0];
      const capabilities = track?.getCapabilities?.();

      if (capabilities?.zoom) {
        const zoom = Math.min(1.5, capabilities.zoom.max);
        if (zoom >= capabilities.zoom.min) {
          await track.applyConstraints({ advanced: [{ zoom }] });
        }
      }
    } catch (zoomError) {
      console.debug('Zoom não suportado:', zoomError);
    }

    const health = await fetchJson(`${API_BASE}/health`);
    if (!health?.model_loaded) {
      throw new Error('O backend está ativo, mas o modelo GreenSorter não foi carregado.');
    }

    setDetectionStatus('📷 Câmera ativa. Aponte para um resíduo.');
    startDetectionLoop();
  } catch (error) {
    console.error('Erro ao abrir câmera:', error);
    stopCamera();

    setDetectionStatus('❌ Detector indisponível.');

    if (cameraFallback) cameraFallback.hidden = false;
    alert(formatCameraError(error));
  }
}

function stopCamera() {
  detectionLoopActive = false;
  lastPredictions = [];
  lastDetectionData = null;

  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  if (cameraFeed) {
    cameraFeed.pause();
    cameraFeed.srcObject = null;
  }

  if (overlay) {
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }
}

function formatCameraError(err) {
  if (err?.message?.includes('backend') || err?.message?.includes('modelo')) {
    return `${err.message}\n\nInicie o backend e confirme que backend/model/model.pt existe.`;
  }

  if (!err?.name) return err?.message || 'Erro ao acessar a câmera.';
  if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
    return 'Permita o acesso à câmera para continuar.';
  }
  if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
    return 'Nenhuma câmera foi encontrada.';
  }
  if (err.name === 'NotReadableError') {
    return 'A câmera está em uso por outro aplicativo.';
  }
  if (err.name === 'SecurityError') {
    return 'A câmera exige HTTPS ou localhost.';
  }

  return err.message || 'Não foi possível abrir a câmera.';
}

function waitForVideoMetadata(video, timeoutMs) {
  return new Promise((resolve, reject) => {
    if (video.videoWidth && video.videoHeight) {
      resolve();
      return;
    }

    const timer = setTimeout(() => {
      video.removeEventListener('loadedmetadata', onLoaded);
      reject(new Error('Tempo excedido ao carregar a câmera.'));
    }, timeoutMs);

    function onLoaded() {
      clearTimeout(timer);
      video.removeEventListener('loadedmetadata', onLoaded);
      resolve();
    }

    video.addEventListener('loadedmetadata', onLoaded);
  });
}

function resizeOverlay() {
  if (!cameraFeed?.videoWidth || !cameraFeed?.videoHeight || !overlay) return;
  overlay.width = cameraFeed.videoWidth;
  overlay.height = cameraFeed.videoHeight;
}

async function startDetectionLoop() {
  if (detectionLoopActive) return;
  detectionLoopActive = true;

  frameCanvas = document.createElement('canvas');
  frameContext = frameCanvas.getContext('2d', { willReadFrequently: false });

  while (detectionLoopActive && currentScreen === 'cameraScreen') {
    try {
      if (cameraFeed.readyState >= 2) {
        const result = await detectFrame();
        lastPredictions = result?.predictions || [];
        drawPredictions(lastPredictions);
        updateDetectionCard(lastPredictions);
      }
    } catch (error) {
      console.error('Detection error:', error);
      setDetectionStatus('⚠️ Falha momentânea no detector...');
    }

    await sleep(DETECTION_INTERVAL_MS);
  }
}

async function detectFrame() {
  const maxWidth = 640;
  const scale = Math.min(1, maxWidth / cameraFeed.videoWidth);

  frameCanvas.width = Math.max(1, Math.round(cameraFeed.videoWidth * scale));
  frameCanvas.height = Math.max(1, Math.round(cameraFeed.videoHeight * scale));

  frameContext.drawImage(
    cameraFeed,
    0,
    0,
    frameCanvas.width,
    frameCanvas.height
  );

  const blob = await canvasToBlob(frameCanvas, 0.72);

  const formData = new FormData();
  formData.append('file', blob, 'frame.jpg');

  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
    cache: 'no-store'
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`API ${response.status}: ${message}`);
  }

  const result = await response.json();

  // O backend devolve coordenadas na imagem reduzida. Voltamos para
  // a resolução original do vídeo para desenhar no canvas.
  const sx = cameraFeed.videoWidth / frameCanvas.width;
  const sy = cameraFeed.videoHeight / frameCanvas.height;

  result.predictions = (result.predictions || []).map((p) => ({
    ...p,
    bbox: [
      p.bbox[0] * sx,
      p.bbox[1] * sy,
      p.bbox[2] * sx,
      p.bbox[3] * sy
    ]
  }));

  return result;
}

function canvasToBlob(canvas, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      blob => blob ? resolve(blob) : reject(new Error('Não foi possível criar o frame.')),
      'image/jpeg',
      quality
    );
  });
}

function drawPredictions(predictions) {
  resizeOverlay();

  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!predictions?.length) return;

  const fontSize = Math.max(15, Math.round(overlay.width / 42));
  ctx.font = `700 ${fontSize}px Outfit`;
  ctx.lineWidth = Math.max(3, Math.round(overlay.width / 500));

  predictions.slice(0, MAX_BOXES).forEach((pred) => {
    const [x, y, width, height] = pred.bbox;
    const category = pred.category || 'Indeterminado';
    const color = CATEGORY_COLORS[category] || CATEGORY_COLORS.Indeterminado;
    const confidence = Number(pred.score || 0);
    const text = `${prettifyClassName(pred.source_class)} • ${category} • ${(confidence * 100).toFixed(0)}%`;

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(x, y, width, height);

    const textWidth = ctx.measureText(text).width;
    const boxHeight = fontSize + 12;
    const textY = Math.max(boxHeight, y);

    ctx.fillRect(x, textY - boxHeight, textWidth + 14, boxHeight);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, x + 7, textY - 7);
  });
}

function updateDetectionCard(predictions) {
  const valid = (predictions || [])
    .filter(p => Number(p.score) >= MIN_CONFIDENCE)
    .sort((a, b) => b.score - a.score);

  const best = valid[0];

  if (!best) {
    lastDetectionData = null;
    setDetectionCard({
      name: 'Nenhum objeto detectado',
      category: '-',
      bin: '-',
      dest: '-',
      time: '-',
      fact: 'Mire a câmera para um material reconhecido pelo modelo.',
      color: 'var(--primary-dark)'
    });
    return;
  }

  const rule = WASTE_RULES[best.category];

  lastDetectionData = {
    name: prettifyClassName(best.source_class),
    category: rule.category,
    bin: rule.bin,
    dest: rule.dest,
    time: rule.time,
    fact: rule.fact,
    confidence: best.score,
    sourceClass: best.source_class
  };

  setDetectionCard(lastDetectionData);
}

function setDetectionCard(data) {
  detName.textContent = data.name;
  detType.textContent = data.category;
  detBin.textContent = data.bin;
  detDest.textContent = data.dest;
  detTime.textContent = data.time;
  detFact.textContent = data.fact;
}

function setDetectionStatus(message) {
  if (detFact) detFact.textContent = message;
}

function saveCurrentDetection() {
  if (!lastDetectionData) return;

  if (!WASTE_RULES[normalizeKey(lastDetectionData.category)]) return;

  savedDetections.push({
    name: lastDetectionData.name,
    category: lastDetectionData.category,
    bin: lastDetectionData.bin,
    confidence: lastDetectionData.confidence,
    sourceClass: lastDetectionData.sourceClass,
    detectedAt: new Date().toISOString()
  });

  persistDetections();
  renderAll();

  saveBtn.textContent = '✓ Salvo!';
  saveBtn.disabled = true;

  setTimeout(() => {
    saveBtn.textContent = 'Salvar Detecção';
    saveBtn.disabled = false;
  }, 1300);
}

function normalizeKey(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/\s+/g, '_');
}

function prettifyClassName(label) {
  return String(label || '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase());
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`API ${response.status}`);
  return response.json();
}

function escapeHTML(str) {
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
