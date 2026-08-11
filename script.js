// ============================================================
// EcoScan AI
// Frontend + Camera + YOLO API
// ============================================================


// ============================================================
// ELEMENTOS
// ============================================================

const screens = Array.from(
  document.querySelectorAll('.screen')
);

const startScanBtn =
  document.getElementById('startScanBtn');

const saveBtn =
  document.getElementById('saveBtn');

const themeSwitch =
  document.getElementById('themeSwitch');

const notifSwitch =
  document.getElementById('notifSwitch');

const loginForm =
  document.getElementById('loginForm');

const goRegisterBtn =
  document.getElementById('goRegisterBtn');

const logoutBtn =
  document.getElementById('logoutBtn');

const cameraFeed =
  document.getElementById('cameraFeed');

const overlay =
  document.getElementById('overlay');

const detName =
  document.getElementById('detName');

const detType =
  document.getElementById('detType');

const detBin =
  document.getElementById('detBin');

const detDest =
  document.getElementById('detDest');

const detTime =
  document.getElementById('detTime');

const detFact =
  document.getElementById('detFact');


// ============================================================
// API
// ============================================================

const API_BASE = (
  window.ECOSCAN_API_BASE ||
  'http://127.0.0.1:8000'
).replace(/\/$/, '');


// ============================================================
// CONFIGURAÇÕES
// ============================================================

const DETECTION_INTERVAL_MS = 1200;

const MIN_CONFIDENCE = 0.35;

const MAX_BOXES = 5;

const MAX_FRAME_WIDTH = 512;

const API_TIMEOUT_MS = 12000;


// ============================================================
// ESTADO
// ============================================================

let currentScreen = 'loginScreen';

let stream = null;

let detectionLoopActive = false;

let frameCanvas = null;

let frameContext = null;

let lastPredictions = [];

let lastDetectionData = null;

let apiOnline = false;

let savedDetections =
  loadSavedDetections();


// ============================================================
// REGRAS
// ============================================================

const WASTE_RULES = {

  papel: {

    category: 'Papel',

    bin: '🔵 Azul',

    dest: 'Reciclagem',

    time: '3–6 meses',

    fact:
      'Papel e papelão devem estar, de preferência, secos e sem restos de comida.'
  },


  plastico: {

    category: 'Plástico',

    bin: '🔴 Vermelha',

    dest: 'Reciclagem',

    time: 'Varia por material',

    fact:
      'Garrafas PET, embalagens e outros plásticos devem ir para a coleta seletiva.'
  },


  vidro: {

    category: 'Vidro',

    bin: '🟢 Verde',

    dest: 'Reciclagem',

    time: 'Muito longo',

    fact:
      'Vidro deve ser encaminhado para a coleta seletiva.'
  },


  metal: {

    category: 'Metal',

    bin: '🟡 Amarela',

    dest: 'Reciclagem',

    time: 'Varia por material',

    fact:
      'Latas, tampas e outros metais devem ser encaminhados para reciclagem.'
  },


  organico: {

    category: 'Orgânico',

    bin: '🟤 Marrom',

    dest: 'Compostagem',

    time: 'Varia',

    fact:
      'Restos de alimentos e cascas podem ser destinados à compostagem.'
  },


  rejeito: {

    category: 'Rejeito',

    bin: '⚫ Cinza/Preta',

    dest: 'Rejeitos',

    time: 'Varia',

    fact:
      'Resíduos que não podem ser reciclados devem ser destinados aos rejeitos.'
  }

};


// ============================================================
// CORES
// ============================================================

const CATEGORY_COLORS = {

  Papel: '#2f6ef3',

  Plástico: '#df4b42',

  Vidro: '#43a047',

  Metal: '#d6a800',

  Orgânico: '#8b5a2b',

  Rejeito: '#5b6068',

  Indeterminado: '#43a047'

};


// ============================================================
// INIT
// ============================================================

init();


function init() {

  applyTheme(
    getTheme()
  );


  document.addEventListener(
    'click',
    handleGlobalClicks
  );


  loginForm?.addEventListener(
    'submit',
    event => {

      event.preventDefault();

      navigateTo(
        'homeScreen'
      );

    }
  );


  goRegisterBtn?.addEventListener(
    'click',
    () => navigateTo('homeScreen')
  );


  startScanBtn?.addEventListener(
    'click',
    () => navigateTo('cameraScreen')
  );


  saveBtn?.addEventListener(
    'click',
    saveCurrentDetection
  );


  logoutBtn?.addEventListener(
    'click',
    () => navigateTo('loginScreen')
  );


  themeSwitch?.addEventListener(
    'click',
    toggleTheme
  );


  notifSwitch?.addEventListener(
    'click',
    () =>
      notifSwitch.classList.toggle('on')
  );


  window.addEventListener(
    'resize',
    resizeOverlay
  );


  document.addEventListener(
    'visibilitychange',
    () => {

      if (document.hidden) {

        stopCamera();

      }

    }
  );


  window.addEventListener(
    'beforeunload',
    stopCamera
  );


  if (window.lucide) {

    lucide.createIcons();

  }


  renderAll();

}


// ============================================================
// NAVEGAÇÃO
// ============================================================

function handleGlobalClicks(event) {

  const go =
    event.target.closest('[data-go]');

  if (!go) return;

  const target =
    go.getAttribute('data-go');

  if (target) {

    navigateTo(target);

  }

}


function navigateTo(screenId) {

  // IMPORTANTE:
  //
  // Primeiro mudamos currentScreen.
  //
  // Depois abrimos a câmera.
  //
  // Antes estava invertido e o loop de
  // detecção podia terminar imediatamente.

  const previousScreen =
    currentScreen;


  if (
    previousScreen === 'cameraScreen' &&
    screenId !== 'cameraScreen'
  ) {

    stopCamera();

  }


  currentScreen =
    screenId;


  screens.forEach(
    screen =>
      screen.classList.remove('active')
  );


  const target =
    document.getElementById(screenId);


  if (target) {

    target.classList.add('active');

  }


  updateNavState(
    screenId
  );


  if (
    screenId === 'cameraScreen'
  ) {

    openCamera();

  }


  if (
    screenId === 'historyScreen'
  ) {

    renderHistory();

  }


  if (
    screenId === 'statsScreen'
  ) {

    renderStats();

  }


  if (
    screenId === 'achievementsScreen'
  ) {

    renderAchievements();

  }


  if (window.lucide) {

    lucide.createIcons();

  }

}


function updateNavState(
  screenId
) {

  document
    .querySelectorAll('.nav-item')
    .forEach(button => {

      button.classList.toggle(
        'active',
        button.getAttribute(
          'data-go'
        ) === screenId
      );

    });

}


// ============================================================
// TEMA
// ============================================================

function getTheme() {

  return (
    localStorage.getItem(
      'ecoscan-theme'
    ) || 'light'
  );

}


function applyTheme(theme) {

  document.body.dataset.theme =
    theme;

  localStorage.setItem(
    'ecoscan-theme',
    theme
  );

  setThemeSwitchUI();

}


function toggleTheme() {

  applyTheme(
    getTheme() === 'light'
      ? 'dark'
      : 'light'
  );

}


function setThemeSwitchUI() {

  themeSwitch?.classList.toggle(
    'on',
    getTheme() === 'dark'
  );

}


// ============================================================
// STORAGE
// ============================================================

function loadSavedDetections() {

  try {

    return JSON.parse(
      localStorage.getItem(
        'ecoscan-detections'
      ) || '[]'
    );

  } catch {

    return [];

  }

}


function persistDetections() {

  localStorage.setItem(
    'ecoscan-detections',
    JSON.stringify(
      savedDetections
    )
  );

}


// ============================================================
// RENDER
// ============================================================

function renderAll() {

  renderHomeStats();

  renderHistory();

  renderStats();

  renderAchievements();

}


function renderHomeStats() {

  const total =
    document.getElementById(
      'totalCount'
    );

  const recycle =
    document.getElementById(
      'recycleCount'
    );

  const organic =
    document.getElementById(
      'organicCount'
    );


  if (total) {

    total.textContent =
      savedDetections.length;

  }


  if (recycle) {

    recycle.textContent =
      savedDetections.filter(
        item =>
          [
            'Papel',
            'Plástico',
            'Vidro',
            'Metal'
          ].includes(
            item.category
          )
      ).length;

  }


  if (organic) {

    organic.textContent =
      savedDetections.filter(
        item =>
          item.category ===
          'Orgânico'
      ).length;

  }

}


// ============================================================
// HISTÓRICO
// ============================================================

function renderHistory() {

  const list =
    document.getElementById(
      'historyList'
    );


  if (!list) return;


  if (
    !savedDetections.length
  ) {

    list.innerHTML =
      '<p class="empty">Nenhuma detecção salva ainda.</p>';

    return;

  }


  list.innerHTML =
    savedDetections
      .slice()
      .reverse()
      .map(item => {

        const date =
          new Date(
            item.detectedAt
          ).toLocaleDateString(
            'pt-BR'
          );


        const confidence =
          Number.isFinite(
            item.confidence
          )
            ? ` • ${(item.confidence * 100).toFixed(0)}%`
            : '';


        return `
          <article class="info-card">

            <div
              style="
                display:flex;
                justify-content:space-between;
                gap:16px;
                align-items:flex-start
              "
            >

              <div>

                <h3
                  style="
                    margin-bottom:6px
                  "
                >
                  ${escapeHTML(
                    item.name
                  )}
                </h3>

                <p
                  style="
                    margin:0;
                    color:var(--muted)
                  "
                >
                  ${escapeHTML(
                    item.category
                  )}
                  •
                  ${escapeHTML(
                    item.bin
                  )}
                  ${confidence}
                </p>

              </div>

              <small
                style="
                  color:var(--muted)
                "
              >
                ${date}
              </small>

            </div>

          </article>
        `;

      })
      .join('');

}


// ============================================================
// ESTATÍSTICAS
// ============================================================

function renderStats() {

  const counts = {

    'Orgânico': 0,

    'Papel': 0,

    'Plástico': 0,

    'Vidro': 0,

    'Metal': 0,

    'Rejeito': 0

  };


  savedDetections.forEach(
    item => {

      if (
        counts[item.category]
        !== undefined
      ) {

        counts[item.category]++;

      }

    }
  );


  const max =
    Math.max(
      1,
      ...Object.values(counts)
    );


  const setBar =
    (id, count) => {

      const element =
        document.getElementById(
          id
        );


      if (element) {

        element.style.width =
          `${(
            count / max
          ) * 100}%`;

      }

    };


  const ids = {

    'Orgânico':
      'sbOrganic',

    'Papel':
      'sbPaper',

    'Plástico':
      'sbPlastic',

    'Vidro':
      'sbGlass',

    'Metal':
      'sbMetal',

    'Rejeito':
      'sbReject'

  };


  Object.entries(ids)
    .forEach(
      ([category, id]) => {

        const element =
          document.getElementById(
            id
          );


        if (element) {

          element.textContent =
            counts[category];

        }

      }
    );


  setBar(
    'barOrganic',
    counts['Orgânico']
  );


  setBar(
    'barPaper',
    counts['Papel']
  );


  setBar(
    'barPlastic',
    counts['Plástico']
  );


  setBar(
    'barGlass',
    counts['Vidro']
  );


  setBar(
    'barMetal',
    counts['Metal']
  );


  setBar(
    'barReject',
    counts['Rejeito']
  );


  const total =
    savedDetections.length;


  let level =
    '🌱 Iniciante Verde';


  if (total >= 50) {

    level =
      '🏆 Mestre Sustentável';

  } else if (total >= 25) {

    level =
      '🌎 Guardião Ambiental';

  } else if (total >= 10) {

    level =
      '♻️ Reciclador';

  }


  const levelElement =
    document.getElementById(
      'ecoLevel'
    );


  const pointsElement =
    document.getElementById(
      'ecoPoints'
    );


  if (levelElement) {

    levelElement.textContent =
      level;

  }


  if (pointsElement) {

    pointsElement.textContent =
      `${total * 10} pontos`;

  }

}


// ============================================================
// CONQUISTAS
// ============================================================

function renderAchievements() {

  const total =
    savedDetections.length;


  document
    .getElementById('ach1')
    ?.classList.toggle(
      'locked',
      total < 1
    );


  document
    .getElementById('ach2')
    ?.classList.toggle(
      'locked',
      total < 10
    );


  document
    .getElementById('ach3')
    ?.classList.toggle(
      'locked',
      total < 25
    );


  document
    .getElementById('ach4')
    ?.classList.toggle(
      'locked',
      total < 50
    );

}


// ============================================================
// CÂMERA
// ============================================================

async function openCamera() {

  if (stream) return;


  try {

    if (
      !cameraFeed ||
      !overlay
    ) {

      throw new Error(
        'Elementos da câmera não encontrados.'
      );

    }


    if (
      !navigator
        .mediaDevices
        ?.getUserMedia
    ) {

      throw new Error(
        'Seu navegador não suporta acesso à câmera.'
      );

    }


    setDetectionStatus(
      '📷 Abrindo câmera...'
    );


    // --------------------------------------------------------
    // CÂMERA
    // --------------------------------------------------------

    stream =
      await navigator
        .mediaDevices
        .getUserMedia({

          video: {

            facingMode: {
              ideal: 'environment'
            },

            width: {
              ideal: 1280
            },

            height: {
              ideal: 720
            }

          },

          audio: false

        });


    cameraFeed.srcObject =
      stream;


    cameraFeed.muted =
      true;


    cameraFeed.playsInline =
      true;


    await waitForVideoMetadata(
      cameraFeed,
      10000
    );


    await cameraFeed.play();


    resizeOverlay();


    setDetectionStatus(
      '📷 Câmera ativa • 🤖 conectando YOLO...'
    );


    // --------------------------------------------------------
    // TENTAR ZOOM
    // --------------------------------------------------------

    try {

      const track =
        stream
          .getVideoTracks()[0];


      const capabilities =
        track?.getCapabilities?.();


      if (
        capabilities?.zoom
      ) {

        const zoom =
          Math.min(
            1.5,
            capabilities.zoom.max
          );


        if (
          zoom >=
          capabilities.zoom.min
        ) {

          await track.applyConstraints({

            advanced: [
              {
                zoom
              }
            ]

          });

        }

      }

    } catch (error) {

      console.debug(
        'Zoom não suportado:',
        error
      );

    }


    // --------------------------------------------------------
    // NÃO ESPERAR O HEALTH PARA MOSTRAR A CÂMERA
    // --------------------------------------------------------
    //
    // Isso é importante no Render Free.
    //
    // O serviço pode estar dormindo.
    //
    // A câmera deve aparecer imediatamente.
    //
    // O primeiro /predict irá acordar o backend.

    startDetectionLoop();


  } catch (error) {

    console.error(
      'Erro ao abrir câmera:',
      error
    );


    stopCamera();


    setDetectionStatus(
      '❌ Não foi possível abrir a câmera.'
    );


    alert(
      formatCameraError(
        error
      )
    );

  }

}


// ============================================================
// PARAR CÂMERA
// ============================================================

function stopCamera() {

  detectionLoopActive =
    false;


  lastPredictions =
    [];


  lastDetectionData =
    null;


  apiOnline =
    false;


  if (stream) {

    stream
      .getTracks()
      .forEach(
        track =>
          track.stop()
      );


    stream = null;

  }


  if (cameraFeed) {

    cameraFeed.pause();

    cameraFeed.srcObject =
      null;

  }


  if (overlay) {

    const ctx =
      overlay.getContext(
        '2d'
      );


    ctx.clearRect(
      0,
      0,
      overlay.width,
      overlay.height
    );

  }

}


// ============================================================
// ERROS DA CÂMERA
// ============================================================

function formatCameraError(
  error
) {

  if (!error?.name) {

    return (
      error?.message ||
      'Erro ao acessar a câmera.'
    );

  }


  if (
    error.name ===
      'NotAllowedError' ||
    error.name ===
      'PermissionDeniedError'
  ) {

    return (
      'Permita o acesso à câmera '
      + 'nas configurações do navegador.'
    );

  }


  if (
    error.name ===
      'NotFoundError' ||
    error.name ===
      'DevicesNotFoundError'
  ) {

    return (
      'Nenhuma câmera foi encontrada.'
    );

  }


  if (
    error.name ===
    'NotReadableError'
  ) {

    return (
      'A câmera está sendo usada '
      + 'por outro aplicativo.'
    );

  }


  if (
    error.name ===
    'SecurityError'
  ) {

    return (
      'A câmera exige HTTPS.'
    );

  }


  return (
    error.message ||
    'Não foi possível abrir a câmera.'
  );

}


// ============================================================
// VIDEO METADATA
// ============================================================

function waitForVideoMetadata(
  video,
  timeoutMs
) {

  return new Promise(
    (resolve, reject) => {

      if (
        video.videoWidth &&
        video.videoHeight
      ) {

        resolve();

        return;

      }


      const timer =
        setTimeout(
          () => {

            video.removeEventListener(
              'loadedmetadata',
              onLoaded
            );


            reject(
              new Error(
                'Tempo excedido ao carregar a câmera.'
              )
            );

          },
          timeoutMs
        );


      function onLoaded() {

        clearTimeout(
          timer
        );


        video.removeEventListener(
          'loadedmetadata',
          onLoaded
        );


        resolve();

      }


      video.addEventListener(
        'loadedmetadata',
        onLoaded
      );

    }
  );

}


// ============================================================
// OVERLAY
// ============================================================

function resizeOverlay() {

  if (
    !cameraFeed?.videoWidth ||
    !cameraFeed?.videoHeight ||
    !overlay
  ) {

    return;

  }


  overlay.width =
    cameraFeed.videoWidth;


  overlay.height =
    cameraFeed.videoHeight;

}


// ============================================================
// LOOP YOLO
// ============================================================

async function startDetectionLoop() {

  if (
    detectionLoopActive
  ) {

    return;

  }


  detectionLoopActive =
    true;


  frameCanvas =
    document.createElement(
      'canvas'
    );


  frameContext =
    frameCanvas.getContext(
      '2d'
    );


  while (
    detectionLoopActive &&
    currentScreen ===
      'cameraScreen'
  ) {

    try {

      if (
        cameraFeed.readyState >=
        2
      ) {

        const result =
          await detectFrame();


        lastPredictions =
          result?.predictions ||
          [];


        apiOnline =
          true;


        setDetectionStatus(
          '📷 Câmera ativa • 🤖 YOLO conectado'
        );


        drawPredictions(
          lastPredictions
        );


        updateDetectionCard(
          lastPredictions
        );

      }

    } catch (error) {

      console.warn(
        'YOLO:',
        error
      );


      apiOnline =
        false;


      setDetectionStatus(
        '📷 Câmera ativa • ⏳ aguardando YOLO...'
      );


      updateDetectionCard(
        []
      );


      // Render Free pode demorar
      // para acordar.

      await sleep(
        2500
      );

    }


    await sleep(
      DETECTION_INTERVAL_MS
    );

  }

}


// ============================================================
// DETECT FRAME
// ============================================================

async function detectFrame() {

  if (
    !cameraFeed.videoWidth ||
    !cameraFeed.videoHeight
  ) {

    throw new Error(
      'Vídeo ainda não possui dimensões.'
    );

  }


  const scale =
    Math.min(
      1,
      MAX_FRAME_WIDTH /
        cameraFeed.videoWidth
    );


  frameCanvas.width =
    Math.max(
      1,
      Math.round(
        cameraFeed.videoWidth *
          scale
      )
    );


  frameCanvas.height =
    Math.max(
      1,
      Math.round(
        cameraFeed.videoHeight *
          scale
      )
    );


  frameContext.drawImage(

    cameraFeed,

    0,
    0,

    frameCanvas.width,
    frameCanvas.height

  );


  const blob =
    await canvasToBlob(
      frameCanvas,
      0.70
    );


  const formData =
    new FormData();


  formData.append(
    'file',
    blob,
    'frame.jpg'
  );


  const response =
    await fetchWithTimeout(

      `${API_BASE}/predict`,

      {

        method: 'POST',

        body: formData,

        cache: 'no-store'

      },

      API_TIMEOUT_MS

    );


  if (!response.ok) {

    const message =
      await response.text();


    throw new Error(
      `API ${response.status}: ${message}`
    );

  }


  const result =
    await response.json();


  // ----------------------------------------------------------
  // CONVERTER BBOX
  // ----------------------------------------------------------

  const sx =
    cameraFeed.videoWidth /
    frameCanvas.width;


  const sy =
    cameraFeed.videoHeight /
    frameCanvas.height;


  result.predictions =
    (result.predictions || [])
      .map(
        prediction => {

          if (
            !prediction.bbox ||
            prediction.bbox.length <
              4
          ) {

            return null;

          }


          return {

            ...prediction,

            bbox: [

              prediction.bbox[0] *
                sx,

              prediction.bbox[1] *
                sy,

              prediction.bbox[2] *
                sx,

              prediction.bbox[3] *
                sy

            ]

          };

        }
      )
      .filter(Boolean);


  return result;

}


// ============================================================
// FETCH COM TIMEOUT
// ============================================================

async function fetchWithTimeout(
  url,
  options = {},
  timeout = 10000
) {

  const controller =
    new AbortController();


  const timer =
    setTimeout(
      () =>
        controller.abort(),
      timeout
    );


  try {

    return await fetch(
      url,
      {
        ...options,
        signal:
          controller.signal
      }
    );

  } finally {

    clearTimeout(
      timer
    );

  }

}


// ============================================================
// CANVAS BLOB
// ============================================================

function canvasToBlob(
  canvas,
  quality
) {

  return new Promise(
    (resolve, reject) => {

      canvas.toBlob(

        blob => {

          if (blob) {

            resolve(blob);

          } else {

            reject(
              new Error(
                'Não foi possível criar o frame.'
              )
            );

          }

        },

        'image/jpeg',

        quality

      );

    }
  );

}


// ============================================================
// DESENHAR DETECÇÕES
// ============================================================

function drawPredictions(
  predictions
) {

  resizeOverlay();


  if (!overlay) return;


  const ctx =
    overlay.getContext(
      '2d'
    );


  ctx.clearRect(
    0,
    0,
    overlay.width,
    overlay.height
  );


  if (
    !predictions?.length
  ) {

    return;

  }


  const fontSize =
    Math.max(
      15,
      Math.round(
        overlay.width / 42
      )
    );


  ctx.font =
    `700 ${fontSize}px Outfit`;


  ctx.lineWidth =
    Math.max(
      3,
      Math.round(
        overlay.width / 500
      )
    );


  predictions
    .slice(
      0,
      MAX_BOXES
    )
    .forEach(
      prediction => {

        const [
          x,
          y,
          width,
          height
        ] =
          prediction.bbox;


        const category =
          prediction.category ||
          'Indeterminado';


        const color =
          CATEGORY_COLORS[
            category
          ] ||
          CATEGORY_COLORS
            .Indeterminado;


        const confidence =
          Number(
            prediction.score ||
            0
          );


        const text =
          `${prettifyClassName(
            prediction.source_class
          )} • ${category} • ${(confidence * 100).toFixed(0)}%`;


        ctx.strokeStyle =
          color;


        ctx.fillStyle =
          color;


        ctx.strokeRect(
          x,
          y,
          width,
          height
        );


        const textWidth =
          ctx.measureText(
            text
          ).width;


        const boxHeight =
          fontSize + 12;


        const textY =
          Math.max(
            boxHeight,
            y
          );


        ctx.fillRect(

          x,

          textY -
            boxHeight,

          textWidth + 14,

          boxHeight

        );


        ctx.fillStyle =
          '#fff';


        ctx.fillText(

          text,

          x + 7,

          textY - 7

        );

      }
    );

}


// ============================================================
// CARD DE DETECÇÃO
// ============================================================

function updateDetectionCard(
  predictions
) {

  const valid =
    (predictions || [])
      .filter(
        prediction =>
          Number(
            prediction.score
          ) >=
          MIN_CONFIDENCE
      )
      .sort(
        (a, b) =>
          b.score - a.score
      );


  const best =
    valid[0];


  if (!best) {

    lastDetectionData =
      null;


    setDetectionCard({

      name:
        'Nenhum objeto detectado',

      category:
        '-',

      bin:
        '-',

      dest:
        '-',

      time:
        '-',

      fact:
        apiOnline
          ? 'Mire a câmera para um material reconhecido pelo modelo.'
          : 'Aguardando conexão com o detector YOLO...'

    });


    return;

  }


  const rule =
    WASTE_RULES[
      best.category
    ];


  if (!rule) {

    return;

  }


  lastDetectionData = {

    name:
      prettifyClassName(
        best.source_class
      ),

    category:
      rule.category,

    bin:
      rule.bin,

    dest:
      rule.dest,

    time:
      rule.time,

    fact:
      rule.fact,

    confidence:
      Number(
        best.score
      ),

    sourceClass:
      best.source_class

  };


  setDetectionCard(
    lastDetectionData
  );

}


function setDetectionCard(
  data
) {

  if (detName) {

    detName.textContent =
      data.name;

  }


  if (detType) {

    detType.textContent =
      data.category;

  }


  if (detBin) {

    detBin.textContent =
      data.bin;

  }


  if (detDest) {

    detDest.textContent =
      data.dest;

  }


  if (detTime) {

    detTime.textContent =
      data.time;

  }


  if (detFact) {

    detFact.textContent =
      data.fact;

  }

}


function setDetectionStatus(
  message
) {

  if (detFact) {

    detFact.textContent =
      message;

  }

}


// ============================================================
// SALVAR
// ============================================================

function saveCurrentDetection() {

  if (
    !lastDetectionData
  ) {

    return;

  }


  const key =
    normalizeKey(
      lastDetectionData.category
    );


  if (
    !WASTE_RULES[key]
  ) {

    return;

  }


  savedDetections.push({

    name:
      lastDetectionData.name,

    category:
      lastDetectionData.category,

    bin:
      lastDetectionData.bin,

    confidence:
      lastDetectionData.confidence,

    sourceClass:
      lastDetectionData.sourceClass,

    detectedAt:
      new Date().toISOString()

  });


  persistDetections();

  renderAll();


  if (saveBtn) {

    saveBtn.textContent =
      '✓ Salvo!';

    saveBtn.disabled =
      true;


    setTimeout(
      () => {

        saveBtn.textContent =
          'Salvar Detecção';

        saveBtn.disabled =
          false;

      },
      1300
    );

  }

}


// ============================================================
// HELPERS
// ============================================================

function normalizeKey(
  value
) {

  return String(
    value || ''
  )
    .normalize('NFD')
    .replace(
      /[\u0300-\u036f]/g,
      ''
    )
    .toLowerCase()
    .replace(
      /\s+/g,
      '_'
    );

}


function prettifyClassName(
  label
) {

  return String(
    label || ''
  )
    .replace(
      /_/g,
      ' '
    )
    .replace(
      /\b\w/g,
      letter =>
        letter.toUpperCase()
    );

}


function escapeHTML(
  value
) {

  return String(
    value
  )
    .replaceAll(
      '&',
      '&amp;'
    )
    .replaceAll(
      '<',
      '&lt;'
    )
    .replaceAll(
      '>',
      '&gt;'
    )
    .replaceAll(
      '"',
      '&quot;'
    )
    .replaceAll(
      "'",
      '&#039;'
    );

}


function sleep(
  ms
) {

  return new Promise(
    resolve =>
      setTimeout(
        resolve,
        ms
      )
  );

}