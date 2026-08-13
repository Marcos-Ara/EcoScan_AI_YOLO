// ============================================================
// EcoScan AI
// Frontend + Camera + YOLO API + Firebase
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

const selectImageBtn =
  document.getElementById('selectImageBtn');

const imageInput =
  document.getElementById('imageInput');

const themeSwitch =
  document.getElementById('themeSwitch');

const notifSwitch =
  document.getElementById('notifSwitch');

const loginForm =
  document.getElementById('loginForm');

const goRegisterBtn =
  document.getElementById('goRegisterBtn');

const forgotPasswordBtn =
  document.getElementById('forgotPasswordBtn');

const googleLoginBtn =
  document.getElementById('googleLoginBtn');

const registerForm =
  document.getElementById('registerForm');

const googleRegisterBtn =
  document.getElementById('googleRegisterBtn');

const backToLoginBtn =
  document.getElementById('backToLoginBtn');

const checkVerificationBtn =
  document.getElementById('checkVerificationBtn');

const resendVerificationBtn =
  document.getElementById('resendVerificationBtn');

const verifyLogoutBtn =
  document.getElementById('verifyLogoutBtn');

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

const DETECTION_INTERVAL_MS = 2500;

const MIN_CONFIDENCE = 0.35;

const MAX_BOXES = 5;

const MAX_FRAME_WIDTH = 320;

const API_TIMEOUT_MS = 30000;


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

let staticImageMode = false;

let savedDetections = loadSavedDetections();

let auth = null;

let currentUser = null;

let authReady = false;


const firebaseConfig =
  window.ECOSCAN_FIREBASE_CONFIG || {};


initializeFirebase();


// ============================================================
// REGRAS DE RECICLAGEM
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
// ALIASES DO YOLO
// ============================================================
//
// Alguns modelos podem retornar:
//
// cardboard
// paper
// paperboard
// plastic
// bottle
// glass
// metal
//
// enquanto o EcoScan usa:
//
// papel
// plastico
// vidro
// metal
//
// Aqui fazemos a conversão.
//

const WASTE_ALIASES = {

  // PAPEL
  cardboard: 'papel',
  card_board: 'papel',
  paper: 'papel',
  paperboard: 'papel',
  paper_board: 'papel',
  papel: 'papel',
  papelao: 'papel',
  papelao: 'papel',

  // PLÁSTICO
  plastic: 'plastico',
  plastics: 'plastico',
  plastico: 'plastico',
  garrafa_plastica: 'plastico',
  plastic_bottle: 'plastico',
  bottle: 'plastico',
  pet: 'plastico',

  // VIDRO
  glass: 'vidro',
  vidro: 'vidro',
  glass_bottle: 'vidro',

  // METAL
  metal: 'metal',
  metals: 'metal',
  lata: 'metal',
  can: 'metal',
  aluminum: 'metal',
  aluminium: 'metal',

  // ORGÂNICO
  organic: 'organico',
  organico: 'organico',
  food: 'organico',
  food_waste: 'organico',
  resto_de_comida: 'organico',

  // REJEITO
  reject: 'rejeito',
  rejeito: 'rejeito',
  trash: 'rejeito',
  garbage: 'rejeito',
  waste: 'rejeito'

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
    handleLogin
  );


  registerForm?.addEventListener(
    'submit',
    handleRegister
  );


  goRegisterBtn?.addEventListener(
    'click',
    () => {

      clearAuthMessages();

      navigateTo(
        'registerScreen'
      );

    }
  );


  backToLoginBtn?.addEventListener(
    'click',
    () => {

      clearAuthMessages();

      navigateTo(
        'loginScreen'
      );

    }
  );


  forgotPasswordBtn?.addEventListener(
    'click',
    handleForgotPassword
  );


  googleLoginBtn?.addEventListener(
    'click',
    handleGoogleSignIn
  );


  googleRegisterBtn?.addEventListener(
    'click',
    handleGoogleSignIn
  );


  checkVerificationBtn?.addEventListener(
    'click',
    checkEmailVerification
  );


  resendVerificationBtn?.addEventListener(
    'click',
    resendVerificationEmail
  );


  verifyLogoutBtn?.addEventListener(
    'click',
    handleLogout
  );


  startScanBtn?.addEventListener(
    'click',
    () =>
      navigateTo(
        'cameraScreen'
      )
  );


  saveBtn?.addEventListener(
    'click',
    saveCurrentDetection
  );


  selectImageBtn?.addEventListener(
    'click',
    () =>
      imageInput?.click()
  );


  imageInput?.addEventListener(
    'change',
    handleImageSelection
  );


  logoutBtn?.addEventListener(
    'click',
    handleLogout
  );


  themeSwitch?.addEventListener(
    'click',
    toggleTheme
  );


  notifSwitch?.addEventListener(
    'click',
    () =>
      notifSwitch.classList.toggle(
        'on'
      )
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
// FIREBASE AUTHENTICATION
// ============================================================

function initializeFirebase() {

  try {

    if (
      !window.firebase ||
      !window.firebase.auth
    ) {

      console.error(
        'Firebase Auth SDK não foi carregado.'
      );

      setAuthMessage(
        'loginMessage',
        'O Firebase não foi carregado. Verifique sua internet e o index.html.',
        'error'
      );

      return;

    }


    if (
      !firebaseConfig.apiKey ||
      firebaseConfig.apiKey.includes(
        'COLE_'
      )
    ) {

      console.error(
        'Firebase config incompleto:',
        firebaseConfig
      );

      setAuthMessage(
        'loginMessage',
        'A configuração do Firebase está incompleta no config.js.',
        'error'
      );

      return;

    }


    if (!firebase.apps.length) {

      firebase.initializeApp(
        firebaseConfig
      );

    }


    auth =
      firebase.auth();


    auth
      .setPersistence(
        firebase.auth.Auth.Persistence.LOCAL
      )
      .catch(
        error =>
          console.error(
            'Não foi possível definir persistência da sessão:',
            error
          )
      );


    getRedirectResultSafe();


    auth.onAuthStateChanged(
      async user => {

        currentUser =
          user;

        authReady =
          true;


        if (!user) {

          navigateTo(
            'loginScreen'
          );

          updateUserUI(
            null
          );

          return;

        }


        updateUserUI(
          user
        );


        if (
          !user.emailVerified &&
          user.providerData.some(
            p =>
              p.providerId ===
              'password'
          )
        ) {

          navigateTo(
            'verifyScreen'
          );

          setAuthMessage(
            'verifyMessage',
            `Confirme o e-mail ${user.email} para continuar.`,
            'info'
          );

          return;

        }


        navigateTo(
          'homeScreen'
        );

      }
    );

  } catch (error) {

    console.error(
      'Falha ao inicializar Firebase:',
      error
    );

    setAuthMessage(
      'loginMessage',
      'Não foi possível inicializar a autenticação. Confira o config.js.',
      'error'
    );

  }

}


async function getRedirectResultSafe() {

  if (!auth) return;


  try {

    const result =
      await auth.getRedirectResult();


    if (
      result &&
      result.user
    ) {

      updateUserUI(
        result.user
      );

    }

  } catch (error) {

    console.error(
      'Resultado do login Google por redirecionamento:',
      error
    );


    const message =
      firebaseAuthError(
        error
      );


    if (message) {

      setAuthMessage(
        'loginMessage',
        message,
        'error'
      );

    }

  }

}


async function handleLogin(
  event
) {

  event.preventDefault();


  if (!auth) {

    setAuthMessage(
      'loginMessage',
      'Configure o Firebase no arquivo config.js antes de entrar.',
      'error'
    );

    return;

  }


  const email =
    document
      .getElementById(
        'loginEmail'
      )
      ?.value
      .trim();


  const password =
    document
      .getElementById(
        'loginPassword'
      )
      ?.value;


  if (
    !email ||
    !password
  ) {

    setAuthMessage(
      'loginMessage',
      'Informe seu e-mail e sua senha.',
      'error'
    );

    return;

  }


  setButtonLoading(
    'loginBtn',
    true,
    'Entrando...'
  );


  clearAuthMessage(
    'loginMessage'
  );


  try {

    await auth.signInWithEmailAndPassword(
      email,
      password
    );

  } catch (error) {

    setAuthMessage(
      'loginMessage',
      firebaseAuthError(
        error
      ),
      'error'
    );

  } finally {

    setButtonLoading(
      'loginBtn',
      false,
      'Entrar'
    );

  }

}


async function handleRegister(
  event
) {

  event.preventDefault();


  if (!auth) {

    setAuthMessage(
      'registerMessage',
      'Configure o Firebase no arquivo config.js antes de criar a conta.',
      'error'
    );

    return;

  }


  const name =
    document
      .getElementById(
        'registerName'
      )
      ?.value
      .trim();


  const email =
    document
      .getElementById(
        'registerEmail'
      )
      ?.value
      .trim();


  const password =
    document
      .getElementById(
        'registerPassword'
      )
      ?.value;


  const confirm =
    document
      .getElementById(
        'registerPasswordConfirm'
      )
      ?.value;


  if (
    !name ||
    !email ||
    !password ||
    !confirm
  ) {

    setAuthMessage(
      'registerMessage',
      'Preencha todos os campos.',
      'error'
    );

    return;

  }


  if (
    password.length < 6
  ) {

    setAuthMessage(
      'registerMessage',
      'A senha precisa ter pelo menos 6 caracteres.',
      'error'
    );

    return;

  }


  if (
    password !== confirm
  ) {

    setAuthMessage(
      'registerMessage',
      'As senhas não coincidem.',
      'error'
    );

    return;

  }


  setButtonLoading(
    'registerBtn',
    true,
    'Criando...'
  );


  clearAuthMessage(
    'registerMessage'
  );


  try {

    const credential =
      await auth.createUserWithEmailAndPassword(
        email,
        password
      );


    if (credential.user) {

      await credential.user.updateProfile({
        displayName: name
      });


      await credential.user.sendEmailVerification();


      await credential.user.reload();

    }


    document
      .getElementById(
        'registerForm'
      )
      ?.reset();


    setAuthMessage(
      'verifyMessage',
      `Enviamos uma mensagem de confirmação para ${email}.`,
      'success'
    );


    navigateTo(
      'verifyScreen'
    );

  } catch (error) {

    setAuthMessage(
      'registerMessage',
      firebaseAuthError(
        error
      ),
      'error'
    );

  } finally {

    setButtonLoading(
      'registerBtn',
      false,
      'Criar Conta'
    );

  }

}


async function handleGoogleSignIn() {

  if (!auth) {

    setAuthMessage(
      'loginMessage',
      'Configure o Firebase no arquivo config.js antes de usar o Google.',
      'error'
    );

    return;

  }


  const provider =
    new firebase.auth.GoogleAuthProvider();


  provider.setCustomParameters({
    prompt: 'select_account'
  });


  try {

    await auth.signInWithPopup(
      provider
    );

  } catch (error) {

    if (
      error?.code ===
      'auth/popup-closed-by-user'
    ) {

      return;

    }


    if (
      error?.code ===
        'auth/popup-blocked' ||
      error?.code ===
        'auth/cancelled-popup-request'
    ) {

      try {

        await auth.signInWithRedirect(
          provider
        );

        return;

      } catch (redirectError) {

        error =
          redirectError;

      }

    }


    const message =
      firebaseAuthError(
        error
      );


    setAuthMessage(
      'loginMessage',
      message,
      'error'
    );


    setAuthMessage(
      'registerMessage',
      message,
      'error'
    );

  }

}


async function handleForgotPassword() {

  if (!auth) {

    setAuthMessage(
      'loginMessage',
      'Configure o Firebase no arquivo config.js antes de recuperar a senha.',
      'error'
    );

    return;

  }


  const email =
    document
      .getElementById(
        'loginEmail'
      )
      ?.value
      .trim();


  if (!email) {

    setAuthMessage(
      'loginMessage',
      'Digite seu e-mail no campo acima para receber o link de recuperação.',
      'info'
    );


    document
      .getElementById(
        'loginEmail'
      )
      ?.focus();


    return;

  }


  try {

    await auth.sendPasswordResetEmail(
      email
    );


    setAuthMessage(
      'loginMessage',
      'Enviamos o link de recuperação para o seu e-mail.',
      'success'
    );

  } catch (error) {

    setAuthMessage(
      'loginMessage',
      firebaseAuthError(
        error
      ),
      'error'
    );

  }

}


async function resendVerificationEmail() {

  if (!auth?.currentUser) {

    navigateTo(
      'loginScreen'
    );

    return;

  }


  try {

    await auth.currentUser.sendEmailVerification();


    setAuthMessage(
      'verifyMessage',
      'Novo e-mail de confirmação enviado.',
      'success'
    );

  } catch (error) {

    setAuthMessage(
      'verifyMessage',
      firebaseAuthError(
        error
      ),
      'error'
    );

  }

}


async function checkEmailVerification() {

  if (!auth?.currentUser) {

    navigateTo(
      'loginScreen'
    );

    return;

  }


  try {

    await auth.currentUser.reload();


    if (
      auth.currentUser.emailVerified
    ) {

      setAuthMessage(
        'verifyMessage',
        'E-mail confirmado com sucesso.',
        'success'
      );


      navigateTo(
        'homeScreen'
      );

    } else {

      setAuthMessage(
        'verifyMessage',
        'Ainda não identificamos a confirmação. Abra o e-mail e clique no link antes de tentar novamente.',
        'info'
      );

    }

  } catch (error) {

    setAuthMessage(
      'verifyMessage',
      firebaseAuthError(
        error
      ),
      'error'
    );

  }

}


async function handleLogout() {

  try {

    stopCamera();


    if (auth) {

      await auth.signOut();

    } else {

      navigateTo(
        'loginScreen'
      );

    }

  } catch (error) {

    console.error(
      'Erro ao sair:',
      error
    );

  }

}


function updateUserUI(
  user
) {

  const name =
    user?.displayName ||
    user?.email?.split('@')[0] ||
    'usuário';


  const userName =
    document.getElementById(
      'userName'
    );


  const avatar =
    document.getElementById(
      'avatarBtn'
    );


  if (userName) {

    userName.textContent =
      name.split(' ')[0];

  }


  if (avatar) {

    avatar.textContent =
      name
        .charAt(0)
        .toUpperCase();

  }

}


function setAuthMessage(
  id,
  message,
  type = 'info'
) {

  const element =
    document.getElementById(
      id
    );


  if (!element) return;


  element.textContent =
    message || '';


  element.className =
    `auth-message ${type}`;

}


function clearAuthMessage(
  id
) {

  const element =
    document.getElementById(
      id
    );


  if (!element) return;


  element.textContent =
    '';


  element.className =
    'auth-message';

}


function clearAuthMessages() {

  [
    'loginMessage',
    'registerMessage',
    'verifyMessage'
  ]
    .forEach(
      clearAuthMessage
    );

}


function setButtonLoading(
  id,
  loading,
  label
) {

  const button =
    document.getElementById(
      id
    );


  if (!button) return;


  button.disabled =
    loading;


  button.classList.toggle(
    'is-loading',
    loading
  );


  button.textContent =
    label;

}


function firebaseAuthError(
  error
) {

  const code =
    error?.code || '';


  const messages = {

    'auth/invalid-email':
      'Digite um e-mail válido.',

    'auth/missing-password':
      'Digite sua senha.',

    'auth/weak-password':
      'A senha é muito fraca. Use pelo menos 6 caracteres.',

    'auth/email-already-in-use':
      'Este e-mail já possui uma conta.',

    'auth/invalid-credential':
      'E-mail ou senha incorretos.',

    'auth/user-not-found':
      'Não encontramos uma conta com esse e-mail.',

    'auth/wrong-password':
      'E-mail ou senha incorretos.',

    'auth/too-many-requests':
      'Muitas tentativas. Aguarde alguns minutos e tente novamente.',

    'auth/popup-blocked':
      'O navegador bloqueou a janela do Google. Permita pop-ups para este site.',

    'auth/operation-not-allowed':
      'Esse método de login ainda não foi ativado no Firebase.',

    'auth/account-exists-with-different-credential':
      'Esse e-mail já está cadastrado usando outro método de login.',

    'auth/network-request-failed':
      'Falha de conexão. Verifique sua internet e tente novamente.'

  };


  return (
    messages[code] ||
    'Não foi possível concluir a autenticação. Tente novamente.'
  );

}


// ============================================================
// NAVEGAÇÃO
// ============================================================

function handleGlobalClicks(
  event
) {

  const go =
    event.target.closest(
      '[data-go]'
    );


  if (!go) return;


  const target =
    go.getAttribute(
      'data-go'
    );


  if (target) {

    navigateTo(
      target
    );

  }

}


function navigateTo(
  screenId
) {

  const publicScreens =
    new Set([
      'loginScreen',
      'registerScreen',
      'verifyScreen'
    ]);


  if (
    authReady &&
    !currentUser &&
    !publicScreens.has(
      screenId
    )
  ) {

    screenId =
      'loginScreen';

  }


  const previousScreen =
    currentScreen;


  if (
    previousScreen ===
      'cameraScreen' &&
    screenId !==
      'cameraScreen'
  ) {

    stopCamera();

  }


  currentScreen =
    screenId;


  screens.forEach(
    screen =>
      screen.classList.remove(
        'active'
      )
  );


  const target =
    document.getElementById(
      screenId
    );


  if (target) {

    target.classList.add(
      'active'
    );

  }


  updateNavState(
    screenId
  );


  if (
    screenId ===
    'cameraScreen'
  ) {

    openCamera();

  }


  if (
    screenId ===
    'historyScreen'
  ) {

    renderHistory();

  }


  if (
    screenId ===
    'statsScreen'
  ) {

    renderStats();

  }


  if (
    screenId ===
    'achievementsScreen'
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
    .querySelectorAll(
      '.nav-item'
    )
    .forEach(
      button => {

        button.classList.toggle(
          'active',
          button.getAttribute(
            'data-go'
          ) === screenId
        );

      }
    );

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


function applyTheme(
  theme
) {

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
      .map(
        item => {

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

        }
      )
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
        counts[item.category] !==
        undefined
      ) {

        counts[item.category]++;

      }

    }
  );


  const max =
    Math.max(
      1,
      ...Object.values(
        counts
      )
    );


  const setBar =
    (
      id,
      count
    ) => {

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


  Object.entries(
    ids
  )
    .forEach(
      (
        [category, id]
      ) => {

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


  if (
    total >= 50
  ) {

    level =
      '🏆 Mestre Sustentável';

  } else if (
    total >= 25
  ) {

    level =
      '🌎 Guardião Ambiental';

  } else if (
    total >= 10
  ) {

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
    .getElementById(
      'ach1'
    )
    ?.classList.toggle(
      'locked',
      total < 1
    );


  document
    .getElementById(
      'ach2'
    )
    ?.classList.toggle(
      'locked',
      total < 10
    );


  document
    .getElementById(
      'ach3'
    )
    ?.classList.toggle(
      'locked',
      total < 25
    );


  document
    .getElementById(
      'ach4'
    )
    ?.classList.toggle(
      'locked',
      total < 50
    );

}


// ============================================================
// CÂMERA
// ============================================================

async function openCamera() {

  staticImageMode =
    false;


  if (cameraFeed) {

    cameraFeed.style.opacity =
      '1';

  }


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
    // ZOOM
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


  staticImageMode =
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


    stream =
      null;

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
      'Permita o acesso à câmera nas configurações do navegador.'
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
      'A câmera está sendo usada por outro aplicativo.'
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
    (
      resolve,
      reject
    ) => {

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

  if (staticImageMode) {

    return;

  }


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
// TESTE MANUAL POR IMAGEM
// ============================================================

async function handleImageSelection(
  event
) {

  const file =
    event.target.files?.[0];


  event.target.value =
    '';


  if (!file) return;


  if (
    !file.type.startsWith(
      'image/'
    )
  ) {

    alert(
      'Selecione uma imagem válida.'
    );

    return;

  }


  try {

    stopCamera();


    staticImageMode =
      true;


    setDetectionStatus(
      '🖼️ Preparando imagem...'
    );


    const image =
      await loadImageFile(
        file
      );


    const maxDimension =
      960;


    const scale =
      Math.min(
        1,
        maxDimension /
          Math.max(
            image.naturalWidth,
            image.naturalHeight
          )
      );


    const width =
      Math.max(
        1,
        Math.round(
          image.naturalWidth *
          scale
        )
      );


    const height =
      Math.max(
        1,
        Math.round(
          image.naturalHeight *
          scale
        )
      );


    frameCanvas =
      document.createElement(
        'canvas'
      );


    frameCanvas.width =
      width;


    frameCanvas.height =
      height;


    frameContext =
      frameCanvas.getContext(
        '2d'
      );


    frameContext.drawImage(
      image,
      0,
      0,
      width,
      height
    );


    const blob =
      await canvasToBlob(
        frameCanvas,
        0.60
      );


    const formData =
      new FormData();


    formData.append(
      'file',
      blob,
      'image.jpg'
    );


    setDetectionStatus(
      '🤖 Analisando imagem...'
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


    apiOnline =
      true;


    overlay.width =
      width;


    overlay.height =
      height;


    const ctx =
      overlay.getContext(
        '2d'
      );


    ctx.clearRect(
      0,
      0,
      width,
      height
    );


    ctx.drawImage(
      image,
      0,
      0,
      width,
      height
    );


    lastPredictions =
      result.predictions || [];


    drawPredictions(
      lastPredictions,
      true
    );


    updateDetectionCard(
      lastPredictions
    );


    setDetectionStatus(
      lastPredictions.length
        ? '🖼️ Imagem analisada • 🤖 YOLO conectado'
        : '🖼️ Imagem analisada • nenhum objeto reconhecido'
    );

  } catch (error) {

    console.error(
      'Erro ao analisar imagem:',
      error
    );


    apiOnline =
      false;


    setDetectionStatus(
      '❌ Não foi possível analisar a imagem.'
    );


    updateDetectionCard(
      []
    );


    alert(
      `Não foi possível analisar a imagem.\n\n${
        error.message || error
      }`
    );

  }

}


function loadImageFile(
  file
) {

  return new Promise(
    (
      resolve,
      reject
    ) => {

      const url =
        URL.createObjectURL(
          file
        );


      const image =
        new Image();


      image.onload =
        () => {

          URL.revokeObjectURL(
            url
          );


          if (
            !image.naturalWidth ||
            !image.naturalHeight
          ) {

            reject(
              new Error(
                'A imagem não possui dimensões válidas.'
              )
            );

            return;

          }


          resolve(
            image
          );

        };


      image.onerror =
        () => {

          URL.revokeObjectURL(
            url
          );


          reject(
            new Error(
              'Não foi possível abrir a imagem.'
            )
          );

        };


      image.src =
        url;

    }
  );

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
    (
      result.predictions || []
    )
      .map(
        prediction => {

          if (
            !prediction.bbox ||
            prediction.bbox.length < 4
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
      .filter(
        Boolean
      );


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
    (
      resolve,
      reject
    ) => {

      canvas.toBlob(
        blob => {

          if (blob) {

            resolve(
              blob
            );

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
  predictions,
  preserveBackground = false
) {

  if (!overlay) return;


  if (!staticImageMode) {

    resizeOverlay();

  }


  const ctx =
    overlay.getContext(
      '2d'
    );


  if (!preserveBackground) {

    ctx.clearRect(
      0,
      0,
      overlay.width,
      overlay.height
    );

  }


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

        if (
          !prediction.bbox ||
          prediction.bbox.length < 4
        ) {

          return;

        }


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
            prediction.score || 0
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
          textY - boxHeight,
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
// RESOLVER REGRA DA DETECÇÃO
// ============================================================
//
// Essa é uma das partes mais importantes.
//
// O backend pode enviar:
//
// category_key: "cardboard"
// category: "Papel"
// source_class: "cardboard"
//
// O EcoScan precisa entender que isso é PAPEL.
//

function resolveWasteRule(
  prediction
) {

  if (!prediction) {

    return null;

  }


  const candidates = [

    prediction.category_key,

    prediction.category,

    prediction.source_class

  ];


  for (
    const candidate of candidates
  ) {

    const normalized =
      normalizeKey(
        candidate
      );


    if (
      !normalized
    ) {

      continue;

    }


    const alias =
      WASTE_ALIASES[
        normalized
      ];


    if (
      alias &&
      WASTE_RULES[alias]
    ) {

      return WASTE_RULES[
        alias
      ];

    }


    if (
      WASTE_RULES[
        normalized
      ]
    ) {

      return WASTE_RULES[
        normalized
      ];

    }

  }


  return null;

}


// ============================================================
// CARD DE DETECÇÃO
// ============================================================

function updateDetectionCard(
  predictions
) {

  const valid =
    (
      predictions || []
    )
      .filter(
        prediction =>
          Number(
            prediction.score
          ) >= MIN_CONFIDENCE
      )
      .sort(
        (
          a,
          b
        ) =>
          Number(
            b.score
          ) -
          Number(
            a.score
          )
      );


  const best =
    valid[0];


  // ----------------------------------------------------------
  // NENHUMA DETECÇÃO VÁLIDA
  // ----------------------------------------------------------

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


  // ----------------------------------------------------------
  // DEBUG
  // ----------------------------------------------------------
  //
  // Isso vai mostrar no console exatamente o que o backend
  // está mandando.
  //

  console.log(
    '[EcoScan] Detecção recebida:',
    best
  );


  // ----------------------------------------------------------
  // ENCONTRAR REGRA
  // ----------------------------------------------------------

  const rule =
    resolveWasteRule(
      best
    );


  // ----------------------------------------------------------
  // DETECÇÃO SEM REGRA
  // ----------------------------------------------------------

  if (!rule) {

    console.warn(
      '[EcoScan] Regra não encontrada:',
      {
        category:
          best.category,

        category_key:
          best.category_key,

        source_class:
          best.source_class
      }
    );


    lastDetectionData =
      null;


    setDetectionCard({

      name:
        prettifyClassName(
          best.source_class ||
          best.category
        ),

      category:
        best.category ||
        'Indeterminado',

      bin:
        best.bin ||
        '-',

      dest:
        best.destination ||
        best.dest ||
        '-',

      time:
        best.decomposition ||
        best.time ||
        '-',

      fact:
        'Objeto reconhecido pelo modelo, mas sem regra de reciclagem cadastrada.'

    });


    return;

  }


  // ----------------------------------------------------------
  // DETECÇÃO COM REGRA
  // ----------------------------------------------------------

  lastDetectionData = {

    name:
      prettifyClassName(
        best.source_class ||
        best.category
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


// ============================================================
// ATUALIZAR CARD
// ============================================================

function setDetectionCard(
  data
) {

  if (!data) {

    return;

  }


  if (detName) {

    detName.textContent =
      data.name || '-';

  }


  if (detType) {

    detType.textContent =
      data.category || '-';

  }


  if (detBin) {

    detBin.textContent =
      data.bin || '-';

  }


  if (detDest) {

    detDest.textContent =
      data.dest || '-';

  }


  if (detTime) {

    detTime.textContent =
      data.time || '-';

  }


  if (detFact) {

    detFact.textContent =
      data.fact || '-';

  }

}


// ============================================================
// STATUS DA DETECÇÃO
// ============================================================

function setDetectionStatus(
  message
) {

  if (detFact) {

    detFact.textContent =
      message;

  }

}


// ============================================================
// SALVAR DETECÇÃO
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
    .trim()
    .replace(
      /[\s-]+/g,
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