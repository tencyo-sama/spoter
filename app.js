/**
 * app.js
 * Tactical Spotter HUD with YOLO26 Native End-to-End ONNX Inference
 * 
 * - Execution Provider: WebGPU priority with WASM fallback
 * - End-to-End (NMS-free) output parsing: [x1, y1, x2, y2, score, class_id]
 * - Pinhole distance estimation & MIL elevation calculation
 * - Military HUD drawing with dynamic lock-on brackets & reticle
 */

// ==========================================
// 定数・設定
// ==========================================
const MODEL_PATH = './yolo26n.onnx?v=20261019_04';
const INPUT_WIDTH = 640;
const INPUT_HEIGHT = 640;
const CONFIDENCE_THRESHOLD = 0.15; // 認識感度を0.15に調整 (室内・スマホ画角・遠距離でも確実に捉えられる高感度設定)
const TARGET_CLASS_ID = 0; // 0: person
const REAL_HUMAN_HEIGHT = 1.7; // 人間の想定平均身長 (m)

// COCO 80クラス名
const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush'
];

const OPTICS_MODES = [
  { id: 'nvg', label: 'OPTICS: NVG' },
  { id: 'amber', label: 'OPTICS: AMBER' },
  { id: 'flir', label: 'OPTICS: FLIR (B&W)' }
];

// ターゲット追尾モード定義 (初期値をALLにして室内テスト時でも確実に即時検知)
const TARGET_MODES = [
  {
    id: 'ALL',
    label: 'TGT: ALL',
    headerTag: 'TARGET: ALL HOSTILE',
    classIds: null // 全物体 (人・物・車すべて)
  },
  {
    id: 'PERSON',
    label: 'TGT: PERSON',
    headerTag: 'TARGET: PERSON',
    classIds: [0] // personのみ
  },
  {
    id: 'VEHICLE',
    label: 'TGT: VEHICLE',
    headerTag: 'TARGET: VEHICLE',
    classIds: [1, 2, 3, 5, 7] // bicycle, car, motorcycle, bus, truck
  }
];

// ==========================================
// アプリケーション状態管理
// ==========================================
const state = {
  session: null,
  activeEp: 'WASM',
  cameraStream: null,
  facingMode: 'environment', // 背面カメラ
  isRunning: false,
  isProcessing: false,
  themeIndex: 0, // OPTICS_MODES[0] -> 'nvg'
  targetModeIndex: 0, // TARGET_MODES[0] -> 'ALL'
  focalLength: 0,
  scaleInfo: { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0, renderWidth: window.innerWidth || 1080, renderHeight: window.innerHeight || 1920 },
  fps: 0,
  inferenceTime: 0,
  lastFrameTimestamp: performance.now(),
  frameCount: 0,
  renderCount: 0,
  windSpeed: 2.4,
  windAngle: 45,
  compassHeading: 0,
  debugMaxScore: 0,
  topDetection: null,
  lastError: '',
  useDemoSimulation: false // モデル未配置時のHUD動作確認用シミュレーション
};

// ==========================================
// DOM要素取得
// ==========================================
const video = document.getElementById('webcam');
const canvas = document.getElementById('hudCanvas');
// HUDベクター描画専用の透過コンテキスト (alpha: true)
const ctx = canvas.getContext('2d', { alpha: true });

const startModal = document.getElementById('startModal');
const startSystemBtn = document.getElementById('startSystemBtn');
const modalStatusLog = document.getElementById('modalStatusLog');

const opticsStatusEl = document.getElementById('opticsStatus');
const engineProviderEl = document.getElementById('engineProvider');
const inferenceTimeEl = document.getElementById('inferenceTime');
const fpsCounterEl = document.getElementById('fpsCounter');
const peakConfDisplayEl = document.getElementById('peakConfDisplay');
const windInfoEl = document.getElementById('windInfo');
const threatLevelEl = document.getElementById('threatLevel');
const lockCriteriaEl = document.getElementById('lockCriteria');
const targetSummaryEl = document.getElementById('targetSummary');
const compassTapeEl = document.getElementById('compassTape');
const toggleTargetBtn = document.getElementById('toggleTargetBtn');
const toggleThemeBtn = document.getElementById('toggleThemeBtn');
const toggleCameraBtn = document.getElementById('toggleCameraBtn');
const selectModelBtn = document.getElementById('selectModelBtn');
const modelFileInput = document.getElementById('modelFileInput');

// 前処理用オフスクリーンキャンバス
const preprocessCanvas = document.createElement('canvas');
preprocessCanvas.width = INPUT_WIDTH;
preprocessCanvas.height = INPUT_HEIGHT;
const preprocessCtx = preprocessCanvas.getContext('2d', { willReadFrequently: true });

// ==========================================
// イベントリスナー初期化
// ==========================================
startSystemBtn.addEventListener('click', initSystem);
toggleTargetBtn.addEventListener('click', toggleTargetMode);
toggleThemeBtn.addEventListener('click', toggleTheme);
toggleCameraBtn.addEventListener('click', switchCamera);
selectModelBtn.addEventListener('click', () => modelFileInput.click());
modelFileInput.addEventListener('change', handleCustomModelUpload);

// ジャイロ / コンパス追従
if (window.DeviceOrientationEvent) {
  window.addEventListener('deviceorientation', (e) => {
    if (e.webkitCompassHeading) {
      state.compassHeading = e.webkitCompassHeading;
    } else if (e.alpha !== null) {
      state.compassHeading = (360 - e.alpha) % 360;
    }
    updateCompassDisplay();
  });
}

// リサイズ対応
window.addEventListener('resize', handleResize);

// ==========================================
// システム初期化
// ==========================================
async function initSystem() {
  startSystemBtn.disabled = true;
  updateStatusLog('Initializing camera & optics subsystem...');

  try {
    // 1. カメラ起動 (失敗時はPC用シミュレーション映像にフォールバック)
    try {
      await startCamera();
    } catch (camErr) {
      console.warn('Camera could not be started (using simulated feed for PC/testing):', camErr);
      state.isSimulatedCamera = true;
    }
    handleResize();

    // 2. ONNX Runtime Web セッション初期化 (WebGPU優先 -> WASMフォールバック)
    updateStatusLog('Establishing ONNX Runtime Web session...');
    await initOnnxSession(MODEL_PATH);

    // 3. UI切替 & 推論ループ開始
    startModal.style.display = 'none';
    state.isRunning = true;
    opticsStatusEl.textContent = 'ONLINE';
    opticsStatusEl.style.color = 'var(--primary-color)';

    // 環境風速ダミーシミュレータ (1.5秒周期で微変動)
    setInterval(updateWindEnvironment, 1500);

    // メインループ起動
    requestAnimationFrame(renderLoop);

  } catch (err) {
    console.error('Initialization error:', err);
    updateStatusLog(`ERROR: ${err.message || err}\nCheck console or load a custom .onnx model.`);
    startSystemBtn.disabled = false;
  }
}

function updateStatusLog(msg) {
  if (modalStatusLog) {
    modalStatusLog.textContent = msg;
  }
}

// ==========================================
// カメラ制御
// ==========================================
async function startCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(track => track.stop());
  }

  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: state.facingMode },
      width: { ideal: 1280 },
      height: { ideal: 720 }
    }
  };

  state.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = state.cameraStream;
  state.isSimulatedCamera = false;

  return new Promise((resolve) => {
    let resolved = false;
    const onReady = () => {
      handleResize();
      video.play().catch(e => console.warn('Video play warning:', e));
      if (!resolved) {
        resolved = true;
        resolve();
      }
    };
    video.onloadedmetadata = onReady;
    video.onloadeddata = onReady;
    video.oncanplay = onReady;
    // タイムアウトフォールバック
    setTimeout(onReady, 1000);
  });
}

video.addEventListener('loadeddata', handleResize);
video.addEventListener('playing', handleResize);
video.addEventListener('resize', handleResize);

async function switchCamera() {
  state.facingMode = state.facingMode === 'environment' ? 'user' : 'environment';
  await startCamera();
  handleResize();
}

function handleResize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  // カメラ焦点距離ピクセル目安 (canvas.height * 1.2)
  state.focalLength = canvas.height * 1.2;

  // videoWidth/Height 未準備時でも破綻しないフォールバック
  const vw = (video && video.videoWidth > 0) ? video.videoWidth : (canvas.width || 1280);
  const vh = (video && video.videoHeight > 0) ? video.videoHeight : (canvas.height || 720);

  const videoAspect = vw / vh;
  const canvasAspect = canvas.width / canvas.height;

  let renderWidth, renderHeight, offsetX, offsetY;

  if (canvasAspect > videoAspect) {
    renderWidth = canvas.width;
    renderHeight = canvas.width / videoAspect;
    offsetX = 0;
    offsetY = (canvas.height - renderHeight) / 2;
  } else {
    renderHeight = canvas.height;
    renderWidth = canvas.height * videoAspect;
    offsetX = (canvas.width - renderWidth) / 2;
    offsetY = 0;
  }

  state.scaleInfo = {
    scaleX: renderWidth / vw,
    scaleY: renderHeight / vh,
    offsetX,
    offsetY,
    renderWidth,
    renderHeight
  };
}

// ==========================================
// ONNX Runtime Web セッション確立 (WebGPU -> WASM)
// ==========================================
async function initOnnxSession(modelSource) {
  if (typeof ort === 'undefined') {
    console.warn('ONNX Runtime Web library (ort) is not loaded or blocked. Falling back to simulation demo.');
    state.useDemoSimulation = true;
    engineProviderEl.textContent = 'MOCK / SIM';
    engineProviderEl.style.color = '#FFB000';
    return;
  }

  // WASMリソースパスの設定
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);

  // WebGPU の利用可否チェックと優先試行
  let session = null;
  const tryProviders = ['webgpu', 'wasm'];

  for (const ep of tryProviders) {
    try {
      updateStatusLog(`Attempting provider: [${ep.toUpperCase()}]...`);
      const options = {
        executionProviders: [ep],
        graphOptimizationLevel: 'all'
      };

      if (typeof modelSource === 'string') {
        session = await ort.InferenceSession.create(modelSource, options);
      } else {
        // ArrayBuffer (ユーザーアップロード)
        session = await ort.InferenceSession.create(modelSource, options);
      }

      state.activeEp = ep.toUpperCase();
      console.log(`[✓] ONNX Session created successfully with EP: ${state.activeEp}`);
      break;
    } catch (e) {
      console.warn(`Execution provider [${ep}] failed:`, e);
      if (ep === 'wasm') {
        // 全プロバイダで失敗した場合 (モデルファイル未配置など)
        console.warn('Could not load yolo26n.onnx. Enabling simulation demo mode.');
        state.useDemoSimulation = true;
      }
    }
  }

  state.session = session;
  engineProviderEl.textContent = state.useDemoSimulation ? 'MOCK / SIM' : state.activeEp;
  engineProviderEl.style.color = state.activeEp === 'WEBGPU' ? 'var(--primary-color)' : '#FFB000';
}

async function handleCustomModelUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  try {
    updateStatusLog(`Loading custom model: ${file.name}...`);
    const buffer = await file.arrayBuffer();
    await initOnnxSession(buffer);
    state.useDemoSimulation = false;
    alert(`Model [${file.name}] loaded successfully via ${state.activeEp}!`);
  } catch (err) {
    alert(`Failed to load custom model: ${err.message}`);
  }
}

// ==========================================
// 画像前処理 (640x640 リサイズ & RGB正規化)
// ==========================================
function preprocessImage(sourceVideo) {
  // video準備完了時のみdrawImageを実行 (クラッシュ防止)
  if (sourceVideo && sourceVideo.videoWidth > 0 && sourceVideo.videoHeight > 0) {
    preprocessCtx.drawImage(sourceVideo, 0, 0, INPUT_WIDTH, INPUT_HEIGHT);
  } else {
    preprocessCtx.fillStyle = '#000000';
    preprocessCtx.fillRect(0, 0, INPUT_WIDTH, INPUT_HEIGHT);
  }

  const imgData = preprocessCtx.getImageData(0, 0, INPUT_WIDTH, INPUT_HEIGHT);
  const { data } = imgData;

  const float32Data = new Float32Array(3 * INPUT_WIDTH * INPUT_HEIGHT);
  const channelSize = INPUT_WIDTH * INPUT_HEIGHT;

  // CHW形式 (1, 3, 640, 640) & 0.0〜1.0正規化
  for (let i = 0; i < channelSize; i++) {
    float32Data[i] = data[i * 4] / 255.0;                          // R
    float32Data[channelSize + i] = data[i * 4 + 1] / 255.0;        // G
    float32Data[channelSize * 2 + i] = data[i * 4 + 2] / 255.0;    // B
  }

  return new ort.Tensor('float32', float32Data, [1, 3, INPUT_HEIGHT, INPUT_WIDTH]);
}

// ==========================================
// 推論実行 & End-to-End (NMS-free) 出力パーサー
// ==========================================
async function runInference() {
  if (!state.session) {
    if (state.useDemoSimulation) {
      return generateSimulatedTargets();
    }
    return [];
  }

  const startTime = performance.now();
  let inputTensor;
  try {
    inputTensor = preprocessImage(video);
  } catch (prepErr) {
    state.lastError = `Preprocess: ${prepErr.message}`;
    return [];
  }

  const inputName = state.session.inputNames[0];
  const feeds = {};
  feeds[inputName] = inputTensor;

  let results;
  try {
    results = await state.session.run(feeds);
  } catch (runErr) {
    console.error('Inference session.run error:', runErr);
    state.lastError = `Infer Err: ${runErr.message || runErr}`;

    // WebGPU で実行時エラーが発生した場合、直ちに WASM に安全フォールバック
    if (state.activeEp === 'WEBGPU') {
      console.warn('WebGPU execution failed. Falling back to WASM...');
      state.activeEp = 'WASM';
      engineProviderEl.textContent = 'WASM (FALLBACK)';
      engineProviderEl.style.color = '#FFB000';
      try {
        state.session = await ort.InferenceSession.create(MODEL_PATH, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
        state.lastError = '';
        console.log('[✓] Switched session to WASM successfully');
      } catch (fbErr) {
        console.error('WASM fallback failed:', fbErr);
        state.lastError = `WASM Init Err: ${fbErr.message}`;
      }
    }
    return [];
  }

  state.inferenceTime = (performance.now() - startTime).toFixed(1);
  state.lastError = '';

  // 出力テンソルのパース
  const outputName = state.session.outputNames[0];
  const outputTensor = results[outputName];
  return parseYOLO26EndToEndOutput(outputTensor);
}

// クラス別想定サイズ (ピンホール測距用)
function getTargetRealHeight(classId) {
  if (classId === 0) return 1.7; // 人間 (1.7m)
  if ([2, 5, 7].includes(classId)) return 1.5; // 乗用車・バス・トラック等の車高目安 (1.5m)
  if ([1, 3].includes(classId)) return 1.1; // 自転車・バイク (1.1m)
  return 0.8; // その他 (0.8m)
}

/**
 * YOLO26 Native End-to-End (NMS-free) 出力デコーダー
 * 出力形状仕様: [1, 300, 6] または [batch, num_detections, 6]
 * 行フォーマット: [x1, y1, x2, y2, score, class_id]
 */
function parseYOLO26EndToEndOutput(tensor) {
  if (!tensor || !tensor.data) return [];

  const data = tensor.data;
  const dims = tensor.dims;
  const detections = [];
  const currentTargetMode = TARGET_MODES[state.targetModeIndex];

  let numBoxes = 300;
  let elementStride = 6;

  if (dims.length === 3) {
    if (dims[2] === 6) {
      numBoxes = dims[1];
      elementStride = 6;
    } else if (dims[1] === 6) {
      // 転置形式 [1, 6, 300] の場合
      numBoxes = dims[2];
      for (let i = 0; i < numBoxes; i++) {
        const x1 = data[0 * numBoxes + i];
        const y1 = data[1 * numBoxes + i];
        const x2 = data[2 * numBoxes + i];
        const y2 = data[3 * numBoxes + i];
        const score = data[4 * numBoxes + i];
        const classId = Math.round(data[5 * numBoxes + i]);

        const isMatch = (!currentTargetMode.classIds || currentTargetMode.classIds.includes(classId));
        if (score >= CONFIDENCE_THRESHOLD && isMatch) {
          detections.push(scaleBoundingBox(x1, y1, x2, y2, score, classId));
        }
      }
      return detections;
    }
  } else if (dims.length === 2 && dims[1] === 6) {
    numBoxes = dims[0];
    elementStride = 6;
  }

  let maxScore = 0;
  let topDetection = null;

  // 標準形式 [1, num_boxes, 6]
  for (let i = 0; i < numBoxes; i++) {
    const offset = i * elementStride;
    const x1 = data[offset];
    const y1 = data[offset + 1];
    const x2 = data[offset + 2];
    const y2 = data[offset + 3];
    const score = data[offset + 4];
    const classId = Math.round(data[offset + 5]);

    if (score > maxScore) {
      maxScore = score;
      topDetection = { classId, score, className: COCO_CLASSES[classId] || `CLASS_${classId}` };
    }

    const isMatch = (!currentTargetMode.classIds || currentTargetMode.classIds.includes(classId));
    if (score >= CONFIDENCE_THRESHOLD && isMatch) {
      detections.push(scaleBoundingBox(x1, y1, x2, y2, score, classId));
    }
  }

  state.debugMaxScore = maxScore;
  state.topDetection = topDetection;
  return detections;
}

/**
 * 640x640 または 0~1 正規化座標系から Canvas描画座標系へのマッピング (NaN防止・完全安全版)
 */
function scaleBoundingBox(x1, y1, x2, y2, score, classId) {
  // 座標が 0.0〜1.0 正規化系か、0〜640ピクセル系かを判定
  const maxVal = Math.max(Math.abs(x1), Math.abs(x2), Math.abs(y1), Math.abs(y2));
  const isNormalized = (maxVal <= 1.05 && maxVal > 0);

  let normX1 = isNormalized ? x1 : x1 / INPUT_WIDTH;
  let normY1 = isNormalized ? y1 : y1 / INPUT_HEIGHT;
  let normX2 = isNormalized ? x2 : x2 / INPUT_WIDTH;
  let normY2 = isNormalized ? y2 : y2 / INPUT_HEIGHT;

  // 0.0〜1.0にクランプ
  normX1 = Math.max(0, Math.min(1, normX1));
  normY1 = Math.max(0, Math.min(1, normY1));
  normX2 = Math.max(0, Math.min(1, normX2));
  normY2 = Math.max(0, Math.min(1, normY2));

  const rW = (state.scaleInfo && state.scaleInfo.renderWidth > 0) ? state.scaleInfo.renderWidth : canvas.width;
  const rH = (state.scaleInfo && state.scaleInfo.renderHeight > 0) ? state.scaleInfo.renderHeight : canvas.height;
  const offX = (state.scaleInfo && typeof state.scaleInfo.offsetX === 'number') ? state.scaleInfo.offsetX : 0;
  const offY = (state.scaleInfo && typeof state.scaleInfo.offsetY === 'number') ? state.scaleInfo.offsetY : 0;

  const boxX = normX1 * rW + offX;
  const boxY = normY1 * rH + offY;
  const boxW = Math.max(10, (normX2 - normX1) * rW);
  const boxH = Math.max(10, (normY2 - normY1) * rH);

  return {
    x: boxX,
    y: boxY,
    width: boxW,
    height: boxH,
    score: score,
    classId: classId,
    className: COCO_CLASSES[classId] || `CLASS_${classId}`
  };
}

// ==========================================
// 観測パラメータ・距離計算 (Pinhole & Elevation MIL)
// ==========================================
function calculateBallisticParams(target) {
  const centerY = canvas.height / 2;

  // 1. ターゲット種別ごとの実サイズを用いた距離算出（ピンホール近似）
  const realHeight = getTargetRealHeight(target.classId);
  const distance = (realHeight * state.focalLength) / Math.max(1, target.height);

  // 2. 仰角オフセット (Elevation / MIL)
  const targetCenterY = target.y + target.height / 2;
  const elevationMil = (((targetCenterY - centerY) / state.focalLength) * 1000).toFixed(1);

  return {
    distance: distance.toFixed(1),
    elevationMil: elevationMil > 0 ? `+${elevationMil}` : `${elevationMil}`
  };
}

// ==========================================
// メイン描画・推論ループ (2フレームに1回推論 + 毎フレーム補間描画)
// ==========================================
let rawDetections = [];
let smoothedTargets = []; // 毎フレームスムーズにLerp補間される描画用ターゲット

async function renderLoop() {
  if (!state.isRunning) return;

  state.renderCount++;

  // ビデオ解像度が遅れて確定した場合の自動再計算
  if (video.videoWidth > 0 && (!state.scaleInfo.renderWidth || state.scaleInfo.renderWidth === canvas.width)) {
    handleResize();
  }

  // FPS計測
  calculateFps();

  // 1. HUDキャンバスのクリア
  drawCameraFeed();

  // 2. 推論実行：スマホカメラ準備判定 (videoWidth > 0 かつ readyState >= 2)
  const isVideoReady = (!state.isSimulatedCamera && video.videoWidth > 0 && (video.readyState >= 2 || video.currentTime > 0));
  const canRunInference = (isVideoReady || state.isSimulatedCamera);
  const shouldInferThisFrame = (state.renderCount % 2 === 0);

  if (shouldInferThisFrame && !state.isProcessing && canRunInference) {
    state.isProcessing = true;
    runInference().then((detections) => {
      rawDetections = detections;
      state.isProcessing = false;
      updateHudStats(detections);
    }).catch(err => {
      console.error('Inference error:', err);
      state.isProcessing = false;
    });
  }

  // 3. バウンディングボックスの線形補間（Lerp）追従計算
  updateSmoothedTargets();

  // 4. レティクル（中央十字線 & MILドット目盛り）描画
  drawReticle();

  // 5. ターゲット検出枠（カギ括弧 [ ]）および情報テキスト描画 (毎フレーム描画)
  drawTargets(smoothedTargets);

  requestAnimationFrame(renderLoop);
}

/** ターゲット位置の指数移動平均（Lerp）補間処理 */
function updateSmoothedTargets() {
  const lerpFactor = 0.5;

  if (!rawDetections || rawDetections.length === 0) {
    smoothedTargets = [];
    return;
  }

  // ターゲット数が異なる場合は直接更新
  if (smoothedTargets.length !== rawDetections.length) {
    smoothedTargets = rawDetections.map(d => ({ ...d }));
    return;
  }

  // 同一ターゲット数なら滑らかに追尾
  for (let i = 0; i < rawDetections.length; i++) {
    const raw = rawDetections[i];
    const cur = smoothedTargets[i];

    if (!cur || cur.classId !== raw.classId) {
      smoothedTargets[i] = { ...raw };
      continue;
    }

    cur.x += (raw.x - cur.x) * lerpFactor;
    cur.y += (raw.y - cur.y) * lerpFactor;
    cur.width += (raw.width - cur.width) * lerpFactor;
    cur.height += (raw.height - cur.height) * lerpFactor;
    cur.score = raw.score;
  }
}

// ==========================================
// HUD グラフィック描画ルーチン
// ==========================================

/** カメラ映像は背面videoがCSS GPUフィルターでゼロ負荷直接描画。Canvasはクリアのみ */
function drawCameraFeed() {
  // 高速全画面クリア (GPUコンポジタにより最速)
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (state.isSimulatedCamera || !video.videoWidth) {
    // カメラ未接続時 (PC等のテスト用暗視スコープ背景)
    ctx.save();
    const primaryDim = getComputedStyle(document.body).getPropertyValue('--primary-dim').trim() || 'rgba(0, 255, 102, 0.15)';
    ctx.strokeStyle = primaryDim;
    ctx.lineWidth = 1;
    const gridSize = 40;
    
    // 背景グリッド線
    for (let x = 0; x < canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // 中央にシミュレーション案内
    ctx.fillStyle = primaryDim;
    ctx.font = '12px "Share Tech Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('[SENSOR FEED: SYNTHETIC TEST PATTERN]', canvas.width / 2, canvas.height * 0.28);
    ctx.restore();
  }
}

/** 画面中央の細い十字線とMILドットスケール目盛りの描画 */
function drawReticle() {
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const primaryColor = getComputedStyle(document.body).getPropertyValue('--primary-color').trim() || '#00FF66';

  ctx.save();
  ctx.strokeStyle = primaryColor;
  ctx.fillStyle = primaryColor;
  ctx.lineWidth = 1;
  ctx.shadowColor = primaryColor;
  ctx.shadowBlur = 4;

  const gap = 14;
  const length = 90;

  // 十字線 (水平・垂直)
  ctx.beginPath();
  // 水平線 (左)
  ctx.moveTo(cx - length, cy);
  ctx.lineTo(cx - gap, cy);
  // 水平線 (右)
  ctx.moveTo(cx + gap, cy);
  ctx.lineTo(cx + length, cy);
  // 垂直線 (上)
  ctx.moveTo(cx, cy - length);
  ctx.lineTo(cx, cy - gap);
  // 垂直線 (下)
  ctx.moveTo(cx, cy + gap);
  ctx.lineTo(cx, cy + length);
  ctx.stroke();

  // 中央ミニドット
  ctx.beginPath();
  ctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
  ctx.fill();

  // MIL ドット目盛り描画 (上下左右に等間隔目盛り)
  const milStep = 18;
  const numSteps = 4;

  for (let i = 1; i <= numSteps; i++) {
    const d = gap + i * milStep;
    const tick = i % 2 === 0 ? 5 : 2.5;

    // 水平軸目盛り
    ctx.beginPath();
    ctx.moveTo(cx - d, cy - tick);
    ctx.lineTo(cx - d, cy + tick);
    ctx.moveTo(cx + d, cy - tick);
    ctx.lineTo(cx + d, cy + tick);
    // 垂直軸目盛り
    ctx.moveTo(cx - tick, cy - d);
    ctx.lineTo(cx + tick, cy - d);
    ctx.moveTo(cx - tick, cy + d);
    ctx.lineTo(cx + tick, cy + d);
    ctx.stroke();
  }

  // 中央レティクル円弧
  ctx.beginPath();
  ctx.arc(cx, cy, 40, 0, Math.PI * 2);
  ctx.setLineDash([4, 16]);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.restore();
}

/** ターゲットの四隅「カギ括弧 [ ]」および観測パラメータ描画 */
function drawTargets(detections) {
  const primaryColor = getComputedStyle(document.body).getPropertyValue('--primary-color').trim() || '#00FF66';
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const currentTargetMode = TARGET_MODES[state.targetModeIndex];

  detections.forEach((target) => {
    // 現在のターゲットモード対象なら主ターゲットとしてロックオン
    const isPrimaryTarget = (!currentTargetMode.classIds || currentTargetMode.classIds.includes(target.classId));
    const { distance, elevationMil } = calculateBallisticParams(target);

    ctx.save();
    ctx.strokeStyle = primaryColor;
    ctx.fillStyle = primaryColor;
    ctx.lineWidth = isPrimaryTarget ? 2 : 1;
    ctx.shadowColor = primaryColor;
    ctx.shadowBlur = isPrimaryTarget ? 6 : 2;

    const x = target.x;
    const y = target.y;
    const w = target.width;
    const h = target.height;

    // --- カギ括弧 [ ] の描画 ---
    const bracketSize = Math.min(22, Math.min(w, h) * 0.28);

    ctx.beginPath();
    // 左上 ┌
    ctx.moveTo(x, y + bracketSize);
    ctx.lineTo(x, y);
    ctx.lineTo(x + bracketSize, y);

    // 右上 ┐
    ctx.moveTo(x + w - bracketSize, y);
    ctx.lineTo(x + w);
    ctx.lineTo(x + w, y + bracketSize);

    // 左下 └
    ctx.moveTo(x, y + h - bracketSize);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + bracketSize, y + h);

    // 右下 ┘
    ctx.moveTo(x + w - bracketSize, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - bracketSize);
    ctx.stroke();

    // ターゲット中心と画面中心の照準リード線（主ターゲットのみ結線）
    if (isPrimaryTarget) {
      const targetCx = x + w / 2;
      const targetCy = y + h / 2;
      ctx.beginPath();
      ctx.setLineDash([2, 6]);
      ctx.strokeStyle = primaryColor;
      ctx.moveTo(cx, cy);
      ctx.lineTo(targetCx, targetCy);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // --- 情報テキストの描画 (直上 / 直下) ---
    ctx.font = 'bold 11px "Share Tech Mono", "Courier New", monospace';

    // 上部ヘッダータグ: [LOCKED] HOSTILE (NAME)
    const statusText = isPrimaryTarget 
      ? `[LOCKED] HOSTILE (${target.className.toUpperCase()})` 
      : `[TRACK] ${target.className.toUpperCase()}`;
    
    ctx.fillText(statusText, x, y - 8);

    // 枠下部: 距離(Range) & 仰角(Elevation MIL) & 信頼度
    const infoLine1 = `RNG : ${distance} m`;
    const infoLine2 = `ELEV: ${elevationMil} MIL  CONF: ${(target.score * 100).toFixed(0)}%`;

    // 背景に暗いバッジを描画して視認性を確保
    const textWidth = Math.max(ctx.measureText(infoLine1).width, ctx.measureText(infoLine2).width);
    ctx.fillStyle = 'rgba(0, 20, 5, 0.65)';
    ctx.fillRect(x - 2, y + h + 4, textWidth + 8, 30);

    ctx.fillStyle = primaryColor;
    ctx.fillText(infoLine1, x + 2, y + h + 16);
    ctx.fillText(infoLine2, x + 2, y + h + 28);

    ctx.restore();
  });
}

// ==========================================
// UI & 環境パラメータ更新
// ==========================================
function updateHudStats(detections) {
  if (targetSummaryEl) {
    const maxConf = (typeof state.debugMaxScore === 'number' && state.debugMaxScore > 0)
      ? `${Math.round(state.debugMaxScore * 100)}%`
      : '0%';
    targetSummaryEl.textContent = `TARGETS: ${detections.length} [PEAK: ${maxConf}]`;
  }
  if (peakConfDisplayEl) {
    if (state.topDetection && state.topDetection.score > 0.05) {
      const pct = Math.round(state.topDetection.score * 100);
      peakConfDisplayEl.textContent = `${pct}% (${state.topDetection.className.toUpperCase()})`;
      peakConfDisplayEl.style.color = pct >= 15 ? 'var(--primary-color)' : '#FFB000';
    } else {
      peakConfDisplayEl.textContent = 'NO ECHO';
      peakConfDisplayEl.style.color = 'rgba(255,255,255,0.4)';
    }
  }
  if (lockCriteriaEl) {
    lockCriteriaEl.textContent = `LOCK-ON CRITERIA: CONF >= ${Math.round(CONFIDENCE_THRESHOLD * 100)}%`;
  }
  if (inferenceTimeEl) {
    if (state.lastError) {
      inferenceTimeEl.textContent = state.lastError;
      inferenceTimeEl.style.color = '#ff3344';
    } else {
      inferenceTimeEl.textContent = `${state.inferenceTime} ms (${state.activeEp})`;
      inferenceTimeEl.style.color = 'var(--primary-color)';
    }
  }
}

function calculateFps() {
  const now = performance.now();
  state.frameCount++;
  const delta = now - state.lastFrameTimestamp;

  if (delta >= 1000) {
    state.fps = Math.round((state.frameCount * 1000) / delta);
    fpsCounterEl.textContent = `${state.fps} FPS`;
    state.frameCount = 0;
    state.lastFrameTimestamp = now;
  }
}

function updateWindEnvironment() {
  // 風速: 1.8 〜 4.5 m/s の範囲でランダム微変動
  state.windSpeed = (2.2 + (Math.random() * 1.8 - 0.9)).toFixed(1);
  // 風向角: 020° 〜 075°
  state.windAngle = Math.round(45 + (Math.random() * 20 - 10));
  windInfoEl.textContent = `WIND: ${state.windSpeed} m/s ${String(state.windAngle).padStart(3, '0')}° >`;
}

function updateCompassDisplay() {
  if (compassTapeEl) {
    const offset = -(state.compassHeading % 360) * 2;
    compassTapeEl.style.transform = `translateX(${offset}px)`;
  }
}

function toggleTargetMode() {
  state.targetModeIndex = (state.targetModeIndex + 1) % TARGET_MODES.length;
  const currentMode = TARGET_MODES[state.targetModeIndex];
  toggleTargetBtn.textContent = currentMode.label;
  threatLevelEl.textContent = currentMode.headerTag;
}

function toggleTheme() {
  state.themeIndex = (state.themeIndex + 1) % OPTICS_MODES.length;
  const currentMode = OPTICS_MODES[state.themeIndex];
  document.body.setAttribute('data-theme', currentMode.id);
  toggleThemeBtn.textContent = currentMode.label;
}

// ==========================================
// モデル未配置時のシミュレーション用モック生成
// ==========================================
let mockAngle = 0;
function generateSimulatedTargets() {
  mockAngle += 0.03;
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;

  const currentMode = TARGET_MODES[state.targetModeIndex];

  if (currentMode.id === 'VEHICLE') {
    // 車両のシミュレーションターゲット (車高1.5m目安)
    const mockX = cx + Math.sin(mockAngle) * (canvas.width * 0.2) - 90;
    const mockY = cy + Math.cos(mockAngle * 0.6) * (canvas.height * 0.15) - 50;
    return [{
      x: mockX,
      y: mockY,
      width: 180,
      height: 110,
      score: 0.94,
      classId: 2,
      className: 'car'
    }];
  }

  // デフォルト / PERSONシミュレーションターゲット (身長1.7m目安)
  const mockX = cx + Math.sin(mockAngle) * (canvas.width * 0.22) - 45;
  const mockY = cy + Math.cos(mockAngle * 0.7) * (canvas.height * 0.18) - 90;

  return [
    {
      x: mockX,
      y: mockY,
      width: 90,
      height: 180,
      score: 0.92,
      classId: 0,
      className: 'person'
    }
  ];
}
