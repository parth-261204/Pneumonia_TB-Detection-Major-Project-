/* ═══════════════════════════════════════════
   PulmoScan AI — Clinical Interface Logic
   ═══════════════════════════════════════════ */

// ── Class metadata ──
const CLASS_META = {
  NORMAL: {
    color:   '#1e7e3e',
    fill:    '#1e7e3e',
    bg:      '#edfbf1',
    border:  '#a8dbb8',
    arcColor:'#1e7e3e',
    severity: { label: 'No Pathology Detected', color: '#1e7e3e', bg: '#edfbf1' },
    description: 'The chest radiograph does not demonstrate radiographic features of active pulmonary infection. No consolidation, cavitation, or infiltrates are identified. This finding is consistent with a normal lung parenchyma.',
  },
  PNEUMONIA: {
    color:   '#b45309',
    fill:    '#d97706',
    bg:      '#fff8ed',
    border:  '#f6cc8a',
    arcColor:'#d97706',
    severity: { label: 'Clinical Attention Recommended', color: '#b45309', bg: '#fff8ed' },
    description: 'The radiograph demonstrates features consistent with a pulmonary consolidation pattern, which may be indicative of bacterial or viral pneumonia. Correlation with clinical symptoms, laboratory findings, and specialist review is strongly advised.',
  },
  TUBERCULOSIS: {
    color:   '#b91c1c',
    fill:    '#dc2626',
    bg:      '#fff1f1',
    border:  '#f5b0b0',
    arcColor:'#dc2626',
    severity: { label: 'Urgent Clinical Review Required', color: '#b91c1c', bg: '#fff1f1' },
    description: 'The radiograph exhibits features potentially consistent with pulmonary tuberculosis, including possible apical infiltrates or cavitating lesions. This finding warrants urgent specialist review, sputum culture, and appropriate infection-control measures.',
  },
  UNKNOWN: {
    color:   '#6d28d9',
    fill:    '#7c3aed',
    bg:      '#f5f0ff',
    border:  '#c4b0f0',
    arcColor:'#7c3aed',
    severity: { label: 'Unable to Classify — Manual Review Required', color: '#6d28d9', bg: '#f5f0ff' },
    description: 'The uploaded image does not match any of the trained diagnostic patterns. This may indicate a non-standard radiograph, poor image quality, or a pathology outside the system\'s scope. Manual radiologist review is required.',
  },
};

const CLASS_ORDER  = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS', 'UNKNOWN'];
const MODEL_LABELS = ['DenseNet121', 'ResNet50', 'EfficientNetB0', 'MobileNetV2'];
const API_BASE_URL = (window.PULMOSCAN_API_URL || '').replace(/\/$/, '');
const apiUrl = path => `${API_BASE_URL}${path}`;

// ── DOM refs ──
const $  = id => document.getElementById(id);
const uploadZone    = $('uploadZone');
const uploadIdle    = $('uploadIdle');
const uploadPreview = $('uploadPreview');
const previewImg    = $('previewImg');
const previewFilename = $('previewFilename');
const fileInput     = $('fileInput');
const clearBtn      = $('clearBtn');
const analyzeBtn    = $('analyzeBtn');
const btnLabel      = $('btnLabel');
const btnLoading    = $('btnLoading');
const reportIdle    = $('reportIdle');
const report        = $('report');
const statusDot     = $('statusDot');
const statusText    = $('statusText');
const toast         = $('toast');

let selectedFile  = null;
let toastTimer    = null;

// ── System status ──
async function checkStatus() {
  statusDot.className = 'status-dot loading';
  statusText.textContent = 'Initialising models…';
  try {
    const res  = await fetch(apiUrl('/api/status'));
    const data = await res.json();
    const ok   = data.models_loaded && data.ensemble_ready;
    statusDot.className = 'status-dot ' + (ok ? 'ready' : 'error');
    statusText.textContent = ok
      ? `System ready · ${data.base_models.length} models · ${data.device.toUpperCase()}`
      : `Partial load: ${data.base_models.length}/4 models`;
    if (data.load_errors && Object.keys(data.load_errors).length) {
      const keys = Object.keys(data.load_errors).join(', ');
      statusText.textContent += ` (errors: ${keys})`;
    }
  } catch {
    statusDot.className = 'status-dot error';
    statusText.textContent = 'System unreachable';
  }
}
checkStatus();

// ── Drag & drop ──
uploadZone.addEventListener('dragenter', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', e => { e.preventDefault(); uploadZone.classList.remove('dragover'); });
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});
uploadZone.addEventListener('click', e => {
  if (!selectedFile && (e.target === uploadZone || uploadIdle.contains(e.target))) fileInput.click();
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(file) {
  if (!file.type.startsWith('image/')) { showToast('Please upload a valid image file.'); return; }
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewFilename.textContent = file.name;
  uploadIdle.style.display    = 'none';
  uploadPreview.style.display = 'flex';
  analyzeBtn.disabled         = false;
  showIdle();
}

clearBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src  = '';
  uploadPreview.style.display = 'none';
  uploadIdle.style.display    = 'flex';
  analyzeBtn.disabled         = true;
  showIdle();
});

// ── Analyse ──
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;
  setLoading(true);
  const fd = new FormData();
  fd.append('file', selectedFile);
  try {
    const res = await fetch(apiUrl('/api/predict'), { method: 'POST', body: fd });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || `HTTP ${res.status}`); }
    renderReport(await res.json());
  } catch (err) {
    showToast(`Analysis error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  analyzeBtn.disabled    = on;
  btnLabel.style.display   = on ? 'none'  : 'inline';
  btnLoading.style.display = on ? 'flex'  : 'none';
}

// ── Render report ──
function renderReport(data) {
  const cls  = data.predicted_class;
  const meta = CLASS_META[cls] || CLASS_META.UNKNOWN;
  const conf = data.ensemble_conf;

  // Timestamp + device
  $('reportMeta').textContent = `Generated ${new Date().toLocaleString('en-GB', { dateStyle:'long', timeStyle:'short' })} · AI-assisted, not clinician-verified`;
  $('reportDevice').innerHTML = `<span class="device-chip">${data.device?.toUpperCase() || 'CPU'}</span>`;

  // Primary finding
  findingCard.style.borderColor = meta.border;
  $('findingName').textContent  = cls.charAt(0) + cls.slice(1).toLowerCase();
  $('findingName').style.color  = meta.color;
  $('findingDesc').textContent  = meta.description;

  // Confidence ring
  const circ   = 238.8;
  const arc    = $('confArc');
  arc.style.stroke          = meta.arcColor;
  arc.style.strokeDashoffset = (circ - conf * circ).toFixed(2);
  $('confPct').textContent   = (conf * 100).toFixed(1) + '%';
  $('confPct').style.color   = meta.color;

  // Severity strip
  const sev = meta.severity;
  const strip = $('severityStrip');
  strip.style.background = sev.bg;
  strip.style.color      = sev.color;
  strip.innerHTML = `<span style="font-size:15px">●</span> ${sev.label}`;

  // Differential probability bars
  const diffRows = $('diffRows');
  diffRows.innerHTML = '';
  CLASS_ORDER.forEach(c => {
    const p   = (data.ensemble_probs[c] || 0) * 100;
    const cm  = CLASS_META[c];
    const row = document.createElement('div');
    row.className = 'diff-row';
    row.innerHTML = `
      <div class="diff-label">
        <span class="diff-dot" style="background:${cm.color}"></span>${c.charAt(0)+c.slice(1).toLowerCase()}
      </div>
      <div class="diff-track">
        <div class="diff-fill" style="background:${cm.fill}" data-w="${p.toFixed(2)}"></div>
      </div>
      <div class="diff-pct" style="color:${c===cls?cm.color:'var(--text-soft)'}">${p.toFixed(1)}%</div>`;
    diffRows.appendChild(row);
  });

  // Model consensus cards
  const grid = $('consensusGrid');
  grid.innerHTML = '';
  MODEL_LABELS.forEach(label => {
    const info  = data.base_predictions[label] || {};
    const pCls  = info.class || 'N/A';
    const pm    = CLASS_META[pCls] || {};
    const agree = pCls === cls;

    const card = document.createElement('div');
    card.className = 'consensus-card' + (agree ? ' agree' : '');

    const probs = info.probs || {};
    const miniRows = CLASS_ORDER.map(c => {
      const v  = ((probs[c] || 0) * 100).toFixed(1);
      const cm = CLASS_META[c];
      return `<div class="mini-row">
        <div class="mini-track">
          <div class="mini-fill" style="background:${cm.fill}" data-w="${v}"></div>
        </div>
        <div class="mini-pct">${v}%</div>
      </div>`;
    }).join('');

    card.innerHTML = `
      <div class="consensus-model">${label}</div>
      <div class="consensus-pred" style="color:${pm.color||'var(--text)'}">${pCls.charAt(0)+(pCls.slice(1).toLowerCase())}</div>
      <div class="consensus-conf">${((info.confidence||0)*100).toFixed(1)}% confidence ${agree ? '· ✓ Agrees with ensemble' : ''}</div>
      <div class="mini-bars">${miniRows}</div>`;
    grid.appendChild(card);
  });

  // XAI
  const hasGC  = !!data.gradcam_image;
  const hasSal = !!data.saliency_image;
  if (hasGC || hasSal) {
    $('xaiSection').style.display = 'flex';
    $('xaiSection').style.flexDirection = 'column';
    $('xaiSection').style.gap = '14px';

    if (hasGC) {
      $('gradcamImg').src = 'data:image/png;base64,' + data.gradcam_image;
      $('gradcamImg').style.display = 'block';
    }
    if (hasSal) {
      $('saliencyImg').src = 'data:image/png;base64,' + data.saliency_image;
    }
    $('xaiNote').textContent =
      `Grad-CAM computed using ${data.gradcam_model} — the network with highest confidence for the predicted class. ` +
      `Red/yellow regions indicate high diagnostic weight; blue regions are low influence.`;

    document.querySelectorAll('.xai-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.xai-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const which = btn.dataset.tab;
        $('gradcamImg').style.display  = which === 'gradcam'  ? 'block' : 'none';
        $('saliencyImg').style.display = which === 'saliency' ? 'block' : 'none';
      });
    });
  } else {
    $('xaiSection').style.display = 'none';
  }

  $('footerModel').textContent = ` Ensemble: ${MODEL_LABELS.join(', ')} → Logistic Regression meta-learner.`;

  // Show report
  reportIdle.style.display = 'none';
  report.style.display     = 'flex';

  // Animate bars after paint
  requestAnimationFrame(() => {
    document.querySelectorAll('.diff-fill').forEach(el => { el.style.width = el.dataset.w + '%'; });
    document.querySelectorAll('.mini-fill').forEach(el => { el.style.width = el.dataset.w + '%'; });
  });
}

function showIdle() {
  reportIdle.style.display = 'flex';
  report.style.display     = 'none';
}

function showToast(msg) {
  clearTimeout(toastTimer);
  toast.textContent = msg;
  toast.classList.add('show');
  toastTimer = setTimeout(() => toast.classList.remove('show'), 4500);
}
