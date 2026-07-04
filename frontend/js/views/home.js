/**
 * Home View — Problem Input Workspace
 */
import { api } from '../api.js';
import { store } from '../state.js';
import { runPipeline, regeneratePipeline, createInitialPipelineState } from '../pipeline.js';
import { mountPipelineView, updatePipelineView } from './pipeline-view.js';

// Module-level pipeline state (persists across follow-ups in same session)
let _currentPipelineState = createInitialPipelineState();
let _container = null; // reference to main-content, set on renderHome

export function renderHome(container, followUpContext = null) {
  _container = container;
  container.innerHTML = `
    <div class="page">
      <div class="home-workspace">
        <div class="home-hero">
          <h1 class="page-header__title">What are we solving today?</h1>
          <p class="page-header__subtitle">System ready. Enter your constraints below.</p>
        </div>

        <!-- Input Mode Selector -->
        <div class="input-modes" id="input-modes">
          <button class="input-mode active" data-mode="text">
            <span class="material-symbols-outlined">edit_note</span>
            Text
          </button>
          <button class="input-mode" data-mode="image">
            <span class="material-symbols-outlined">image</span>
            Image
          </button>
          <button class="input-mode" data-mode="code">
            <span class="material-symbols-outlined">code</span>
            Code
          </button>
          <button class="input-mode" data-mode="voice">
            <span class="material-symbols-outlined">mic</span>
            Voice
          </button>
        </div>

        <!-- Text Input -->
        <div class="input-panel" id="text-panel">
          <textarea
            id="problem-input"
            class="input input--lg problem-textarea"
            placeholder="Describe your problem in detail. The more context you provide, the better the solution..."
            rows="8"
          ></textarea>

          <div class="input-toolbar">
            <div class="input-toolbar__left">
              <div class="domain-selector">
                <span class="text-label">Domain</span>
                <select id="domain-select" class="select">
                  <option value="">Auto-detect</option>
                  <option value="math">Mathematics</option>
                  <option value="code">Code & Programming</option>
                  <option value="science">Science</option>
                  <option value="general">General</option>
                </select>
              </div>
            </div>
            <div class="input-toolbar__right">
              <span id="char-count" class="text-label">0 characters</span>
              <button id="solve-btn" class="btn btn--primary btn--lg">
                <span class="material-symbols-outlined">psychology</span>
                Solve Problem
              </button>
            </div>
          </div>
        </div>

        <!-- Image Upload Panel -->
        <div class="input-panel hidden" id="image-panel">
          <div class="upload-zone" id="image-upload-zone">
            <span class="material-symbols-outlined upload-zone__icon">cloud_upload</span>
            <p class="upload-zone__text">
              <strong>Click to upload</strong> or drag and drop<br/>
              PNG, JPG, GIF up to 10MB
            </p>
            <input type="file" id="image-file-input" accept="image/*" hidden />
          </div>
          <div id="image-preview-area" class="hidden">
            <img id="image-preview" style="max-height: 300px; border-radius: var(--radius-lg); border: 1px solid var(--color-border);" />
            <button id="clear-image" class="btn btn--ghost btn--sm" style="margin-top: var(--space-sm);">
              <span class="material-symbols-outlined">close</span> Remove
            </button>
          </div>
          <button id="solve-image-btn" class="btn btn--primary btn--lg" disabled style="margin-top: var(--space-md);">
            <span class="material-symbols-outlined">psychology</span>
            Analyze Image
          </button>
        </div>

        <!-- Code Input Panel -->
        <div class="input-panel hidden" id="code-panel">
          <div class="code-input-header">
            <select id="code-lang-select" class="select" style="width: 180px;">
              <option value="">Auto-detect language</option>
              <option value="python">Python</option>
              <option value="javascript">JavaScript</option>
              <option value="java">Java</option>
              <option value="c">C</option>
              <option value="cpp">C++</option>
              <option value="go">Go</option>
              <option value="ruby">Ruby</option>
            </select>
          </div>
          <textarea
            id="code-input"
            class="input code-textarea"
            placeholder="Paste your code here..."
            rows="12"
            style="font-family: var(--font-code); font-size: var(--text-code-size);"
          ></textarea>
          <button id="solve-code-btn" class="btn btn--primary btn--lg" style="margin-top: var(--space-md);">
            <span class="material-symbols-outlined">psychology</span>
            Analyze Code
          </button>
        </div>

        <!-- Voice Input Panel -->
        <div class="input-panel hidden" id="voice-panel">
          <div class="voice-input-area">
            <button id="record-btn" class="voice-record-btn">
              <span class="material-symbols-outlined" id="record-icon">mic</span>
            </button>
            <p class="voice-status" id="voice-status">Click to start recording</p>
            <p class="voice-timer hidden" id="voice-timer">00:00</p>
            <div id="voice-transcription-preview" class="voice-transcription-preview hidden">
              <p class="text-label">Transcription Preview</p>
              <p id="voice-transcription-text" class="voice-transcription-text"></p>
            </div>
          </div>
        </div>

        <!-- Loading State -->
        <div class="solve-loading hidden" id="solve-loading">
          <div class="loading-overlay">
            <div class="spinner spinner--lg spinner--accent"></div>
            <p class="loading-overlay__text" id="loading-status">Analyzing problem...</p>
            <div class="loading-steps" id="loading-steps">
              <div class="loading-step active">
                <span class="chip chip--accent"><span class="chip__dot chip__dot--processing"></span> Processing</span>
                Input Analysis
              </div>
            </div>
          </div>
        </div>

        <!-- Session Link -->
        <div class="home-footer">
          <a href="#/sessions" class="btn btn--ghost">
            View Session History
            <span class="material-symbols-outlined">arrow_forward</span>
          </a>
        </div>
      </div>
    </div>
  `;

  // ── Event Handlers ──────────────────────────────────────────
  const problemInput = container.querySelector('#problem-input');
  const solveBtn = container.querySelector('#solve-btn');
  const charCount = container.querySelector('#char-count');
  const domainSelect = container.querySelector('#domain-select');
  const inputModes = container.querySelector('#input-modes');
  const solveLoading = container.querySelector('#solve-loading');

  // Character count
  problemInput.addEventListener('input', () => {
    charCount.textContent = `${problemInput.value.length} characters`;
  });

  // Input mode switching
  inputModes.addEventListener('click', (e) => {
    const modeBtn = e.target.closest('.input-mode');
    if (!modeBtn) return;
    const mode = modeBtn.dataset.mode;

    inputModes.querySelectorAll('.input-mode').forEach(b => b.classList.remove('active'));
    modeBtn.classList.add('active');

    container.querySelectorAll('.input-panel').forEach(p => p.classList.add('hidden'));
    container.querySelector(`#${mode}-panel`).classList.remove('hidden');
  });

  // Image upload
  const imageZone = container.querySelector('#image-upload-zone');
  const imageFileInput = container.querySelector('#image-file-input');
  const imagePreviewArea = container.querySelector('#image-preview-area');
  const imagePreview = container.querySelector('#image-preview');
  const solveImageBtn = container.querySelector('#solve-image-btn');
  const clearImage = container.querySelector('#clear-image');

  let selectedImageBase64 = null;

  imageZone.addEventListener('click', () => imageFileInput.click());
  imageZone.addEventListener('dragover', (e) => { e.preventDefault(); imageZone.classList.add('dragover'); });
  imageZone.addEventListener('dragleave', () => imageZone.classList.remove('dragover'));
  imageZone.addEventListener('drop', (e) => {
    e.preventDefault();
    imageZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleImageFile(file);
  });

  imageFileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleImageFile(e.target.files[0]);
  });

  clearImage.addEventListener('click', () => {
    selectedImageBase64 = null;
    imagePreviewArea.classList.add('hidden');
    imageZone.classList.remove('hidden');
    solveImageBtn.disabled = true;
  });

  function handleImageFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      selectedImageBase64 = dataUrl.split(',')[1];
      imagePreview.src = dataUrl;
      imagePreviewArea.classList.remove('hidden');
      imageZone.classList.add('hidden');
      solveImageBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Pre-populate textarea if returning from a follow-up
  if (followUpContext?.previousQuery) {
    problemInput.value = '';
    problemInput.placeholder = `Follow-up to: "${followUpContext.previousQuery.slice(0, 60)}…"`;
    problemInput.focus();
  } else {
    problemInput.focus();
  }

  // ── Helper: launch pipeline ──────────────────────────────────
  function launchPipeline(query, inputType) {
    _currentPipelineState = createInitialPipelineState();

    const context = followUpContext
      ? { query: followUpContext.previousQuery, answer: followUpContext.previousAnswer }
      : null;

    // Transition: hide input workspace, mount pipeline view
    mountPipelineView(container, query, {
      onBack: () => {
        // Return to clean home — no follow-up context
        renderHome(container);
      },
      onFollowUp: ({ query: prevQuery, answer: prevAnswer }) => {
        renderHome(container, {
          previousQuery: prevQuery,
          previousAnswer: prevAnswer,
        });
      },
      onRegenerate: () => {
        regeneratePipeline(_currentPipelineState, (newState) => {
          _currentPipelineState = newState;
          updatePipelineView(container, newState);
        });
      },
    });

    // Run pipeline
    runPipeline(query, inputType, context, (newState) => {
      _currentPipelineState = newState;
      updatePipelineView(container, newState);
    });
  }

  // Solve text problem
  solveBtn.addEventListener('click', () => {
    const text = problemInput.value.trim();
    if (!text) return;
    launchPipeline(text, 'text');
  });

  // Solve image
  solveImageBtn.addEventListener('click', () => {
    if (!selectedImageBase64) return;
    // Use the filename or a generic label as the query string for the pipeline
    const imageLabel = imagePreview.src
      ? 'Analyze the uploaded image and provide detailed insights'
      : 'Image analysis request';
    launchPipeline(imageLabel, 'image');
  });

  // Solve code
  const codeInput = container.querySelector('#code-input');
  const codeBtn = container.querySelector('#solve-code-btn');
  const codeLang = container.querySelector('#code-lang-select');

  codeBtn.addEventListener('click', () => {
    const code = codeInput.value.trim();
    if (!code) return;
    const lang = codeLang.value ? `${codeLang.value} ` : '';
    launchPipeline(`Analyze and explain this ${lang}code:\n\n${code.slice(0, 400)}`, 'code');
  });

  // ── Voice Recording ───────────────────────────────────────────
  const recordBtn = container.querySelector('#record-btn');
  const recordIcon = container.querySelector('#record-icon');
  const voiceStatus = container.querySelector('#voice-status');
  const voiceTimer = container.querySelector('#voice-timer');
  const voiceTranscriptionPreview = container.querySelector('#voice-transcription-preview');
  const voiceTranscriptionText = container.querySelector('#voice-transcription-text');

  let mediaRecorder = null;
  let audioChunks = [];
  let recordingStartTime = null;
  let timerInterval = null;

  recordBtn.addEventListener('click', async () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      // Stop recording
      mediaRecorder.stop();
      return;
    }

    // Request microphone access
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];

      mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstart = () => {
        recordBtn.classList.add('recording');
        recordIcon.textContent = 'stop';
        voiceStatus.textContent = 'Recording... Click to stop';
        voiceTimer.classList.remove('hidden');
        voiceTranscriptionPreview.classList.add('hidden');
        recordingStartTime = Date.now();
        timerInterval = setInterval(updateTimer, 1000);
      };

      mediaRecorder.onstop = async () => {
        clearInterval(timerInterval);
        recordBtn.classList.remove('recording');
        recordIcon.textContent = 'mic';
        voiceStatus.textContent = 'Processing audio...';
        voiceTimer.classList.add('hidden');

        // Stop all mic tracks
        stream.getTracks().forEach(t => t.stop());

        // Convert to blob and then to base64
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const base64Audio = await blobToBase64(audioBlob);

        // Launch pipeline with the transcription (best-effort)
        try {
          voiceStatus.textContent = 'Processing...';
          // Attempt transcription via backend, fall back to generic label
          let queryText = 'Analyze my voice query and provide a comprehensive answer';
          try {
            const result = await api.solveVoice(base64Audio);
            // If backend returns a transcription in metadata, use it
            if (result?.metadata?.transcription) {
              queryText = result.metadata.transcription;
            }
          } catch {
            // Backend offline or error — use generic label
          }
          launchPipeline(queryText, 'voice');
        } catch (err) {
          voiceStatus.textContent = 'Click to start recording';
          showToast(`Error: ${err.message}`, 'error');
        }
      };

      mediaRecorder.start(250); // Collect data every 250ms

    } catch (err) {
      if (err.name === 'NotAllowedError') {
        showToast('Microphone access denied. Please allow microphone access in your browser settings.', 'error');
      } else {
        showToast(`Microphone error: ${err.message}`, 'error');
      }
      voiceStatus.textContent = 'Click to start recording';
    }
  });

  function updateTimer() {
    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    voiceTimer.textContent = `${mins}:${secs}`;
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const dataUrl = reader.result;
        resolve(dataUrl.split(',')[1]); // Strip the data:audio/webm;base64, prefix
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  // Keyboard shortcut: Ctrl+Enter to solve
  container.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      const activeMode = container.querySelector('.input-mode.active')?.dataset.mode;
      if (activeMode === 'text') solveBtn.click();
      else if (activeMode === 'code') codeBtn.click();
      else if (activeMode === 'image') solveImageBtn.click();
    }
  });

  // showLoading / hideLoading retained for voice processing state
  function showLoading() {
    container.querySelectorAll('.input-panel, .input-modes, .home-footer').forEach(el => el.classList.add('hidden'));
    solveLoading.classList.remove('hidden');
  }

  function hideLoading() {
    solveLoading.classList.add('hidden');
    container.querySelectorAll('.input-panel, .input-modes, .home-footer').forEach(el => el.classList.remove('hidden'));
    // Re-show only active panel
    const activeMode = container.querySelector('.input-mode.active')?.dataset.mode || 'text';
    container.querySelectorAll('.input-panel').forEach(p => p.classList.add('hidden'));
    container.querySelector(`#${activeMode}-panel`)?.classList.remove('hidden');
  }
}

function showToast(message, type = '') {
  const toast = document.createElement('div');
  toast.className = `toast ${type ? 'toast--' + type : ''}`;
  toast.innerHTML = `<span class="material-symbols-outlined">${type === 'error' ? 'error' : 'check_circle'}</span> ${message}`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}
