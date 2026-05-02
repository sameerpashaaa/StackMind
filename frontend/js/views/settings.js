/**
 * Settings View — Configuration panel
 */
export function renderSettings(container) {
  container.innerHTML = `
    <div class="page">
      <div class="page-header">
        <h1 class="page-header__title">Configuration</h1>
        <p class="page-header__subtitle">Manage core system parameters, LLM providers, and data processing rules.</p>
      </div>

      <!-- LLM Provider -->
      <div class="card" style="margin-bottom:var(--space-lg);">
        <div class="card__header">
          <h2 class="card__title"><span class="material-symbols-outlined" style="font-size:20px;vertical-align:-4px;">psychology</span> LLM Provider</h2>
        </div>
        <div class="setting-row">
          <div class="setting-row__info"><div class="setting-row__title">Provider</div><div class="setting-row__desc">Select the LLM backend</div></div>
          <div class="setting-row__control"><select class="select" style="width:180px;"><option>Mistral AI</option><option>OpenAI</option></select></div>
        </div>
        <div class="setting-row">
          <div class="setting-row__info"><div class="setting-row__title">Model</div><div class="setting-row__desc">Model identifier</div></div>
          <div class="setting-row__control"><select class="select" style="width:220px;"><option>mistral-large-latest</option><option>mistral-medium-latest</option></select></div>
        </div>
        <div class="setting-row">
          <div class="setting-row__info"><div class="setting-row__title">Temperature</div><div class="setting-row__desc">Creativity vs determinism (0.0 - 1.0)</div></div>
          <div class="setting-row__control"><input type="range" min="0" max="100" value="70" style="width:120px;" /><span class="text-label" style="margin-left:8px;">0.7</span></div>
        </div>
        <div class="setting-row">
          <div class="setting-row__info"><div class="setting-row__title">Max Tokens</div><div class="setting-row__desc">Maximum response length</div></div>
          <div class="setting-row__control"><input type="number" class="input" style="width:100px;" value="2000" /></div>
        </div>
      </div>

      <!-- Memory -->
      <div class="card" style="margin-bottom:var(--space-lg);">
        <div class="card__header">
          <h2 class="card__title"><span class="material-symbols-outlined" style="font-size:20px;vertical-align:-4px;">database</span> Memory Architecture</h2>
        </div>
        <div class="setting-row">
          <div class="setting-row__info"><div class="setting-row__title">Persistence</div><div class="setting-row__desc">Store memory across sessions using ChromaDB</div></div>
          <div class="setting-row__control"><label class="toggle"><input type="checkbox" checked /><span class="toggle__slider"></span></label></div>
        </div>
        <div class="setting-row">
          <div class="setting-row__info"><div class="setting-row__title">Max History</div><div class="setting-row__desc">Maximum entries in memory</div></div>
          <div class="setting-row__control"><input type="number" class="input" style="width:100px;" value="100" /></div>
        </div>
      </div>

      <!-- Modalities -->
      <div class="card" style="margin-bottom:var(--space-lg);">
        <div class="card__header">
          <h2 class="card__title"><span class="material-symbols-outlined" style="font-size:20px;vertical-align:-4px;">input</span> Modalities</h2>
        </div>
        <div class="setting-row">
          <div class="setting-row__info"><div class="setting-row__title">Text Input</div><div class="setting-row__desc">Enable text-based problem input</div></div>
          <div class="setting-row__control"><label class="toggle"><input type="checkbox" checked /><span class="toggle__slider"></span></label></div>
        </div>
        <div class="setting-row">
          <div class="setting-row__info"><div class="setting-row__title">Image Input</div><div class="setting-row__desc">Enable image analysis via OCR</div></div>
          <div class="setting-row__control"><label class="toggle"><input type="checkbox" checked /><span class="toggle__slider"></span></label></div>
        </div>
        <div class="setting-row">
          <div class="setting-row__info"><div class="setting-row__title">Voice Input</div><div class="setting-row__desc">Enable speech-to-text transcription</div></div>
          <div class="setting-row__control"><label class="toggle"><input type="checkbox" checked /><span class="toggle__slider"></span></label></div>
        </div>
        <div class="setting-row">
          <div class="setting-row__info"><div class="setting-row__title">Code Input</div><div class="setting-row__desc">Enable code parsing and analysis</div></div>
          <div class="setting-row__control"><label class="toggle"><input type="checkbox" checked /><span class="toggle__slider"></span></label></div>
        </div>
      </div>

      <!-- Privacy -->
      <div class="card">
        <div class="card__header">
          <h2 class="card__title"><span class="material-symbols-outlined" style="font-size:20px;vertical-align:-4px;">shield</span> Privacy</h2>
        </div>
        <div class="setting-row">
          <div class="setting-row__info"><div class="setting-row__title">Store Conversations</div><div class="setting-row__desc">Keep conversation history locally</div></div>
          <div class="setting-row__control"><label class="toggle"><input type="checkbox" checked /><span class="toggle__slider"></span></label></div>
        </div>
        <div class="setting-row">
          <div class="setting-row__info"><div class="setting-row__title">Anonymize Data</div><div class="setting-row__desc">Strip personal identifiers before processing</div></div>
          <div class="setting-row__control"><label class="toggle"><input type="checkbox" checked /><span class="toggle__slider"></span></label></div>
        </div>
      </div>
    </div>
  `;
}
