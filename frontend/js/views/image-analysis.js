/**
 * Image Analysis View
 */
export function renderImageAnalysis(container, params) {
  const { sessionId, solutionId } = params;
  container.innerHTML = `
    <div class="page">
      <a href="#/solution/${sessionId}/${solutionId}" class="back-link">
        <span class="material-symbols-outlined">arrow_back</span> Back to Solution
      </a>
      <div class="page-header">
        <h1 class="page-header__title">Image Analysis</h1>
        <p class="page-header__subtitle">Visual content extraction and problem identification.</p>
      </div>
      <div class="content-grid content-grid--sidebar">
        <div>
          <div class="card" style="margin-bottom:var(--space-lg);">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">Input Image</h3>
            <div style="background:var(--color-surface-container);border-radius:var(--radius-lg);padding:var(--space-xl);text-align:center;min-height:200px;display:flex;align-items:center;justify-content:center;">
              <div>
                <span class="material-symbols-outlined" style="font-size:48px;color:var(--color-on-surface-muted);">image</span>
                <p class="text-muted" style="margin-top:var(--space-sm);">Uploaded image preview</p>
              </div>
            </div>
          </div>
          <div class="card">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">Extracted Content</h3>
            <div class="code-block">
              <div class="code-block__header"><span class="code-block__lang">OCR Output</span></div>
              <pre class="code-block__body">Detected text and mathematical expressions
from the uploaded image will appear here.

f(x,y) = x² + y²
subject to: x + y = 1</pre>
            </div>
          </div>
        </div>
        <div>
          <div class="card" style="margin-bottom:var(--space-lg);">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">Analysis Metadata</h3>
            <div style="display:flex;flex-direction:column;gap:var(--space-sm);">
              <div class="setting-row"><div class="setting-row__info"><div class="setting-row__title">Format</div></div><span class="text-body-sm">PNG, 1920×1080</span></div>
              <div class="setting-row"><div class="setting-row__info"><div class="setting-row__title">OCR Engine</div></div><span class="text-body-sm">Tesseract v5</span></div>
              <div class="setting-row"><div class="setting-row__info"><div class="setting-row__title">Confidence</div></div><span class="text-body-sm">96%</span></div>
              <div class="setting-row"><div class="setting-row__info"><div class="setting-row__title">Detected Domain</div></div><span class="chip chip--accent">Mathematics</span></div>
            </div>
          </div>
          <div class="card">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">Identified Elements</h3>
            <div style="display:flex;flex-direction:column;gap:var(--space-xs);">
              <div class="insight-item"><span class="material-symbols-outlined">functions</span><span class="insight-item__text">Mathematical expression detected</span></div>
              <div class="insight-item"><span class="material-symbols-outlined">text_fields</span><span class="insight-item__text">Constraint notation identified</span></div>
              <div class="insight-item"><span class="material-symbols-outlined">grid_on</span><span class="insight-item__text">Graph/chart detected</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}
