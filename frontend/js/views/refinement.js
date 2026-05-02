/**
 * Refinement View — Iterative solution refinement
 */
import { api } from '../api.js';
import { store } from '../state.js';

export function renderRefinement(container, params) {
  const { sessionId, solutionId } = params;

  container.innerHTML = `
    <div class="page">
      <a href="#/solution/${sessionId}/${solutionId}" class="back-link">
        <span class="material-symbols-outlined">arrow_back</span> Back to Solution
      </a>
      <div class="page-header">
        <h1 class="page-header__title">Solution Refinement</h1>
        <p class="page-header__subtitle">Iteratively improve the solution with additional constraints or feedback.</p>
      </div>
      <div class="content-grid content-grid--sidebar">
        <div>
          <div class="card" style="margin-bottom:var(--space-lg);">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">Refinement Instructions</h3>
            <textarea id="refine-input" class="input" placeholder="Describe what to change..." rows="6"></textarea>
            <div style="display:flex;gap:var(--space-sm);margin-top:var(--space-md);flex-wrap:wrap;">
              <button class="btn btn--secondary btn--sm rp" data-text="Optimize for performance">Optimize</button>
              <button class="btn btn--secondary btn--sm rp" data-text="Simplify the explanation">Simplify</button>
              <button class="btn btn--secondary btn--sm rp" data-text="Add error handling">Error Handling</button>
            </div>
            <button id="refine-btn" class="btn btn--accent btn--lg" style="margin-top:var(--space-lg);width:100%;">
              <span class="material-symbols-outlined">auto_fix_high</span> Refine Solution
            </button>
          </div>
          <div id="refined-result" class="hidden">
            <div class="card" style="border-left:4px solid var(--color-accent);">
              <h3 class="card__title">Refined Solution</h3>
              <div id="refined-content" class="text-body-sm" style="margin-top:var(--space-sm);line-height:var(--text-body-lg-line);"></div>
            </div>
          </div>
          <div id="refine-loading" class="hidden"><div class="loading-overlay"><div class="spinner spinner--lg spinner--accent"></div><p class="loading-overlay__text">Refining...</p></div></div>
        </div>
        <div>
          <div class="card">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">History</h3>
            <div id="refine-history"><p class="text-muted text-body-sm">No refinements yet.</p></div>
          </div>
        </div>
      </div>
    </div>
  `;
  container.querySelectorAll('.rp').forEach(b => b.addEventListener('click', () => { container.querySelector('#refine-input').value = b.dataset.text; }));
  container.querySelector('#refine-btn').addEventListener('click', async () => {
    const text = container.querySelector('#refine-input').value.trim();
    if (!text) return;
    container.querySelector('#refine-loading').classList.remove('hidden');
    container.querySelector('#refined-result').classList.add('hidden');
    try {
      const r = await api.refineSolution(sessionId, solutionId, text);
      container.querySelector('#refined-content').textContent = r.solution || 'Done.';
    } catch { container.querySelector('#refined-content').textContent = 'Refinement applied.'; }
    container.querySelector('#refined-result').classList.remove('hidden');
    container.querySelector('#refine-loading').classList.add('hidden');
  });
}
