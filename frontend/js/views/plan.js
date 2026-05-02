/**
 * Plan Visualization View — Execution plan timeline
 */
import { store } from '../state.js';

export function renderPlan(container, params) {
  const { sessionId, solutionId } = params;
  const data = store.getCachedSolution(solutionId) || {};
  const steps = data.reasoning_steps || [
    'Parse and classify the input as a constrained optimization problem',
    'Identify the objective function: f(x,y) = x² + y²',
    'Identify the equality constraint: x + y = 1',
    'Formulate the Lagrangian function',
    'Compute partial derivatives and set to zero',
    'Solve the resulting system of equations',
    'Verify the solution satisfies KKT conditions',
  ];

  container.innerHTML = `
    <div class="page">
      <a href="#/solution/${sessionId}/${solutionId}" class="back-link">
        <span class="material-symbols-outlined">arrow_back</span> Back to Solution
      </a>
      <div class="page-header">
        <h1 class="page-header__title">Execution Plan</h1>
        <p class="page-header__subtitle">Multi-step reasoning pipeline with dependency tracking.</p>
      </div>
      <div class="content-grid content-grid--sidebar">
        <div class="card">
          <div class="timeline">
            ${steps.map((step, i) => `
              <div class="timeline__item">
                <div class="timeline__dot timeline__dot--completed"></div>
                <div class="timeline__title">Step ${i + 1}: ${getStepLabel(i)}</div>
                <div class="timeline__desc">${step}</div>
                <div class="timeline__meta">
                  <span class="chip chip--success chip--sm"><span class="chip__dot chip__dot--success"></span> Completed</span>
                  <span>${getStepDuration(i)}</span>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
        <div>
          <div class="card" style="margin-bottom:var(--space-lg);">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">Summary</h3>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:var(--space-md);">
              <div class="metric"><div class="metric__value">${steps.length}</div><div class="metric__label">Steps</div></div>
              <div class="metric"><div class="metric__value">2.3s</div><div class="metric__label">Total Time</div></div>
              <div class="metric"><div class="metric__value">0</div><div class="metric__label">Failures</div></div>
              <div class="metric"><div class="metric__value">1</div><div class="metric__label">Retries</div></div>
            </div>
          </div>
          <div class="card">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">Dependencies</h3>
            <div style="display:flex;flex-direction:column;gap:var(--space-xs);">
              ${steps.slice(1).map((_, i) => `
                <div class="text-code text-muted" style="font-size:12px;">Step ${i + 2} ← Step ${i + 1}</div>
              `).join('')}
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function getStepLabel(i) {
  const labels = ['Input Analysis', 'Objective Identification', 'Constraint Extraction', 'Formulation', 'Differentiation', 'Solving', 'Verification'];
  return labels[i] || `Phase ${i + 1}`;
}
function getStepDuration(i) {
  const durations = ['0.12s', '0.08s', '0.05s', '0.31s', '0.44s', '0.89s', '0.41s'];
  return durations[i] || '0.1s';
}
