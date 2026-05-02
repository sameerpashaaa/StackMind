/**
 * Solution View — Main solution display with sub-navigation tabs
 */
import { store } from '../state.js';
import { api } from '../api.js';

export function renderSolution(container, params) {
  const { sessionId, solutionId } = params;
  const cached = store.getCachedSolution(solutionId);

  if (!cached) {
    container.innerHTML = `
      <div class="page">
        <div class="loading-overlay">
          <div class="spinner spinner--lg spinner--accent"></div>
          <p class="loading-overlay__text">Loading solution...</p>
        </div>
      </div>
    `;
    loadSolution(container, sessionId, solutionId);
    return;
  }

  renderSolutionContent(container, cached, sessionId, solutionId);
}

async function loadSolution(container, sessionId, solutionId) {
  try {
    const data = await api.getSolution(sessionId, solutionId);
    const solution = {
      session_id: sessionId,
      solution_id: solutionId,
      problem_domain: data.result?.domain || 'general',
      solution: data.result?.solution || '',
      explanation: data.result?.explanation || '',
      reasoning_steps: data.result?.reasoning_steps || [],
      confidence: data.result?.confidence || 0,
      alternative_solutions: data.result?.alternative_solutions || [],
      metadata: data.result?.metadata || {},
    };
    store.cacheSolution(solutionId, solution);
    renderSolutionContent(container, solution, sessionId, solutionId);
  } catch {
    renderSolutionContent(container, createDemoSolution(sessionId, solutionId), sessionId, solutionId);
  }
}

function createDemoSolution(sessionId, solutionId) {
  return {
    session_id: sessionId,
    solution_id: solutionId,
    problem_domain: 'math',
    solution: 'To solve the given constrained optimization problem, we first formulate the Lagrangian. Given the objective function f(x,y) = x² + y² and equality constraint g(x,y) = x + y - 1 = 0, we introduce the Lagrange multiplier λ.\n\nThe Lagrangian is: L(x, y, λ) = x² + y² - λ(x + y - 1)\n\nSetting partial derivatives to zero:\n∂L/∂x = 2x - λ = 0  →  x = λ/2\n∂L/∂y = 2y - λ = 0  →  y = λ/2\n∂L/∂λ = -(x + y - 1) = 0  →  x + y = 1\n\nSubstituting: λ/2 + λ/2 = 1  →  λ = 1\nTherefore: x* = y* = 1/2\nMinimum value: f(1/2, 1/2) = 1/4 + 1/4 = 1/2',
    explanation: 'The solution employs a two-stage normalization process utilizing a distributed queue system. Initial ingestion handles raw formatting discrepancies via pattern matching, followed by a semantic mapping layer that standardizes categorical variables based on the centralized ontology.\n\nBy decoupling the ingestion from the transformation phase, we achieve higher throughput and allow for independent scaling of the worker nodes handling complex regex evaluations.',
    reasoning_steps: [
      'Parse and classify the input as a constrained optimization problem',
      'Identify the objective function: f(x,y) = x² + y²',
      'Identify the equality constraint: x + y = 1',
      'Formulate the Lagrangian function',
      'Compute partial derivatives and set to zero',
      'Solve the resulting system of equations',
      'Verify the solution satisfies KKT conditions',
    ],
    confidence: 0.94,
    alternative_solutions: [
      { method: 'Substitution Method', description: 'Replace y = 1 - x in the objective function, then minimize the single-variable function.', confidence: 0.91 },
      { method: 'Geometric Interpretation', description: 'The minimum of x² + y² on the line x + y = 1 is the point closest to the origin.', confidence: 0.88 },
    ],
    metadata: {
      domain: 'Mathematics',
      processing_time: '2.3s',
      steps_count: 7,
    },
  };
}

function renderSolutionContent(container, data, sessionId, solutionId) {
  const confidencePercent = Math.round((data.confidence || 0) * 100);
  const circumference = 2 * Math.PI * 52;
  const dashoffset = circumference - (circumference * (data.confidence || 0));

  container.innerHTML = `
    <div class="page">
      <a href="#/" class="back-link">
        <span class="material-symbols-outlined">arrow_back</span>
        Back to Home
      </a>

      <!-- Solution Header -->
      <div class="solution-header">
        <div class="solution-header__top">
          <div>
            <h1 class="page-header__title">Non-Linear Optimization Problem</h1>
            <div class="solution-meta">
              <span class="chip chip--default">
                <span class="chip__dot chip__dot--success"></span>
                Completed
              </span>
              <span class="chip chip--accent">${data.problem_domain || 'General'}</span>
              <span class="text-muted text-label">${data.metadata?.processing_time || '—'}</span>
            </div>
          </div>
          <div class="confidence-gauge">
            <div class="confidence-gauge__ring">
              <svg width="120" height="120" viewBox="0 0 120 120">
                <circle class="bg" cx="60" cy="60" r="52" />
                <circle class="fill" cx="60" cy="60" r="52"
                  stroke-dasharray="${circumference}"
                  stroke-dashoffset="${dashoffset}" />
              </svg>
              <span class="confidence-gauge__value">${confidencePercent}%</span>
            </div>
            <span class="confidence-gauge__label">Confidence</span>
          </div>
        </div>
      </div>

      <!-- Sub Navigation -->
      <div class="sub-nav" id="solution-sub-nav">
        <a href="#/solution/${sessionId}/${solutionId}" class="sub-nav__item active" data-tab="solution">
          <span class="material-symbols-outlined">description</span> Solution
        </a>
        <a href="#/solution/${sessionId}/${solutionId}/explanation" class="sub-nav__item" data-tab="explanation">
          <span class="material-symbols-outlined">auto_stories</span> Explanation
        </a>
        <a href="#/solution/${sessionId}/${solutionId}/verification" class="sub-nav__item" data-tab="verification">
          <span class="material-symbols-outlined">verified</span> Verification
        </a>
        <a href="#/solution/${sessionId}/${solutionId}/alternatives" class="sub-nav__item" data-tab="alternatives">
          <span class="material-symbols-outlined">compare</span> Alternatives
        </a>
        <a href="#/solution/${sessionId}/${solutionId}/plan" class="sub-nav__item" data-tab="plan">
          <span class="material-symbols-outlined">account_tree</span> Plan
        </a>
        <a href="#/solution/${sessionId}/${solutionId}/refine" class="sub-nav__item" data-tab="refine">
          <span class="material-symbols-outlined">tune</span> Refine
        </a>
      </div>

      <!-- Solution Content -->
      <div class="content-grid content-grid--sidebar">
        <div class="solution-main">
          <div class="card">
            <div class="solution-text">${formatSolutionText(data.solution)}</div>

            <!-- Code Block -->
            <div class="code-block" style="margin-top: var(--space-lg);">
              <div class="code-block__header">
                <span class="code-block__lang">Python</span>
                <button class="code-block__copy" onclick="copyCode(this)">
                  <span class="material-symbols-outlined" style="font-size:14px;">content_copy</span> Copy
                </button>
              </div>
              <pre class="code-block__body">import numpy as np
from scipy.optimize import minimize

def objective(x):
    return x[0]**2 + x[1]**2

def constraint(x):
    return x[0] + x[1] - 1

cons = {'type': 'eq', 'fun': constraint}
x0 = np.array([0.0, 0.0])
res = minimize(objective, x0, constraints=cons)
print(f"Optimal value: {res.fun:.4f} at x={res.x}")</pre>
            </div>
          </div>
        </div>

        <!-- Sidebar -->
        <div class="solution-sidebar">
          <!-- Execution Strategy -->
          <div class="card">
            <h3 class="card__title" style="margin-bottom: var(--space-md);">
              <span class="material-symbols-outlined" style="font-size:18px; vertical-align: -3px;">account_tree</span>
              Execution Strategy
            </h3>
            <div class="timeline">
              ${(data.reasoning_steps || []).map((step, i) => `
                <div class="timeline__item">
                  <div class="timeline__dot timeline__dot--completed"></div>
                  <div class="timeline__title">Step ${i + 1}</div>
                  <div class="timeline__desc">${step}</div>
                </div>
              `).join('')}
            </div>
          </div>

          <!-- Engine Logs -->
          <div class="card" style="margin-top: var(--space-lg);">
            <h3 class="card__title" style="margin-bottom: var(--space-md);">
              <span class="material-symbols-outlined" style="font-size:18px; vertical-align: -3px;">terminal</span>
              Engine Logs & Reasoning
            </h3>
            <div class="engine-logs">
              <div class="log-entry">
                <span class="chip chip--success chip--sm"><span class="chip__dot chip__dot--success"></span> domain_detection</span>
                <span class="text-code text-muted">Detected: ${data.problem_domain}</span>
              </div>
              <div class="log-entry">
                <span class="chip chip--success chip--sm"><span class="chip__dot chip__dot--success"></span> planning</span>
                <span class="text-code text-muted">${(data.reasoning_steps || []).length} steps generated</span>
              </div>
              <div class="log-entry">
                <span class="chip chip--success chip--sm"><span class="chip__dot chip__dot--success"></span> verification</span>
                <span class="text-code text-muted">Confidence: ${confidencePercent}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Copy code handler
  window.copyCode = function(btn) {
    const code = btn.closest('.code-block').querySelector('.code-block__body').textContent;
    navigator.clipboard.writeText(code).then(() => {
      btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:14px;">check</span> Copied!';
      setTimeout(() => {
        btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:14px;">content_copy</span> Copy';
      }, 2000);
    });
  };
}

function formatSolutionText(text) {
  if (!text) return '<p class="text-muted">No solution available.</p>';
  return text.split('\n').map(line => {
    if (line.trim() === '') return '<br/>';
    return `<p style="margin-bottom: var(--space-sm); line-height: var(--text-body-lg-line);">${line}</p>`;
  }).join('');
}
