/**
 * Alternatives View — Side-by-side solution comparison
 */
import { store } from '../state.js';

export function renderAlternatives(container, params) {
  const { sessionId, solutionId } = params;
  const data = store.getCachedSolution(solutionId) || {};

  const alternatives = data.alternative_solutions || [
    {
      method: 'Lagrangian Method (Primary)',
      description: 'Classic constrained optimization using Lagrange multipliers. Provides exact analytical solution with full mathematical rigor.',
      confidence: 0.94,
      pros: ['Exact solution', 'Mathematically rigorous', 'Well-understood convergence'],
      cons: ['Complex for high-dimensional problems', 'Requires constraint qualification'],
      performance: { accuracy: 98, speed: 85, memory: 72 },
    },
    {
      method: 'Substitution Method',
      description: 'Direct variable elimination by substituting the constraint into the objective function, reducing to unconstrained optimization.',
      confidence: 0.91,
      pros: ['Simpler implementation', 'No multiplier needed', 'Faster for 2 variables'],
      cons: ['Doesn\'t generalize to inequalities', 'Manual algebra required'],
      performance: { accuracy: 95, speed: 92, memory: 90 },
    },
    {
      method: 'Geometric Interpretation',
      description: 'Visual approach: the minimum of x² + y² on the line x + y = 1 is the point closest to the origin. Uses projection.',
      confidence: 0.88,
      pros: ['Intuitive understanding', 'Great for teaching', 'Visual verification'],
      cons: ['Limited to 2D/3D', 'Not computationally efficient'],
      performance: { accuracy: 92, speed: 78, memory: 95 },
    },
  ];

  container.innerHTML = `
    <div class="page">
      <a href="#/solution/${sessionId}/${solutionId}" class="back-link">
        <span class="material-symbols-outlined">arrow_back</span>
        Back to Solution
      </a>

      <div class="page-header">
        <h1 class="page-header__title">Alternative Solutions</h1>
        <p class="page-header__subtitle">Comparative analysis of different solution approaches with trade-off metrics.</p>
      </div>

      <div class="content-grid content-grid--3col">
        ${alternatives.map((alt, i) => `
          <div class="card ${i === 0 ? 'card--recommended' : ''}" style="${i === 0 ? 'border-color: var(--color-accent); border-width: 2px;' : ''}">
            ${i === 0 ? '<span class="chip chip--accent" style="margin-bottom: var(--space-sm);">Recommended</span>' : ''}
            <h3 class="card__title">${alt.method}</h3>
            <p class="text-body-sm text-muted" style="margin: var(--space-sm) 0 var(--space-md);">
              ${alt.description}
            </p>

            <!-- Confidence -->
            <div style="margin-bottom: var(--space-md);">
              <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span class="text-label">Confidence</span>
                <span class="text-label" style="color: var(--color-on-surface);">${Math.round((alt.confidence || 0) * 100)}%</span>
              </div>
              <div class="progress">
                <div class="progress__bar ${(alt.confidence || 0) > 0.9 ? 'progress__bar--success' : ''}" style="width: ${(alt.confidence || 0) * 100}%;"></div>
              </div>
            </div>

            <!-- Performance Metrics -->
            ${alt.performance ? `
              <div style="margin-bottom: var(--space-md);">
                <span class="text-label" style="display: block; margin-bottom: var(--space-sm);">Performance</span>
                <div style="display: flex; flex-direction: column; gap: 6px;">
                  <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="text-body-sm text-muted">Accuracy</span>
                    <div class="progress" style="width: 100px;">
                      <div class="progress__bar progress__bar--success" style="width: ${alt.performance.accuracy}%;"></div>
                    </div>
                  </div>
                  <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="text-body-sm text-muted">Speed</span>
                    <div class="progress" style="width: 100px;">
                      <div class="progress__bar" style="width: ${alt.performance.speed}%;"></div>
                    </div>
                  </div>
                  <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="text-body-sm text-muted">Memory</span>
                    <div class="progress" style="width: 100px;">
                      <div class="progress__bar" style="width: ${alt.performance.memory}%;"></div>
                    </div>
                  </div>
                </div>
              </div>
            ` : ''}

            <!-- Pros -->
            ${alt.pros ? `
              <div style="margin-bottom: var(--space-sm);">
                ${alt.pros.map(p => `
                  <div style="display: flex; align-items: center; gap: 6px; padding: 4px 0;">
                    <span class="material-symbols-outlined" style="font-size: 16px; color: var(--color-success);">add_circle</span>
                    <span class="text-body-sm">${p}</span>
                  </div>
                `).join('')}
              </div>
            ` : ''}

            <!-- Cons -->
            ${alt.cons ? `
              <div>
                ${alt.cons.map(c => `
                  <div style="display: flex; align-items: center; gap: 6px; padding: 4px 0;">
                    <span class="material-symbols-outlined" style="font-size: 16px; color: var(--color-on-surface-muted);">remove_circle_outline</span>
                    <span class="text-body-sm text-muted">${c}</span>
                  </div>
                `).join('')}
              </div>
            ` : ''}
          </div>
        `).join('')}
      </div>
    </div>
  `;
}
