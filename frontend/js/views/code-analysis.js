/**
 * Code Analysis View
 */
export function renderCodeAnalysis(container, params) {
  const { sessionId, solutionId } = params;
  container.innerHTML = `
    <div class="page">
      <a href="#/solution/${sessionId}/${solutionId}" class="back-link">
        <span class="material-symbols-outlined">arrow_back</span> Back to Solution
      </a>
      <div class="page-header">
        <h1 class="page-header__title">Code Analysis</h1>
        <p class="page-header__subtitle">Detailed code inspection with issue detection and suggestions.</p>
      </div>
      <div class="content-grid content-grid--sidebar">
        <div>
          <div class="card">
            <div class="code-block" style="border:none;">
              <div class="code-block__header">
                <span class="code-block__lang">Python</span>
                <div style="display:flex;gap:var(--space-sm);">
                  <span class="chip chip--success"><span class="chip__dot chip__dot--success"></span> No Errors</span>
                  <span class="chip chip--warning">2 Warnings</span>
                </div>
              </div>
              <pre class="code-block__body"><span style="color:#71717A;">1  </span>import numpy as np
<span style="color:#71717A;">2  </span>from scipy.optimize import minimize
<span style="color:#71717A;">3  </span>
<span style="color:#71717A;">4  </span>def objective(x):
<span style="color:#71717A;">5  </span>    return x[0]**2 + x[1]**2
<span style="color:#71717A;">6  </span>
<span style="color:#71717A;">7  </span>def constraint(x):
<span style="color:#71717A;">8  </span>    return x[0] + x[1] - 1
<span style="color:#71717A;">9  </span>
<span style="color:#71717A;">10 </span>cons = {'type': 'eq', 'fun': constraint}
<span style="color:#71717A;">11 </span>x0 = np.array([0.0, 0.0])
<span style="color:#71717A;">12 </span>res = minimize(objective, x0, constraints=cons)
<span style="color:#71717A;">13 </span>print(f"Optimal: {res.fun:.4f} at x={res.x}")</pre>
            </div>
          </div>
        </div>
        <div>
          <div class="card" style="margin-bottom:var(--space-lg);">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">Analysis Summary</h3>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:var(--space-md);">
              <div class="metric"><div class="metric__value" style="color:var(--color-success);">0</div><div class="metric__label">Errors</div></div>
              <div class="metric"><div class="metric__value" style="color:var(--color-warning);">2</div><div class="metric__label">Warnings</div></div>
              <div class="metric"><div class="metric__value">13</div><div class="metric__label">Lines</div></div>
              <div class="metric"><div class="metric__value">A</div><div class="metric__label">Quality</div></div>
            </div>
          </div>
          <div class="card" style="margin-bottom:var(--space-lg);">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">Issues</h3>
            <div style="display:flex;flex-direction:column;gap:var(--space-sm);">
              <div class="report-card">
                <div class="report-card__icon report-card__icon--warn"><span class="material-symbols-outlined">warning</span></div>
                <div class="report-card__body">
                  <div class="report-card__title">Missing docstrings</div>
                  <div class="report-card__desc">Functions lack documentation. Lines 4, 7.</div>
                </div>
              </div>
              <div class="report-card">
                <div class="report-card__icon report-card__icon--warn"><span class="material-symbols-outlined">info</span></div>
                <div class="report-card__body">
                  <div class="report-card__title">No type hints</div>
                  <div class="report-card__desc">Consider adding type annotations for better IDE support.</div>
                </div>
              </div>
            </div>
          </div>
          <div class="card">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">Detected</h3>
            <div style="display:flex;flex-direction:column;gap:var(--space-xs);">
              <div class="setting-row"><div class="setting-row__info"><div class="setting-row__title">Language</div></div><span class="chip chip--default">Python 3</span></div>
              <div class="setting-row"><div class="setting-row__info"><div class="setting-row__title">Dependencies</div></div><span class="text-body-sm">numpy, scipy</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}
