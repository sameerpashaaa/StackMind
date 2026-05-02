/**
 * Verification View — Self-verification results with confidence gauge
 */
import { store } from '../state.js';

export function renderVerification(container, params) {
  const { sessionId, solutionId } = params;
  const data = store.getCachedSolution(solutionId) || {};
  const confidence = data.confidence || 0.94;
  const confidencePercent = Math.round(confidence * 100);
  const circumference = 2 * Math.PI * 52;
  const dashoffset = circumference - (circumference * confidence);

  container.innerHTML = `
    <div class="page">
      <a href="#/solution/${sessionId}/${solutionId}" class="back-link">
        <span class="material-symbols-outlined">arrow_back</span>
        Back to Solution
      </a>

      <div class="page-header">
        <h1 class="page-header__title">Verification Results</h1>
        <p class="page-header__subtitle">Comprehensive analysis of the proposed topological optimization model. All constraints satisfied.</p>
      </div>

      <!-- Verification Status -->
      <div class="card" style="margin-bottom: var(--space-lg); border-left: 4px solid var(--color-success);">
        <div style="display: flex; align-items: center; gap: var(--space-lg);">
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
          <div style="flex: 1;">
            <div style="display: flex; align-items: center; gap: var(--space-sm); margin-bottom: var(--space-sm);">
              <span class="material-symbols-outlined" style="color: var(--color-success); font-size: 24px;">verified</span>
              <h2 class="text-h2">Verified</h2>
            </div>
            <p class="text-body-sm text-muted">
              The reasoning engine has confirmed the mathematical integrity and logical consistency of the current path.
            </p>
          </div>
        </div>
      </div>

      <!-- Detailed Report -->
      <div class="card" style="margin-bottom: var(--space-lg);">
        <h2 class="card__title" style="margin-bottom: var(--space-lg);">Detailed Report</h2>
        <div style="display: flex; flex-direction: column; gap: var(--space-md);">
          <div class="report-card">
            <div class="report-card__icon report-card__icon--pass">
              <span class="material-symbols-outlined">check</span>
            </div>
            <div class="report-card__body">
              <div class="report-card__title">Mathematical Integrity</div>
              <div class="report-card__desc">All formal proofs aligned with the initial constraints. The structural integrity formulas yield expected results within the defined tolerance bounds.</div>
              <div class="progress" style="margin-top: var(--space-sm);">
                <div class="progress__bar progress__bar--success" style="width: 100%;"></div>
              </div>
            </div>
          </div>

          <div class="report-card">
            <div class="report-card__icon report-card__icon--pass">
              <span class="material-symbols-outlined">check</span>
            </div>
            <div class="report-card__body">
              <div class="report-card__title">Numerical Stability</div>
              <div class="report-card__desc">Matrix inversions and tensor operations completed without floating-point overflow. Convergence achieved in 43 iterations.</div>
              <div class="progress" style="margin-top: var(--space-sm);">
                <div class="progress__bar progress__bar--success" style="width: 96%;"></div>
              </div>
            </div>
          </div>

          <div class="report-card">
            <div class="report-card__icon report-card__icon--pass">
              <span class="material-symbols-outlined">check</span>
            </div>
            <div class="report-card__body">
              <div class="report-card__title">Explanation Clarity</div>
              <div class="report-card__desc">The generated explanation paths are highly legible. Minimal cyclomatic complexity detected in the reasoning branches.</div>
              <div class="progress" style="margin-top: var(--space-sm);">
                <div class="progress__bar progress__bar--success" style="width: 94%;"></div>
              </div>
            </div>
          </div>

          <div class="report-card">
            <div class="report-card__icon report-card__icon--warn">
              <span class="material-symbols-outlined">info</span>
            </div>
            <div class="report-card__body">
              <div class="report-card__title">Resource Efficiency</div>
              <div class="report-card__desc">While perfectly correct, the current computational path utilizes 14% more memory bandwidth than the theoretical minimum. Suggest exploring Alternative branch B for hardware acceleration scenarios.</div>
              <div class="progress" style="margin-top: var(--space-sm);">
                <div class="progress__bar progress__bar--warning" style="width: 86%;"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}
