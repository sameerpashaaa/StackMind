/**
 * Explanation View — Detailed breakdown
 */
import { store } from '../state.js';

export function renderExplanation(container, params) {
  const { sessionId, solutionId } = params;
  const data = store.getCachedSolution(solutionId) || getDemoData();

  container.innerHTML = `
    <div class="page">
      <a href="#/solution/${sessionId}/${solutionId}" class="back-link">
        <span class="material-symbols-outlined">arrow_back</span>
        Back to Solution
      </a>

      <div class="page-header">
        <p class="text-muted" style="margin-bottom: var(--space-xs);">Detailed Explanation</p>
        <h1 class="page-header__title">Comprehensive Breakdown</h1>
        <p class="page-header__subtitle">Comprehensive breakdown of the proposed data ingestion and normalization strategy, highlighting algorithmic choices and performance implications.</p>
      </div>

      <!-- Approach Summary -->
      <div class="card" style="margin-bottom: var(--space-lg);">
        <h2 class="card__title">Approach Summary</h2>
        <p class="text-body-sm text-muted" style="margin-top: var(--space-sm); line-height: var(--text-body-lg-line);">
          ${data.explanation || 'The solution employs a two-stage normalization process utilizing a distributed queue system. Initial ingestion handles raw formatting discrepancies via pattern matching, followed by a semantic mapping layer that standardizes categorical variables based on the centralized ontology.'}
        </p>
        <p class="text-body-sm text-muted" style="margin-top: var(--space-sm); line-height: var(--text-body-lg-line);">
          By decoupling the ingestion from the transformation phase, we achieve higher throughput and allow for independent scaling of the worker nodes handling complex regex evaluations.
        </p>
      </div>

      <!-- Key Insights -->
      <div class="card" style="margin-bottom: var(--space-lg);">
        <h2 class="card__title" style="margin-bottom: var(--space-md);">Key Insights</h2>
        <div style="display: flex; flex-direction: column; gap: var(--space-sm);">
          <div class="insight-item">
            <span class="material-symbols-outlined">check_circle</span>
            <span class="insight-item__text">Regex optimization reduces processing time by 40%.</span>
          </div>
          <div class="insight-item">
            <span class="material-symbols-outlined">check_circle</span>
            <span class="insight-item__text">Memory footprint stabilized under heavy concurrent load.</span>
          </div>
          <div class="insight-item">
            <span class="material-symbols-outlined">check_circle</span>
            <span class="insight-item__text">Semantic mapping coverage increased to 98.5%.</span>
          </div>
        </div>
      </div>

      <div class="content-grid content-grid--2col">
        <!-- Asynchronous Queue Management -->
        <div class="card">
          <h3 class="card__title">Asynchronous Queue Management</h3>
          <p class="text-body-sm text-muted" style="margin: var(--space-sm) 0 var(--space-md);">
            Handling backpressure during peak ingestion bursts requires dynamic scaling of worker threads.
          </p>
          <div class="code-block">
            <div class="code-block__header">
              <span class="code-block__lang">Python</span>
              <button class="code-block__copy" onclick="copyCode(this)">
                <span class="material-symbols-outlined" style="font-size:14px;">content_copy</span> Copy
              </button>
            </div>
            <pre class="code-block__body">def handle_ingestion_burst(queue_metrics):
    if queue_metrics.backlog > THRESHOLD:
        scale_workers(
            target=calculate_optimal_workers(
                queue_metrics
            )
        )
    return status.OK</pre>
          </div>
        </div>

        <!-- Semantic Resolution Fallback -->
        <div class="card">
          <h3 class="card__title">Semantic Resolution Fallback</h3>
          <p class="text-body-sm text-muted" style="margin: var(--space-sm) 0 var(--space-md);">
            When the primary ontology fails to match a categorical variable, a secondary fuzzy matching algorithm is deployed before flagging for manual review.
          </p>
          <div class="code-block">
            <div class="code-block__header">
              <span class="code-block__lang">Python</span>
              <button class="code-block__copy" onclick="copyCode(this)">
                <span class="material-symbols-outlined" style="font-size:14px;">content_copy</span> Copy
              </button>
            </div>
            <pre class="code-block__body">def resolve_semantic(variable, ontology):
    match = ontology.lookup(variable)
    if not match:
        match = fuzzy_match(
            variable,
            ontology.categories,
            threshold=0.85
        )
    return match or flag_for_review(variable)</pre>
          </div>
        </div>
      </div>

      <!-- Strengths & Limitations -->
      <div class="content-grid content-grid--2col" style="margin-top: var(--space-lg);">
        <div class="card">
          <h2 class="card__title" style="margin-bottom: var(--space-md);">
            <span class="material-symbols-outlined" style="color: var(--color-success); font-size: 20px; vertical-align: -4px;">thumb_up</span>
            Strengths
          </h2>
          <ul style="display: flex; flex-direction: column; gap: var(--space-sm);">
            <li class="insight-item">
              <span class="material-symbols-outlined" style="color: var(--color-success);">check</span>
              <span class="insight-item__text">Highly scalable architecture suitable for fluctuating data volumes.</span>
            </li>
            <li class="insight-item">
              <span class="material-symbols-outlined" style="color: var(--color-success);">check</span>
              <span class="insight-item__text">Clear separation of concerns simplifies testing and deployment.</span>
            </li>
            <li class="insight-item">
              <span class="material-symbols-outlined" style="color: var(--color-success);">check</span>
              <span class="insight-item__text">Robust error handling ensures no data loss during processing failures.</span>
            </li>
          </ul>
        </div>

        <div class="card">
          <h2 class="card__title" style="margin-bottom: var(--space-md);">
            <span class="material-symbols-outlined" style="color: var(--color-warning); font-size: 20px; vertical-align: -4px;">warning</span>
            Limitations
          </h2>
          <ul style="display: flex; flex-direction: column; gap: var(--space-sm);">
            <li class="insight-item">
              <span class="material-symbols-outlined" style="color: var(--color-warning);">remove_circle_outline</span>
              <span class="insight-item__text">Increased infrastructure complexity compared to monolithic approaches.</span>
            </li>
            <li class="insight-item">
              <span class="material-symbols-outlined" style="color: var(--color-warning);">remove_circle_outline</span>
              <span class="insight-item__text">Cold start latency for worker nodes during sudden traffic spikes.</span>
            </li>
            <li class="insight-item">
              <span class="material-symbols-outlined" style="color: var(--color-warning);">remove_circle_outline</span>
              <span class="insight-item__text">Fuzzy matching fallback requires periodic tuning of threshold parameters.</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  `;

  window.copyCode = function(btn) {
    const code = btn.closest('.code-block').querySelector('.code-block__body').textContent;
    navigator.clipboard.writeText(code).then(() => {
      btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:14px;">check</span> Copied!';
      setTimeout(() => { btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:14px;">content_copy</span> Copy'; }, 2000);
    });
  };
}

function getDemoData() {
  return { explanation: '', reasoning_steps: [] };
}
