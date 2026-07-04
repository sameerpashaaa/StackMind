/**
 * StackMind — Pipeline Flowchart View
 *
 * Renders and live-updates the 5-stage multi-agent pipeline flowchart.
 *
 * Public API:
 *   mountPipelineView(container, query)  — builds the scaffold HTML
 *   updatePipelineView(container, state) — diffs and updates the UI
 */

// ── Helpers ──────────────────────────────────────────────────────

function qs(container, sel) {
  return container.querySelector(sel);
}

function statusBadgeHTML(status) {
  switch (status) {
    case 'active':
      return `<span class="node-status node-status--active">
        <span class="status-spinner"></span> Processing...
      </span>`;
    case 'done':
      return `<span class="node-status node-status--complete">
        <span class="material-symbols-outlined status-icon">check_circle</span> Done
      </span>`;
    case 'error':
      return `<span class="node-status node-status--error">
        <span class="material-symbols-outlined status-icon">error</span> Failed
      </span>`;
    default:
      return `<span class="node-status node-status--pending">
        <span class="material-symbols-outlined status-icon">schedule</span> Waiting
      </span>`;
  }
}

function nodeStateClass(status) {
  const map = {
    pending: 'pipeline-node--pending',
    active:  'pipeline-node--active',
    done:    'pipeline-node--complete',
    error:   'pipeline-node--error',
  };
  return map[status] || 'pipeline-node--pending';
}

function agentNodeStateClass(status) {
  const map = {
    pending: '',
    active:  'pipeline-agent-node--active',
    done:    'pipeline-agent-node--complete',
    error:   'pipeline-agent-node--error',
  };
  return map[status] || '';
}

function truncate(text, maxLen = 180) {
  if (!text) return '';
  return text.length > maxLen ? text.slice(0, maxLen).trimEnd() + '…' : text;
}

function copyToClipboard(text) {
  navigator.clipboard?.writeText(text).catch(() => {
    // Fallback
    const el = document.createElement('textarea');
    el.value = text;
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    el.remove();
  });
}

// ── Mount ─────────────────────────────────────────────────────────

/**
 * Replaces container content with the pipeline scaffold.
 * Called once when the user submits a query.
 *
 * @param {HTMLElement} container - The #main-content element
 * @param {string} query          - The user's raw query (for display)
 * @param {Function} onBack       - Callback when Back button is clicked
 * @param {Function} onFollowUp   - Callback when "Ask follow-up" is clicked
 * @param {Function} onRegenerate - Callback when "Regenerate" is clicked
 */
export function mountPipelineView(container, query, { onBack, onFollowUp, onRegenerate }) {
  const shortQuery = query.length > 80 ? query.slice(0, 80) + '…' : query;

  container.innerHTML = `
    <div class="pipeline-view" id="pipeline-view">
      <!-- Header -->
      <div class="pipeline-header">
        <div>
          <div class="pipeline-header__title">Reasoning Pipeline</div>
          <div class="pipeline-header__query">"${escapeHtml(shortQuery)}"</div>
        </div>
        <button class="pipeline-back-btn" id="pipeline-back-btn">
          <span class="material-symbols-outlined">arrow_back</span>
          New Query
        </button>
      </div>

      <!-- Flow -->
      <div class="pipeline-flow" id="pipeline-flow">

        <!-- Stage 1 — User Query -->
        <div class="pipeline-node pipeline-node--complete pipeline-node--visible" id="node-stage1">
          <div class="node-header">
            <div>
              <div class="node-stage-label">Stage 1</div>
              <div class="node-title">User Query</div>
            </div>
            <span class="node-status node-status--complete">
              <span class="material-symbols-outlined status-icon">check_circle</span> Captured
            </span>
          </div>
          <div class="node-body node-body--primary">${escapeHtml(query)}</div>
        </div>

        <!-- Connector 1 → 2 -->
        <div class="pipeline-connector" id="conn-1-2">
          <div class="pipeline-connector__line"></div>
          <div class="pipeline-connector__arrow"></div>
        </div>

        <!-- Stage 2 — Query Understanding -->
        <div class="pipeline-node pipeline-node--pending" id="node-stage2">
          <div class="node-header">
            <div>
              <div class="node-stage-label">Stage 2</div>
              <div class="node-title">Query Understanding</div>
            </div>
            <span id="badge-stage2">${statusBadgeHTML('pending')}</span>
          </div>
          <div class="node-body" id="body-stage2">
            <div class="dot-loader">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>

        <!-- Fork Connector 2 → 3 agents -->
        <div class="pipeline-connector pipeline-connector--fork" id="conn-fork">
          <svg viewBox="0 0 720 48" preserveAspectRatio="none" aria-hidden="true">
            <!-- Center down, then branch left/center/right -->
            <path class="fork-path"
              d="M 360 0 L 360 20
                 M 360 20 L 120 20 L 120 48
                 M 360 20 L 360 20 L 360 48
                 M 360 20 L 600 20 L 600 48"
            />
          </svg>
        </div>

        <!-- Stage 3 — Parallel Agents Row -->
        <div class="pipeline-agents-row" id="agents-row">
          ${[1, 2, 3].map((n) => `
            <div class="pipeline-agent-node" id="node-agent${n}">
              <div class="agent-node-header">
                <div class="agent-node-title">Agent ${n}</div>
                <span id="badge-agent${n}">${statusBadgeHTML('pending')}</span>
              </div>
              <div class="agent-node-task text-muted" id="task-agent${n}">
                Awaiting assignment...
              </div>
              <div id="output-agent${n}" class="hidden"></div>
            </div>
          `).join('')}
        </div>

        <!-- Merge Connector agents → 4 -->
        <div class="pipeline-connector pipeline-connector--merge" id="conn-merge">
          <svg viewBox="0 0 720 48" preserveAspectRatio="none" aria-hidden="true">
            <path class="merge-path"
              d="M 120 0 L 120 28 L 360 28
                 M 360 0 L 360 28
                 M 600 0 L 600 28 L 360 28
                 M 360 28 L 360 48"
            />
          </svg>
        </div>

        <!-- Stage 4 — Output Compilation -->
        <div class="pipeline-node pipeline-node--pending" id="node-stage4">
          <div class="node-header">
            <div>
              <div class="node-stage-label">Stage 4</div>
              <div class="node-title">Output Compilation</div>
            </div>
            <span id="badge-stage4">${statusBadgeHTML('pending')}</span>
          </div>
          <div class="node-body" id="body-stage4">
            Waiting for all agents to complete...
          </div>
        </div>

        <!-- Connector 4 → 5 -->
        <div class="pipeline-connector" id="conn-4-5">
          <div class="pipeline-connector__line"></div>
          <div class="pipeline-connector__arrow"></div>
        </div>

        <!-- Stage 5 — Final Answer -->
        <div class="pipeline-node pipeline-node--pending" id="node-stage5">
          <div class="node-header">
            <div>
              <div class="node-stage-label">Stage 5</div>
              <div class="node-title">Final Answer</div>
            </div>
            <span id="badge-stage5">${statusBadgeHTML('pending')}</span>
          </div>
          <div class="node-body" id="body-stage5">
            Synthesizing response...
          </div>
          <div id="pipeline-actions-area" class="hidden"></div>
        </div>

      </div><!-- /pipeline-flow -->
    </div><!-- /pipeline-view -->
  `;

  // Animate connector 1→2 after mount
  requestAnimationFrame(() => {
    _drawConnector(container, 'conn-1-2');
  });

  // Back button
  qs(container, '#pipeline-back-btn').addEventListener('click', onBack);

  // Store callbacks on the view root for updatePipelineView to wire up later
  const root = qs(container, '#pipeline-view');
  root._onFollowUp   = onFollowUp;
  root._onRegenerate = onRegenerate;
}

// ── Update ────────────────────────────────────────────────────────

/**
 * Called on every state change by the pipeline orchestrator.
 * Diffs the current DOM against the new state and patches only what changed.
 *
 * @param {HTMLElement} container - The #main-content element
 * @param {Object}      state     - Full pipelineState snapshot
 */
export function updatePipelineView(container, state) {
  const view = qs(container, '#pipeline-view');
  if (!view) return;

  const { decomposition, agents, compilation, finalAnswer, status, error } = state;

  // ── Stage 2 ────────────────────────────────────────────────────
  if (status === 'running' || status === 'error' || decomposition) {
    _activateNode(container, 'node-stage2');
  }

  if (decomposition) {
    // Once we have decomposition, it's always done
    _setNodeState(container, 'node-stage2', 'done');
    _setBadge(container, 'badge-stage2', statusBadgeHTML('done'));

    const body2 = qs(container, '#body-stage2');
    if (body2 && !body2.dataset.rendered) {
      body2.dataset.rendered = '1';
      body2.innerHTML = `
        <p class="node-body">${escapeHtml(decomposition.summary)}</p>
        <div class="agent-assignments">
          ${decomposition.agents.map((a) => `
            <div class="agent-assignment">
              <div class="agent-assignment__badge">${a.id}</div>
              <div class="agent-assignment__task">Agent ${a.id} → ${escapeHtml(a.task)}</div>
            </div>
          `).join('')}
        </div>
      `;

      // Draw fork connector
      setTimeout(() => _drawFork(container, 'conn-fork'), 100);

      // Activate all 3 agent nodes
      setTimeout(() => {
        [1, 2, 3].forEach((n, i) => {
          setTimeout(() => _activateNode(container, `node-agent${n}`), i * 80);
        });
      }, 300);
    }
  } else if (status === 'running') {
    _setNodeState(container, 'node-stage2', 'active');
    _setBadge(container, 'badge-stage2', statusBadgeHTML('active'));
  }

  // ── Stage 3 — Agents ───────────────────────────────────────────
  agents.forEach((agent) => {
    const nodeEl = qs(container, `#node-agent${agent.id}`);
    if (!nodeEl) return;

    // Update task label if just assigned
    const taskEl = qs(container, `#task-agent${agent.id}`);
    if (taskEl && agent.task && taskEl.textContent.includes('Awaiting')) {
      taskEl.textContent = agent.task;
    }

    // Update badge
    _setBadge(container, `badge-agent${agent.id}`, statusBadgeHTML(agent.status));

    // Update node border state
    _setAgentNodeState(nodeEl, agent.status);

    // Show output when done
    if (agent.status === 'done' && agent.output) {
      const outEl = qs(container, `#output-agent${agent.id}`);
      if (outEl && outEl.classList.contains('hidden')) {
        outEl.classList.remove('hidden');
        outEl.className = 'agent-node-output';
        outEl.textContent = truncate(agent.output, 200);
      }
    }
  });

  // ── Stage 4 — Compilation ──────────────────────────────────────
  const comp = compilation;
  if (comp.status === 'active' || comp.status === 'done') {
    if (!qs(container, '#node-stage4').classList.contains('pipeline-node--visible')) {
      // Draw merge connector first
      _drawMerge(container, 'conn-merge');
      setTimeout(() => {
        _activateNode(container, 'node-stage4');
      }, 300);
    }
    _setNodeState(container, 'node-stage4', comp.status === 'done' ? 'done' : 'active');
    _setBadge(container, 'badge-stage4', statusBadgeHTML(comp.status === 'done' ? 'done' : 'active'));
    const body4 = qs(container, '#body-stage4');
    if (body4) {
      body4.innerHTML = comp.status === 'active'
        ? `<span style="color:var(--color-accent)">Merging outputs from 3 agents...</span>
           <div class="dot-loader" style="margin-top:8px"><span></span><span></span><span></span></div>`
        : `<span style="color:var(--color-on-surface-variant)">All outputs merged successfully.</span>`;
    }
  }

  // ── Stage 5 — Final Answer ─────────────────────────────────────
  if (finalAnswer) {
    if (!qs(container, '#node-stage5').classList.contains('pipeline-node--visible')) {
      _drawConnector(container, 'conn-4-5');
      setTimeout(() => _activateNode(container, 'node-stage5'), 250);
    }
    _setNodeState(container, 'node-stage5', status === 'done' ? 'done' : 'active');
    _setBadge(container, 'badge-stage5', statusBadgeHTML(status === 'done' ? 'done' : 'active'));

    const body5 = qs(container, '#body-stage5');
    if (body5 && !body5.dataset.rendered) {
      body5.dataset.rendered = '1';
      body5.innerHTML = `<div class="node-final-answer">${formatAnswer(finalAnswer)}</div>`;
    }

    // Pipeline actions
    if (status === 'done') {
      const actionsArea = qs(container, '#pipeline-actions-area');
      if (actionsArea && actionsArea.classList.contains('hidden')) {
        actionsArea.classList.remove('hidden');
        actionsArea.innerHTML = `
          <div class="pipeline-actions">
            <button class="btn btn--secondary" id="action-copy">
              <span class="material-symbols-outlined">content_copy</span>
              Copy
            </button>
            <button class="btn btn--secondary" id="action-regenerate">
              <span class="material-symbols-outlined">refresh</span>
              Regenerate
            </button>
            <button class="btn btn--primary" id="action-followup">
              <span class="material-symbols-outlined">chat</span>
              Ask follow-up
            </button>
          </div>
        `;

        // Wire action buttons
        qs(actionsArea, '#action-copy').addEventListener('click', () => {
          copyToClipboard(finalAnswer);
          const btn = qs(actionsArea, '#action-copy');
          btn.innerHTML = `<span class="material-symbols-outlined">check</span> Copied!`;
          setTimeout(() => {
            btn.innerHTML = `<span class="material-symbols-outlined">content_copy</span> Copy`;
          }, 2000);
        });

        qs(actionsArea, '#action-regenerate').addEventListener('click', () => {
          // Clear stage 5 body so it can re-render
          body5.dataset.rendered = '';
          body5.innerHTML = `<span style="color:var(--color-on-surface-variant)">Regenerating answer...</span>`;
          actionsArea.classList.add('hidden');
          view._onRegenerate?.();
        });

        qs(actionsArea, '#action-followup').addEventListener('click', () => {
          view._onFollowUp?.({ query: state.query, answer: finalAnswer });
        });
      }
    }
  }

  // ── Error State ────────────────────────────────────────────────
  if (status === 'error' && error) {
    // Find the last visible node and show error on it
    const lastActive = qs(container, '.pipeline-node--active') ||
                       qs(container, '#node-stage2');
    if (lastActive) {
      lastActive.classList.remove('pipeline-node--active');
      lastActive.classList.add('pipeline-node--error');
      const badge = lastActive.querySelector('[id^="badge-"]');
      if (badge) badge.innerHTML = statusBadgeHTML('error');
      const body = lastActive.querySelector('.node-body');
      if (body) {
        body.innerHTML = `<span style="color:var(--color-error)">
          <strong>Error:</strong> ${escapeHtml(error)}
        </span>`;
      }
    }
  }
}

// ── Internal DOM helpers ───────────────────────────────────────────

function _activateNode(container, id) {
  const node = qs(container, `#${id}`);
  if (!node) return;
  node.classList.add('pipeline-node--visible');
}

function _setNodeState(container, id, status) {
  const node = qs(container, `#${id}`);
  if (!node) return;
  node.classList.remove(
    'pipeline-node--pending',
    'pipeline-node--active',
    'pipeline-node--complete',
    'pipeline-node--error'
  );
  node.classList.add(nodeStateClass(status));
}

function _setAgentNodeState(nodeEl, status) {
  nodeEl.classList.remove(
    'pipeline-agent-node--active',
    'pipeline-agent-node--complete',
    'pipeline-agent-node--error'
  );
  const cls = agentNodeStateClass(status);
  if (cls) nodeEl.classList.add(cls);
  if (status !== 'pending' && !nodeEl.classList.contains('pipeline-node--visible')) {
    nodeEl.classList.add('pipeline-agent-node--visible');
  }
}

function _setBadge(container, id, html) {
  const el = qs(container, `#${id}`);
  if (el) el.innerHTML = html;
}

function _drawConnector(container, id) {
  const conn = qs(container, `#${id}`);
  if (!conn) return;
  conn.querySelector('.pipeline-connector__line')?.classList.add('pipeline-connector__line--drawn');
  setTimeout(() => {
    conn.querySelector('.pipeline-connector__arrow')?.classList.add('pipeline-connector__arrow--drawn');
  }, 450);
}

function _drawFork(container, id) {
  const conn = qs(container, `#${id}`);
  if (!conn) return;
  conn.querySelector('.fork-path')?.classList.add('fork-path--drawn');
}

function _drawMerge(container, id) {
  const conn = qs(container, `#${id}`);
  if (!conn) return;
  conn.querySelector('.merge-path')?.classList.add('merge-path--drawn');
}

// ── Text helpers ───────────────────────────────────────────────────

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Converts plain text answer to light HTML.
 * Preserves newlines as <br> and wraps paragraphs.
 */
function formatAnswer(text) {
  return text
    .split(/\n\n+/)
    .map((para) => `<p>${escapeHtml(para.trim()).replace(/\n/g, '<br>')}</p>`)
    .join('');
}
