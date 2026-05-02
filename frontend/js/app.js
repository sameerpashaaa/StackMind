/**
 * StackMind — Application Entry Point
 */
import { Router } from './router.js';
import { api } from './api.js';
import { store } from './state.js';

// Views
import { renderHome } from './views/home.js';
import { renderSolution } from './views/solution.js';
import { renderExplanation } from './views/explanation.js';
import { renderVerification } from './views/verification.js';
import { renderAlternatives } from './views/alternatives.js';
import { renderRefinement } from './views/refinement.js';
import { renderPlan } from './views/plan.js';
import { renderKnowledgeGraph } from './views/knowledge-graph.js';
import { renderMemory } from './views/memory.js';
import { renderSessions } from './views/sessions.js';
import { renderSettings } from './views/settings.js';
import { renderImageAnalysis } from './views/image-analysis.js';
import { renderCodeAnalysis } from './views/code-analysis.js';
import { renderApiDocs } from './views/api-docs.js';

// ── Init ────────────────────────────────────────────────────────
const mainContent = document.getElementById('main-content');
const router = new Router();

// ── Routes ──────────────────────────────────────────────────────
router
  .on('/', () => renderHome(mainContent))
  .on('/sessions', () => renderSessions(mainContent))
  .on('/knowledge-graph', () => renderKnowledgeGraph(mainContent))
  .on('/memory', () => renderMemory(mainContent))
  .on('/settings', () => renderSettings(mainContent))
  .on('/api-docs', () => renderApiDocs(mainContent))
  .on('/solution/:sessionId/:solutionId', (params) => renderSolution(mainContent, params))
  .on('/solution/:sessionId/:solutionId/explanation', (params) => renderExplanation(mainContent, params))
  .on('/solution/:sessionId/:solutionId/verification', (params) => renderVerification(mainContent, params))
  .on('/solution/:sessionId/:solutionId/alternatives', (params) => renderAlternatives(mainContent, params))
  .on('/solution/:sessionId/:solutionId/refine', (params) => renderRefinement(mainContent, params))
  .on('/solution/:sessionId/:solutionId/plan', (params) => renderPlan(mainContent, params))
  .on('/solution/:sessionId/:solutionId/image', (params) => renderImageAnalysis(mainContent, params))
  .on('/solution/:sessionId/:solutionId/code', (params) => renderCodeAnalysis(mainContent, params));

// ── Sidebar Active State ────────────────────────────────────────
router.beforeEach = (route) => {
  const hash = route.hash;
  document.querySelectorAll('#sidebar-nav .nav-item').forEach(item => {
    const itemRoute = item.dataset.route;
    const isActive =
      (itemRoute === '/' && hash === '/') ||
      (itemRoute !== '/' && hash.startsWith(itemRoute)) ||
      (itemRoute === '/sessions' && hash.startsWith('/solution'));
    item.classList.toggle('active', isActive);
  });

  // Scroll main content to top on navigation
  mainContent.scrollTo(0, 0);
};

// ── Health Check ────────────────────────────────────────────────
async function checkHealth() {
  const statusEl = document.getElementById('system-status');
  try {
    await api.healthCheck();
    store.set('backendOnline', true);
    statusEl.innerHTML = `
      <span class="status-dot status-dot--online"></span>
      <span class="status-text">System Online</span>
    `;
  } catch {
    store.set('backendOnline', false);
    statusEl.innerHTML = `
      <span class="status-dot status-dot--offline"></span>
      <span class="status-text">Backend Offline</span>
    `;
  }
}

// ── Boot ─────────────────────────────────────────────────────────
checkHealth();
setInterval(checkHealth, 30000);
router.start();
