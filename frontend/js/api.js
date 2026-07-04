/**
 * StackMind — API Client
 * All fetch calls to the FastAPI backend.
 */

/**
 * API base URL — resolved at runtime:
 *  1. window.__STACKMIND_API_BASE (injected by deployment)
 *  2. In development, Vite proxies /solve, /feedback, /sessions, /health
 *     to localhost:8010, so we can use '' (same origin).
 *  3. Fallback to localhost:8010 for direct API testing.
 */
const API_BASE =
  window.__STACKMIND_API_BASE ||
  (import.meta.env?.DEV ? '' : '') ||
  'http://localhost:8010';

class ApiClient {
  constructor(baseUrl = API_BASE) {
    this.baseUrl = baseUrl;
  }

  async _request(method, path, body = null, options = {}) {
    const url = `${this.baseUrl}${path}`;
    const config = {
      method,
      headers: { 'Content-Type': 'application/json', ...options.headers },
    };
    if (body) {
      config.body = JSON.stringify(body);
    }
    const res = await fetch(url, config);
    if (!res.ok) {
      const errorBody = await res.json().catch(() => ({}));
      throw new ApiError(res.status, errorBody.detail || res.statusText, errorBody);
    }
    return res.json();
  }

  // ── Health ───────────────────────────────────────────────────
  async healthCheck() {
    return this._request('GET', '/health');
  }

  // ── Solve ────────────────────────────────────────────────────
  async solveText(text, sessionId = null, domain = null) {
    return this._request('POST', '/solve/text', {
      text,
      session_id: sessionId,
      domain,
    });
  }

  async solveImage(imageBase64, sessionId = null, domain = null) {
    return this._request('POST', '/solve/image', {
      image_data: imageBase64,
      session_id: sessionId,
      domain,
    });
  }

  async solveVoice(audioBase64, sessionId = null, domain = null) {
    return this._request('POST', '/solve/voice', {
      audio_data: audioBase64,
      session_id: sessionId,
      domain,
    });
  }

  async solveCode(code, language = null, sessionId = null) {
    return this._request('POST', '/solve/code', {
      code,
      language,
      session_id: sessionId,
    });
  }

  // ── Sessions ─────────────────────────────────────────────────
  async getSession(sessionId) {
    return this._request('GET', `/sessions/${sessionId}`);
  }

  async getSolution(sessionId, solutionId) {
    return this._request('GET', `/sessions/${sessionId}/solutions/${solutionId}`);
  }

  // ── Feedback ─────────────────────────────────────────────────
  async submitFeedback(sessionId, solutionId, rating, feedbackText = null) {
    return this._request('POST', '/feedback', {
      session_id: sessionId,
      solution_id: solutionId,
      rating,
      feedback_text: feedbackText,
    });
  }

  // ── Refine ───────────────────────────────────────────────────
  async refineSolution(sessionId, solutionId, text) {
    return this._request('POST', `/sessions/${sessionId}/refine/${solutionId}`, {
      text,
    });
  }

  // ── Multi-Agent Pipeline ──────────────────────────────────────

  /**
   * Stage 2: Decompose query into summary + 3 agent sub-tasks.
   * @param {string}      query   - Raw user query
   * @param {string|null} context - Optional prior conversation context string
   * @returns {{ summary: string, agents: Array<{id: number, task: string}> }}
   */
  async pipelineDecompose(query, context = null) {
    return this._request('POST', '/pipeline/decompose', { query, context });
  }

  /**
   * Stage 3: Run a single agent on its sub-task.
   * @param {string} task    - The agent's assigned sub-task
   * @param {number} agentId - Agent ID (1, 2, or 3)
   * @param {string} query   - Original user query (for context)
   * @returns {{ agent_id: number, output: string }}
   */
  async pipelineRunAgent(task, agentId, query) {
    return this._request('POST', '/pipeline/agent', {
      task,
      agent_id: agentId,
      query,
    });
  }

  /**
   * Stage 4+5: Synthesize 3 agent outputs into a final answer.
   * @param {string} query  - Original user query
   * @param {Array}  agents - [{id, task, output}]
   * @returns {{ final_answer: string }}
   */
  async pipelineSynthesize(query, agents) {
    return this._request('POST', '/pipeline/synthesize', { query, agents });
  }
}

class ApiError extends Error {
  constructor(status, message, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

export const api = new ApiClient();
export { ApiError };
