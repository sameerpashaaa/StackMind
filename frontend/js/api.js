/**
 * StackMind — API Client
 * All fetch calls to the FastAPI backend.
 */

const API_BASE = 'http://localhost:8000';

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
