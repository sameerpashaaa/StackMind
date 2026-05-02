/**
 * StackMind — Reactive State Store
 * Simple pub/sub state management.
 */

const initialState = {
  // Connection
  backendOnline: false,

  // Current session
  currentSessionId: null,
  currentSolutionId: null,

  // Solution data (cached)
  solutions: {},       // { [solutionId]: SolverResponse }
  sessions: [],        // Array of session summaries

  // UI state
  isLoading: false,
  loadingMessage: '',
  activeSolutionTab: 'solution',

  // Settings (client-side mirror)
  settings: {
    llmProvider: 'mistral',
    llmModel: 'mistral-large-latest',
    temperature: 0.7,
    maxTokens: 2000,
    enableText: true,
    enableImage: true,
    enableVoice: true,
    enableCode: true,
  },
};

class Store {
  constructor() {
    this._state = { ...initialState };
    this._listeners = {};
  }

  get(key) {
    return this._state[key];
  }

  set(key, value) {
    const old = this._state[key];
    this._state[key] = value;
    if (old !== value) {
      this._notify(key, value, old);
    }
  }

  /**
   * Update multiple keys at once.
   */
  update(partial) {
    for (const [key, value] of Object.entries(partial)) {
      this.set(key, value);
    }
  }

  /**
   * Subscribe to changes on a specific key.
   */
  on(key, fn) {
    if (!this._listeners[key]) this._listeners[key] = [];
    this._listeners[key].push(fn);
    return () => {
      this._listeners[key] = this._listeners[key].filter(f => f !== fn);
    };
  }

  _notify(key, value, old) {
    (this._listeners[key] || []).forEach(fn => fn(value, old));
    (this._listeners['*'] || []).forEach(fn => fn(key, value, old));
  }

  /**
   * Cache a solution response.
   */
  cacheSolution(solutionId, data) {
    const solutions = { ...this._state.solutions, [solutionId]: data };
    this.set('solutions', solutions);
  }

  /**
   * Get a cached solution.
   */
  getCachedSolution(solutionId) {
    return this._state.solutions[solutionId] || null;
  }
}

export const store = new Store();
