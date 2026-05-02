/**
 * StackMind — Hash-based SPA Router
 */

export class Router {
  constructor() {
    this.routes = [];
    this.currentRoute = null;
    this.beforeEach = null;
    window.addEventListener('hashchange', () => this._onHashChange());
  }

  /**
   * Register a route pattern with a handler function.
   * Supports :param style URL parameters.
   */
  on(pattern, handler) {
    const paramNames = [];
    const regexStr = pattern.replace(/:([^/]+)/g, (_, name) => {
      paramNames.push(name);
      return '([^/]+)';
    });
    this.routes.push({
      pattern,
      regex: new RegExp(`^${regexStr}$`),
      paramNames,
      handler,
    });
    return this;
  }

  /**
   * Navigate to a hash route.
   */
  navigate(hash) {
    window.location.hash = hash;
  }

  /**
   * Start the router — process the current hash.
   */
  start() {
    this._onHashChange();
  }

  /**
   * Internal: match the current hash against registered routes.
   */
  _onHashChange() {
    const hash = window.location.hash.slice(1) || '/';
    
    for (const route of this.routes) {
      const match = hash.match(route.regex);
      if (match) {
        const params = {};
        route.paramNames.forEach((name, i) => {
          params[name] = decodeURIComponent(match[i + 1]);
        });

        this.currentRoute = { pattern: route.pattern, hash, params };

        if (this.beforeEach) {
          this.beforeEach(this.currentRoute);
        }

        route.handler(params);
        return;
      }
    }

    // Fallback to home if no route matches
    this.navigate('/');
  }

  /**
   * Get current route params.
   */
  getParams() {
    return this.currentRoute?.params || {};
  }
}
