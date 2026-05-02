/**
 * Memory Browser View
 */
export function renderMemory(container) {
  const entries = [
    { id: 1, type: 'solution', title: 'Constrained Optimization — Lagrangian', domain: 'math', date: '2 hours ago', relevance: 0.94 },
    { id: 2, type: 'context', title: 'Python scipy.optimize.minimize usage', domain: 'code', date: '5 hours ago', relevance: 0.87 },
    { id: 3, type: 'entity', title: 'KKT Conditions — Necessary conditions for optimality', domain: 'math', date: '1 day ago', relevance: 0.82 },
    { id: 4, type: 'solution', title: 'Gradient Descent convergence analysis', domain: 'math', date: '2 days ago', relevance: 0.76 },
    { id: 5, type: 'context', title: 'NumPy array broadcasting rules', domain: 'code', date: '3 days ago', relevance: 0.71 },
    { id: 6, type: 'entity', title: 'Convexity and global minima', domain: 'science', date: '5 days ago', relevance: 0.65 },
  ];

  container.innerHTML = `
    <div class="page">
      <div class="page-header">
        <h1 class="page-header__title">Memory Browser</h1>
        <p class="page-header__subtitle">Browse and search persistent memory entries stored across sessions.</p>
      </div>
      <div style="display:flex;gap:var(--space-md);margin-bottom:var(--space-lg);align-items:center;">
        <div class="search-bar" style="flex:1;">
          <span class="material-symbols-outlined">search</span>
          <input type="text" class="input" placeholder="Search memory entries..." id="memory-search" />
        </div>
        <select class="select" style="width:160px;" id="memory-filter">
          <option value="">All Types</option>
          <option value="solution">Solutions</option>
          <option value="context">Context</option>
          <option value="entity">Entities</option>
        </select>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:var(--space-md);margin-bottom:var(--space-lg);">
        <div class="card" style="display:flex;align-items:center;gap:var(--space-md);"><div class="metric"><div class="metric__value">${entries.length}</div><div class="metric__label">Total Entries</div></div></div>
        <div class="card" style="display:flex;align-items:center;gap:var(--space-md);"><div class="metric"><div class="metric__value">3</div><div class="metric__label">Domains</div></div></div>
      </div>
      <div id="memory-list" style="display:flex;flex-direction:column;gap:var(--space-sm);">
        ${entries.map(e => `
          <div class="card card--interactive" data-type="${e.type}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
              <div style="flex:1;">
                <div style="display:flex;align-items:center;gap:var(--space-sm);margin-bottom:4px;">
                  <span class="chip chip--${e.type === 'solution' ? 'accent' : e.type === 'context' ? 'default' : 'success'}">${e.type}</span>
                  <span class="chip chip--default">${e.domain}</span>
                </div>
                <h3 class="text-body-sm" style="font-weight:600;">${e.title}</h3>
                <span class="text-label text-muted" style="margin-top:4px;display:block;">${e.date}</span>
              </div>
              <div class="metric" style="text-align:right;"><div class="metric__value" style="font-size:18px;">${Math.round(e.relevance * 100)}%</div><div class="metric__label">Relevance</div></div>
            </div>
          </div>
        `).join('')}
      </div>
    </div>
  `;
  const search = container.querySelector('#memory-search');
  const filter = container.querySelector('#memory-filter');
  function applyFilter() {
    const q = search.value.toLowerCase();
    const t = filter.value;
    container.querySelectorAll('#memory-list .card').forEach(card => {
      const text = card.textContent.toLowerCase();
      const type = card.dataset.type;
      card.style.display = (text.includes(q) && (!t || type === t)) ? '' : 'none';
    });
  }
  search.addEventListener('input', applyFilter);
  filter.addEventListener('change', applyFilter);
}
