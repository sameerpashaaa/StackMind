/**
 * Session History View
 */
export function renderSessions(container) {
  const sessions = [
    { id: 'sess-001', title: 'Non-Linear Optimization Problem', domain: 'math', date: 'Today, 10:23 AM', solutions: 3, status: 'completed' },
    { id: 'sess-002', title: 'Python async/await debugging', domain: 'code', date: 'Today, 9:15 AM', solutions: 2, status: 'completed' },
    { id: 'sess-003', title: 'Thermodynamics heat transfer analysis', domain: 'science', date: 'Yesterday, 4:30 PM', solutions: 1, status: 'completed' },
    { id: 'sess-004', title: 'Binary search tree implementation', domain: 'code', date: 'Yesterday, 2:10 PM', solutions: 4, status: 'completed' },
    { id: 'sess-005', title: 'Fourier transform applications', domain: 'math', date: '2 days ago', solutions: 2, status: 'completed' },
  ];

  container.innerHTML = `
    <div class="page">
      <div class="page-header">
        <h1 class="page-header__title">Session History</h1>
        <p class="page-header__subtitle">Browse previous problem-solving sessions.</p>
      </div>
      <div class="search-bar" style="margin-bottom:var(--space-lg);">
        <span class="material-symbols-outlined">search</span>
        <input type="text" class="input" placeholder="Search sessions..." id="session-search" />
      </div>
      <div id="session-list" style="display:flex;flex-direction:column;gap:var(--space-sm);">
        ${sessions.map(s => `
          <a href="#/solution/${s.id}/demo" class="card card--interactive" style="text-decoration:none;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <div>
                <h3 class="text-body-sm" style="font-weight:600;margin-bottom:4px;">${s.title}</h3>
                <div style="display:flex;align-items:center;gap:var(--space-sm);">
                  <span class="chip chip--${s.domain === 'math' ? 'accent' : s.domain === 'code' ? 'default' : 'success'}">${s.domain}</span>
                  <span class="text-label text-muted">${s.date}</span>
                  <span class="text-label text-muted">${s.solutions} solution${s.solutions > 1 ? 's' : ''}</span>
                </div>
              </div>
              <span class="material-symbols-outlined text-muted">chevron_right</span>
            </div>
          </a>
        `).join('')}
      </div>
    </div>
  `;
  container.querySelector('#session-search').addEventListener('input', e => {
    const q = e.target.value.toLowerCase();
    container.querySelectorAll('#session-list .card').forEach(c => {
      c.style.display = c.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });
}
