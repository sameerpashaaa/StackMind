/**
 * API Reference & Playground View
 */
export function renderApiDocs(container) {
  const endpoints = [
    { method: 'POST', path: '/solve/text', desc: 'Solve a text-based problem', body: '{ "text": "Find primes < 50", "domain": "math" }' },
    { method: 'POST', path: '/solve/image', desc: 'Solve from an image (base64)', body: '{ "image_data": "base64...", "domain": null }' },
    { method: 'POST', path: '/solve/voice', desc: 'Solve from audio (base64)', body: '{ "audio_data": "base64..." }' },
    { method: 'POST', path: '/solve/code', desc: 'Analyze/debug code', body: '{ "code": "def foo()...", "language": "python" }' },
    { method: 'POST', path: '/feedback', desc: 'Submit feedback on a solution', body: '{ "session_id": "...", "solution_id": "...", "rating": 5 }' },
    { method: 'GET', path: '/sessions/{id}', desc: 'Get session information', body: null },
    { method: 'GET', path: '/health', desc: 'Health check', body: null },
  ];

  container.innerHTML = `
    <div class="page">
      <div class="page-header">
        <h1 class="page-header__title">API Reference</h1>
        <p class="page-header__subtitle">Interactive documentation and testing playground for the StackMind REST API.</p>
        <div style="margin-top:var(--space-md);display:flex;gap:var(--space-sm);">
          <span class="chip chip--success"><span class="chip__dot chip__dot--success"></span> API Online</span>
          <span class="chip chip--default">Base URL: http://localhost:8010</span>
        </div>
      </div>

      <!-- Playground -->
      <div class="card" style="margin-bottom:var(--space-xl);">
        <h2 class="card__title" style="margin-bottom:var(--space-md);">Playground</h2>
        <div style="display:flex;gap:var(--space-sm);margin-bottom:var(--space-md);">
          <select id="playground-method" class="select" style="width:100px;"><option>POST</option><option>GET</option></select>
          <input id="playground-url" class="input" value="/solve/text" style="flex:1;" />
          <button id="playground-send" class="btn btn--primary"><span class="material-symbols-outlined">send</span> Send</button>
        </div>
        <textarea id="playground-body" class="input" rows="4" style="font-family:var(--font-code);font-size:var(--text-code-size);">{ "text": "What is the derivative of x^3 + 2x?" }</textarea>
        <div id="playground-response" class="hidden" style="margin-top:var(--space-md);">
          <div class="text-label" style="margin-bottom:var(--space-xs);">Response</div>
          <div class="code-block"><pre class="code-block__body" id="playground-output"></pre></div>
        </div>
      </div>

      <!-- Endpoints -->
      <h2 class="text-h2" style="margin-bottom:var(--space-lg);">Endpoints</h2>
      <div style="display:flex;flex-direction:column;gap:var(--space-sm);">
        ${endpoints.map(ep => `
          <div class="card">
            <div style="display:flex;align-items:center;gap:var(--space-sm);margin-bottom:var(--space-sm);">
              <span class="chip chip--${ep.method === 'POST' ? 'accent' : 'success'}">${ep.method}</span>
              <code class="text-code" style="font-weight:500;">${ep.path}</code>
            </div>
            <p class="text-body-sm text-muted">${ep.desc}</p>
            ${ep.body ? `<div class="code-block" style="margin-top:var(--space-sm);"><pre class="code-block__body">${ep.body}</pre></div>` : ''}
          </div>
        `).join('')}
      </div>
    </div>
  `;

  container.querySelector('#playground-send').addEventListener('click', async () => {
    const method = container.querySelector('#playground-method').value;
    const url = container.querySelector('#playground-url').value;
    const body = container.querySelector('#playground-body').value;
    const output = container.querySelector('#playground-output');
    const responseDiv = container.querySelector('#playground-response');
    try {
      const opts = { method, headers: { 'Content-Type': 'application/json' } };
      if (method === 'POST') opts.body = body;
      const res = await fetch(`http://localhost:8010${url}`, opts);
      const data = await res.json();
      output.textContent = JSON.stringify(data, null, 2);
    } catch (err) { output.textContent = `Error: ${err.message}`; }
    responseDiv.classList.remove('hidden');
  });
}
