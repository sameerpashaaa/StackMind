/**
 * Knowledge Graph View — Interactive topology map
 */
export function renderKnowledgeGraph(container) {
  container.innerHTML = `
    <div class="page">
      <div class="page-header">
        <h1 class="page-header__title">Topology Map</h1>
        <p class="page-header__subtitle">Interactive knowledge graph showing entity relationships and causal paths.</p>
      </div>
      <div class="content-grid content-grid--sidebar">
        <div>
          <div class="graph-container" id="graph-canvas-container">
            <canvas id="knowledge-graph-canvas"></canvas>
            <div class="graph-controls">
              <button class="btn btn--secondary btn--icon" id="graph-zoom-in"><span class="material-symbols-outlined">add</span></button>
              <button class="btn btn--secondary btn--icon" id="graph-zoom-out"><span class="material-symbols-outlined">remove</span></button>
              <button class="btn btn--secondary btn--icon" id="graph-reset"><span class="material-symbols-outlined">fit_screen</span></button>
            </div>
          </div>
          <div style="display:flex;gap:var(--space-lg);margin-top:var(--space-md);">
            <div style="display:flex;align-items:center;gap:6px;"><div style="width:12px;height:12px;border-radius:50%;background:#3B82F6;"></div><span class="text-label">Concept</span></div>
            <div style="display:flex;align-items:center;gap:6px;"><div style="width:12px;height:12px;border-radius:50%;background:#22C55E;"></div><span class="text-label">Method</span></div>
            <div style="display:flex;align-items:center;gap:6px;"><div style="width:12px;height:12px;border-radius:50%;background:#F59E0B;"></div><span class="text-label">Variable</span></div>
            <div style="display:flex;align-items:center;gap:6px;"><div style="width:12px;height:12px;border-radius:50%;background:#8B5CF6;"></div><span class="text-label">Result</span></div>
          </div>
        </div>
        <div>
          <div class="card" id="node-detail-panel">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">Entity Detail</h3>
            <div id="node-detail-content">
              <p class="text-muted text-body-sm">Click a node on the graph to view its properties.</p>
            </div>
          </div>
          <div class="card" style="margin-top:var(--space-lg);">
            <h3 class="card__title" style="margin-bottom:var(--space-md);">Relationship Distribution</h3>
            <div style="display:flex;flex-direction:column;gap:var(--space-sm);">
              <div><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span class="text-body-sm">causes</span><span class="text-label">34%</span></div><div class="progress"><div class="progress__bar" style="width:34%;"></div></div></div>
              <div><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span class="text-body-sm">depends_on</span><span class="text-label">28%</span></div><div class="progress"><div class="progress__bar progress__bar--success" style="width:28%;"></div></div></div>
              <div><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span class="text-body-sm">relates_to</span><span class="text-label">22%</span></div><div class="progress"><div class="progress__bar progress__bar--warning" style="width:22%;"></div></div></div>
              <div><div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span class="text-body-sm">produces</span><span class="text-label">16%</span></div><div class="progress"><div class="progress__bar" style="width:16%;background:var(--color-on-surface-muted);"></div></div></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
  initGraph();
}

function initGraph() {
  const canvas = document.getElementById('knowledge-graph-canvas');
  if (!canvas) return;
  const container = document.getElementById('graph-canvas-container');
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
  const ctx = canvas.getContext('2d');
  const nodes = [
    { id: 'opt', label: 'Optimization', x: 400, y: 250, r: 30, color: '#3B82F6', type: 'Concept' },
    { id: 'lag', label: 'Lagrangian', x: 250, y: 150, r: 24, color: '#22C55E', type: 'Method' },
    { id: 'obj', label: 'Objective f(x,y)', x: 550, y: 150, r: 22, color: '#F59E0B', type: 'Variable' },
    { id: 'con', label: 'Constraint', x: 200, y: 320, r: 22, color: '#F59E0B', type: 'Variable' },
    { id: 'sol', label: 'x*=y*=0.5', x: 500, y: 370, r: 26, color: '#8B5CF6', type: 'Result' },
    { id: 'kkt', label: 'KKT Conditions', x: 350, y: 420, r: 20, color: '#22C55E', type: 'Method' },
    { id: 'sub', label: 'Substitution', x: 600, y: 280, r: 20, color: '#22C55E', type: 'Method' },
    { id: 'geo', label: 'Geometric', x: 150, y: 200, r: 18, color: '#22C55E', type: 'Method' },
  ];
  const edges = [
    ['opt','lag'],['opt','obj'],['opt','con'],['lag','sol'],['con','kkt'],['kkt','sol'],['opt','sub'],['sub','sol'],['lag','geo'],
  ];

  let scale = 1, offsetX = 0, offsetY = 0, dragging = false, lastMouse = null;

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save(); ctx.translate(offsetX, offsetY); ctx.scale(scale, scale);
    edges.forEach(([a, b]) => {
      const na = nodes.find(n => n.id === a), nb = nodes.find(n => n.id === b);
      ctx.beginPath(); ctx.moveTo(na.x, na.y); ctx.lineTo(nb.x, nb.y);
      ctx.strokeStyle = '#E5E5E5'; ctx.lineWidth = 1.5; ctx.stroke();
    });
    nodes.forEach(n => {
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = n.color + '20'; ctx.fill();
      ctx.strokeStyle = n.color; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = '#18181B'; ctx.font = '500 11px Inter'; ctx.textAlign = 'center';
      ctx.fillText(n.label, n.x, n.y + n.r + 16);
    });
    ctx.restore();
  }

  canvas.addEventListener('mousedown', e => { dragging = true; lastMouse = { x: e.clientX, y: e.clientY }; });
  canvas.addEventListener('mousemove', e => { if (dragging && lastMouse) { offsetX += e.clientX - lastMouse.x; offsetY += e.clientY - lastMouse.y; lastMouse = { x: e.clientX, y: e.clientY }; draw(); }});
  canvas.addEventListener('mouseup', () => { dragging = false; });
  canvas.addEventListener('click', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left - offsetX) / scale, my = (e.clientY - rect.top - offsetY) / scale;
    const hit = nodes.find(n => Math.hypot(mx - n.x, my - n.y) < n.r + 5);
    if (hit) {
      document.getElementById('node-detail-content').innerHTML = `
        <div class="text-label" style="margin-bottom:var(--space-xs);">${hit.type}</div>
        <h3 class="text-h3">${hit.label}</h3>
        <div class="divider divider--sm"></div>
        <div class="text-body-sm text-muted">Node ID: ${hit.id}<br/>Connections: ${edges.filter(e => e.includes(hit.id)).length}</div>
      `;
    }
  });

  document.getElementById('graph-zoom-in')?.addEventListener('click', () => { scale *= 1.2; draw(); });
  document.getElementById('graph-zoom-out')?.addEventListener('click', () => { scale /= 1.2; draw(); });
  document.getElementById('graph-reset')?.addEventListener('click', () => { scale = 1; offsetX = 0; offsetY = 0; draw(); });

  draw();
}
