// ─── State ───
var data = null;
var viewPath = [];
var TOOLTIP = document.getElementById('tooltip');
var TT_IN   = document.getElementById('tt-in');
var TT_OUT  = document.getElementById('tt-out');
var TT_DIM  = document.getElementById('tt-dim');
var ERR_MSG = document.getElementById('error-msg');

function showError(msg) {
  ERR_MSG.textContent = 'Error: ' + msg;
}

// ─── Colors ───
var C = {
  embed: '#607d8b',  layer: '#42a5f5',  self_attn: '#66bb6a',
  mlp: '#ffa726',    norm: '#ab47bc',   proj: '#90caf9',
  t_norm: '#ce93d8', t_mlp: '#ffcc80',  t_sattn: '#a5d6a7',
  lm_head: '#607d8b', adder: '#ffee58', adder_text: '#4e342e',
  residual: '#78909c', arrow: '#78909c',
};

// ─── Enrich ───
function enrichNode(node, hiddenSize) {
  if (node.type === 'tensor') return;
  if (node.type === 'module' || node.type === 'layer') {
    node.input_dim = hiddenSize;
    node.output_dim = hiddenSize;
    if (node.children) node.children.forEach(function(c) { enrichNode(c, hiddenSize); });
  }
}

// ─── SVG builders ───
function S(tag, attrs) {
  var el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (var k in attrs) el.setAttribute(k, String(attrs[k]));
  return el;
}

function makeBlock(opts) {
  var g = S('g', {'class': 'block'});
  var x = opts.x, y = opts.y, w = opts.w, h = opts.h;
  var node = opts.node;

  if (node && node.type === 'tensor' && node.input_dim !== node.output_dim) {
    var ratio = node.output_dim / node.input_dim;
    var topW, botW;
    if (ratio > 1) {
      topW = Math.max(w * 0.3, w / ratio);
      botW = w;
    } else {
      topW = w;
      botW = Math.max(w * 0.3, w * ratio);
    }
    var topX = x + (w - topW) / 2;
    var botX = x + (w - botW) / 2;
    g.appendChild(S('path', {
      d: 'M' + topX + ',' + y +
         ' L' + (topX + topW) + ',' + y +
         ' L' + (botX + botW) + ',' + (y + h) +
         ' L' + botX + ',' + (y + h) + ' Z',
      fill: opts.color, stroke: '#000', 'stroke-width': 1, 'stroke-linejoin': 'round'
    }));
  } else {
    g.appendChild(S('rect', {
      x: x, y: y, width: w, height: h,
      rx: 6, ry: 6, fill: opts.color, stroke: '#000', 'stroke-width': 1
    }));
  }

  var t = S('text', {
    x: x + w/2, y: y + h/2 + 1,
    'text-anchor': 'middle', 'dominant-baseline': 'middle',
    fill: '#fff', 'font-size': 12, 'font-weight': 'bold'
  });
  t.textContent = opts.label;
  g.appendChild(t);

  if (node && opts.onClick) {
    g.addEventListener('click', function(ev) { ev.stopPropagation(); opts.onClick(node); });
  }
  if (node) {
    g.addEventListener('mouseenter', function(ev) { showTooltip(ev, node); });
    g.addEventListener('mousemove',  function(ev) { moveTooltip(ev); });
    g.addEventListener('mouseleave', hideTooltip);
  }
  return g;
}

// ─── Unified connector ───
// points = [startX, startY, ...chain...]  where the chain alternates between
// V (y-coordinate) and H (x-coordinate) segments. The first segment after
// the start is V by default; set start_horizontal: true to begin with H.
function drawConnector(parent, points, opts, width) {
  width = width || 2;
  opts = opts || {};
  var dashed = opts.dashed || false;
  var color  = opts.color || (dashed ? C.residual : C.arrow);
  var horizontal = opts.start_horizontal || false;

  var d = 'M' + points[0] + ',' + points[1];
  for (var i = 2; i < points.length; i++) {
    d += (horizontal ? 'H' : 'V') + points[i];
    horizontal = !horizontal;
  }

  var dim = Math.round(width * 100);
  var path = S('path', {
    d: d, stroke: color, 'stroke-width': width, fill: 'none',
    'class': 'connector', 'stroke-linejoin': 'round',
    'stroke-dasharray': dashed ? '6,4' : 'none',
    'data-dim': dim
  });
  path.addEventListener('mouseenter', function(ev) { showConnectorTooltip(ev, dim); });
  path.addEventListener('mousemove',  function(ev) { moveTooltip(ev); });
  path.addEventListener('mouseleave', hideTooltip);
  parent.appendChild(path);
}

// ─── Tooltip ───
function showTooltip(ev, node) {
  if (node && node.input_dim != null && node.output_dim != null) {
    TT_IN.textContent = node.input_dim;
    TT_OUT.textContent = node.output_dim;
    TT_IN.parentElement.style.display = 'block';
    TT_OUT.parentElement.style.display = 'block';
    TT_DIM.parentElement.style.display = 'none';
    TOOLTIP.style.display = 'block';
    TOOLTIP.style.left = (ev.pageX + 14) + 'px';
    TOOLTIP.style.top  = (ev.pageY - 24) + 'px';
  }
}
function moveTooltip(ev) {
  TOOLTIP.style.left = (ev.pageX + 14) + 'px';
  TOOLTIP.style.top  = (ev.pageY - 24) + 'px';
}
function hideTooltip() { TOOLTIP.style.display = 'none'; }

function showConnectorTooltip(ev, dim) {
  TT_DIM.textContent = dim;
  TT_IN.parentElement.style.display = 'none';
  TT_OUT.parentElement.style.display = 'none';
  TT_DIM.parentElement.style.display = 'block';
  TOOLTIP.style.display = 'block';
  TOOLTIP.style.left = (ev.pageX + 14) + 'px';
  TOOLTIP.style.top  = (ev.pageY - 24) + 'px';
}

// ─── Render: Top Level ───
function renderTopLevel(svg, nodes) {
  var blockW = 120, blockH = 36, gap = 4;
  var totalH = nodes.length * (blockH + gap) + 60;
  var W = 280;
  svg.setAttribute('width', W);
  svg.setAttribute('height', totalH);
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + totalH);

  var g = S('g');
  var cx = W / 2 - blockW / 2;
  var cy = 20, prevCY = null, prevNode = null;

  nodes.forEach(function(node) {
    var color = C.embed;
    if (node.type === 'layer')            color = C.layer;
    else if (node.name === 'final_norm')  color = C.norm;
    else if (node.name === 'lm_head')     color = C.lm_head;

    var label = node.name;
    if (node.type === 'layer') label = 'Layer ' + node.index;

    g.appendChild(makeBlock({
      x: cx, y: cy, w: blockW, h: blockH, color: color, label: label, node: node,
      onClick: function(n) { if (n.type === 'layer') zoomTo([n.index]); }
    }));

    if (prevCY != null) {
      var mx = cx + blockW / 2;
      var connW = prevNode.output_dim / 100;
      drawConnector(g, [mx, prevCY + blockH, cy], {}, connW);
    }
    prevCY = cy;
    prevNode = node;
    cy += blockH + gap;
  });

  svg.appendChild(g);
}

// ─── Render: Layer Internal ───
function renderLayer(svg, layerNode) {
  var W = 740, H = 700;
  svg.setAttribute('width', W);
  svg.setAttribute('height', H);
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);

  var g = S('g'), cx = W/2;
  var hidden = layerNode.input_dim;
  var bw = 170, bh = 44, gap = 44;
  var resX = 50;

  var inputLN  = layerNode.children.find(function(c) { return c.name === 'input_layernorm'; });
  var selfAttn = layerNode.children.find(function(c) { return c.name === 'self_attn'; });
  var postLN   = layerNode.children.find(function(c) { return c.name === 'post_attention_layernorm'; });
  var mlp      = layerNode.children.find(function(c) { return c.name === 'mlp'; });

  var cy = 30;

  var inLbl = S('text', {x: cx, y: cy, 'text-anchor': 'middle', fill: '#888', 'font-size': 13});
  inLbl.textContent = 'input  (' + hidden + ')';
  g.appendChild(inLbl);
  cy += 15;
  var inputTop = cy;

  g.appendChild(makeBlock({x: cx - bw/2, y: cy, w: bw, h: bh, color: C.norm, label: 'input_layernorm', node: inputLN}));
  var ln1Bot = cy + bh;
  cy += bh + gap;

  g.appendChild(makeBlock({x: cx - bw/2, y: cy, w: bw, h: bh, color: C.self_attn, label: 'self_attn', node: selfAttn, onClick: function() { zoomTo([viewPath[0], 'self_attn']); }}));
  var saBot = cy + bh;
  cy += bh + gap;

  var add1CY = cy + 12;
  g.appendChild(S('circle', {cx: cx, cy: add1CY, r: 12, fill: C.adder, stroke: '#bdbdbd', 'stroke-width': 1}));
  var a1t = S('text', {x: cx, y: add1CY + 3, 'text-anchor': 'middle', fill: C.adder_text, 'font-size': 15, 'font-weight': 'bold'});
  a1t.textContent = '+'; g.appendChild(a1t);

  var resW = hidden / 100;
  drawConnector(g, [resX, inputTop, add1CY, cx - 12], {dashed: true}, resW);
  var r1 = S('text', {x: resX - 8, y: (inputTop + add1CY)/2 + 4, 'text-anchor': 'end', fill: C.residual, 'font-size': 11, 'font-style': 'italic'});
  r1.textContent = 'residual'; g.appendChild(r1);

  cy += 28;

  g.appendChild(makeBlock({x: cx - bw/2, y: cy, w: bw, h: bh, color: C.norm, label: 'post_attn_layernorm', node: postLN}));
  var ln2Bot = cy + bh;
  cy += bh + gap;

  g.appendChild(makeBlock({x: cx - bw/2, y: cy, w: bw, h: bh, color: C.mlp, label: 'mlp', node: mlp, onClick: function() { zoomTo([viewPath[0], 'mlp']); }}));
  var mlpBot = cy + bh;
  cy += bh + gap;

  var add2CY = cy + 12;
  g.appendChild(S('circle', {cx: cx, cy: add2CY, r: 12, fill: C.adder, stroke: '#bdbdbd', 'stroke-width': 1}));
  var a2t = S('text', {x: cx, y: add2CY + 3, 'text-anchor': 'middle', fill: C.adder_text, 'font-size': 15, 'font-weight': 'bold'});
  a2t.textContent = '+'; g.appendChild(a2t);

  drawConnector(g, [resX, add1CY, add2CY, cx - 12], {dashed: true}, resW);
  var r2 = S('text', {x: resX - 8, y: (add1CY + add2CY)/2 + 4, 'text-anchor': 'end', fill: C.residual, 'font-size': 11, 'font-style': 'italic'});
  r2.textContent = 'residual'; g.appendChild(r2);

  var vw = hidden / 100;
  drawConnector(g, [cx, ln1Bot,    saBot - bh], {}, vw);
  drawConnector(g, [cx, saBot,     add1CY - 12], {}, vw);
  drawConnector(g, [cx, add1CY+12, ln2Bot - bh], {}, vw);
  drawConnector(g, [cx, ln2Bot,    mlpBot - bh], {}, vw);
  drawConnector(g, [cx, mlpBot,    add2CY - 12], {}, vw);

  var outLblY = add2CY + 24;
  var outLbl = S('text', {x: cx, y: outLblY, 'text-anchor': 'middle', fill: '#888', 'font-size': 13});
  outLbl.textContent = 'output  (' + hidden + ')';
  g.appendChild(outLbl);
  drawConnector(g, [cx, add2CY+12, outLblY - 12], {}, vw);

  svg.appendChild(g);
}

// ─── Render: Self-Attention Internals ───
function renderSelfAttn(svg, moduleNode) {
  var W = 750, H = 380;
  svg.setAttribute('width', W);
  svg.setAttribute('height', H);
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);

  var g = S('g'), cx = W/2;
  var children = moduleNode.children;
  var qProj = children.find(function(c) { return c.name === 'q_proj'; });
  var kProj = children.find(function(c) { return c.name === 'k_proj'; });
  var vProj = children.find(function(c) { return c.name === 'v_proj'; });
  var oProj = children.find(function(c) { return c.name === 'o_proj'; });
  var qNorm = children.find(function(c) { return c.name === 'q_norm'; });
  var kNorm = children.find(function(c) { return c.name === 'k_norm'; });

  var bw = 124, bh = 34, gapY = 24;
  var bX = [cx - 210, cx, cx + 210];
  var projs = [qProj, kProj, vProj];
  var norms = [qNorm, kNorm, null];

  var cy = 25;
  var inLbl = S('text', {x: cx, y: cy, 'text-anchor': 'middle', fill: '#888', 'font-size': 12});
  inLbl.textContent = 'input (from layernorm)'; g.appendChild(inLbl);
  cy = 48;

  var projTop = cy;
  var inDimW = moduleNode.input_dim / 100;
  bX.forEach(function(bx, i) {
    g.appendChild(makeBlock({x: bx - bw/2, y: cy, w: bw, h: bh, color: C.t_sattn, label: projs[i].name, node: projs[i]}));
    drawConnector(g, [cx, projTop, projTop + 10, bx, cy], {arrow: false}, inDimW);
  });
  cy += bh + gapY;

  var normTop = cy;
  bX.forEach(function(bx, i) {
    if (norms[i]) {
      g.appendChild(makeBlock({x: bx - bw/2, y: cy, w: bw - 24, h: bh, color: C.norm, label: norms[i].name, node: norms[i]}));
      drawConnector(g, [bx, normTop - gapY, cy], {}, projs[i].output_dim / 100);
    }
  });
  cy += bh + gapY;

  drawConnector(g, [bX[2], projTop + bh, cy], {arrow: false}, vProj.output_dim / 100);

  var sdpaW = 220, sdpaH = 42;
  g.appendChild(makeBlock({x: cx - sdpaW/2, y: cy, w: sdpaW, h: sdpaH, color: '#607d8b', label: 'Scaled Dot-Product Attention', node: null}));
  for (var i = 0; i < 3; i++) {
    var srcY = norms[i] ? cy - gapY : projTop + bh;
    var dim = norms[i] ? norms[i].output_dim : projs[i].output_dim;
    drawConnector(g, [bX[i], srcY, srcY + 10, cx, cy], {arrow: false}, dim / 100);
  }
  cy += sdpaH + gapY;

  g.appendChild(makeBlock({x: cx - bw/2, y: cy, w: bw, h: bh, color: C.proj, label: oProj.name, node: oProj}));
  drawConnector(g, [cx, cy - gapY, cy], {}, qProj.output_dim / 100);
  cy += bh + 16;

  var outLbl = S('text', {x: cx, y: cy, 'text-anchor': 'middle', fill: '#888', 'font-size': 12});
  outLbl.textContent = 'output  (' + oProj.output_dim + ')'; g.appendChild(outLbl);

  svg.appendChild(g);
}

// ─── Render: MLP Internals ───
function renderMLP(svg, moduleNode) {
  var W = 660, H = 700;
  svg.setAttribute('width', W);
  svg.setAttribute('height', H);
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);

  var g = S('g'), cx = W/2;
  var children = moduleNode.children;
  var gate = children.find(function(c) { return c.name === 'gate_proj'; });
  var up   = children.find(function(c) { return c.name === 'up_proj'; });
  var down = children.find(function(c) { return c.name === 'down_proj'; });

  var bw = 130, bh = 34, gapY = 80, gapX = 100;
  var lX = cx - gapX, rX = cx + gapX;

  var cy = 24, inLblH = 24;
  var inLbl = S('text', {x: cx, y: cy, 'text-anchor': 'middle', fill: '#888', 'font-size': 12});
  inLbl.textContent = 'input (from layernorm)'; g.appendChild(inLbl);
  cy = cy + inLblH;
  drawConnector(g, [cx, cy, cy + gapY/2, lX, cy + gapY], {arrow: false}, gate.input_dim / 100);
  drawConnector(g, [cx, cy, cy + gapY/2, rX, cy + gapY], {arrow: false}, up.input_dim / 100);
  cy += gapY;

  g.appendChild(makeBlock({x: lX - bw/2, y: cy, w: bw, h: bh, color: C.t_mlp, label: 'gate_proj', node: gate}));
  g.appendChild(makeBlock({x: rX - bw/2, y: cy, w: bw, h: bh, color: C.t_mlp, label: 'up_proj',   node: up}));

  var projBot = cy + bh;
  cy += bh + gapY;

  var siluW = 80, siluH = 34;
  g.appendChild(makeBlock({x: lX - siluW/2, y: cy, w: siluW, h: siluH, color: '#607d8b', label: 'SiLU', node: null}));
  drawConnector(g, [lX, projBot, cy], {}, gate.output_dim / 100);
  var siluBot = cy + siluH;
  cy += siluH + gapY;

  drawConnector(g, [rX, projBot, cy - gapY + siluH], {arrow: false}, up.output_dim / 100);

  var mulX = cx, mulCY = cy + 12;
  g.appendChild(S('circle', {cx: mulX, cy: mulCY, r: 13, fill: C.adder, stroke: '#bdbdbd', 'stroke-width': 1}));
  var mt = S('text', {x: mulX, y: mulCY + 3, 'text-anchor': 'middle', fill: C.adder_text, 'font-size': 15, 'font-weight': 'bold'});
  mt.textContent = '\u00D7'; g.appendChild(mt);

  drawConnector(g, [lX, siluBot, mulCY, mulX - 13], {},width = gate.output_dim / 100);
  drawConnector(g, [rX, siluBot, mulCY, mulX + 13], {},width = up.output_dim / 100);
  cy += gapY;

  g.appendChild(makeBlock({x: cx - bw/2, y: cy, w: bw, h: bh, color: C.t_mlp, label: 'down_proj', node: down}));
  drawConnector(g, [mulX, mulCY + 13, cy], {}, down.input_dim / 100);
  cy += bh;
  drawConnector(g, [mulX, cy, cy + gapY], {}, down.output_dim / 100);
  cy += gapY + 20;

  var outLbl = S('text', {x: cx, y: cy, 'text-anchor': 'middle', fill: '#888', 'font-size': 12});
  outLbl.textContent = 'output  (' + down.output_dim + ')'; g.appendChild(outLbl);

  svg.appendChild(g);
}

// ─── Main Render ───
function render() {
  var svg = document.getElementById('main-svg');
  svg.innerHTML = '';

  if (!data) return;

  try {
    if (viewPath.length === 0) {
      renderTopLevel(svg, data.nodes);
    } else if (viewPath.length === 1) {
      var layerNode = null;
      for (var j = 1; j < data.nodes.length - 1; j++) {
        if (data.nodes[j].index === viewPath[0]) { layerNode = data.nodes[j]; break; }
      }
      if (layerNode) renderLayer(svg, layerNode);
    } else if (viewPath.length === 2) {
      var ln = null;
      for (var k = 1; k < data.nodes.length - 1; k++) {
        if (data.nodes[k].index === viewPath[0]) { ln = data.nodes[k]; break; }
      }
      if (!ln) return;
      var mod = null;
      for (var m = 0; m < ln.children.length; m++) {
        if (ln.children[m].name === viewPath[1]) { mod = ln.children[m]; break; }
      }
      if (!mod) return;
      if (viewPath[1] === 'self_attn') renderSelfAttn(svg, mod);
      else if (viewPath[1] === 'mlp') renderMLP(svg, mod);
    }
  } catch(e) {
    showError(e.message || String(e));
  }

  updateBreadcrumb();
}

// ─── Breadcrumb ───
function updateBreadcrumb() {
  var bc = document.getElementById('breadcrumb');
  var html = '<span class="link" onclick="zoomTo([])">Model</span>';
  if (viewPath.length >= 1) {
    html += '<span class="sep">&gt;</span>';
    html += '<span class="link" onclick="zoomTo([' + viewPath[0] + '])">Layer ' + viewPath[0] + '</span>';
  }
  if (viewPath.length >= 2) {
    html += '<span class="sep">&gt;</span>';
    html += '<span class="cur">' + viewPath[1] + '</span>';
  }
  bc.innerHTML = html;
}

// ─── Zoom navigation ───
function zoomTo(path) {
  viewPath = path;
  ERR_MSG.textContent = '';
  render();
}

document.getElementById('main-svg').addEventListener('click', function(ev) {
  if (ev.target.id === 'main-svg' && viewPath.length > 0) zoomTo(viewPath.slice(0, -1));
});

document.getElementById('svg-container').addEventListener('click', function(ev) {
  if (ev.target.id === 'svg-container' && viewPath.length > 0) zoomTo(viewPath.slice(0, -1));
});

// ─── Fetch & Init ───
function loadArchitecture(model) {
  fetch('/api/architecture?model=' + model)
    .then(function(resp) {
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      return resp.json();
    })
    .then(function(json) {
      data = json;
      if (data.hidden_size) {
        data.nodes.forEach(function(n) { enrichNode(n, data.hidden_size); });
      }
      viewPath = [];
      ERR_MSG.textContent = '';
      render();
    })
    .catch(function(err) {
      showError(err.message || String(err));
    });
}

document.getElementById('model-select').addEventListener('change', function(e) {
  loadArchitecture(e.target.value);
});

loadArchitecture('4b');
