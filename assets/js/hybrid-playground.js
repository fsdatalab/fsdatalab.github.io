/*
 * Interactive playgrounds for "Supporting Hybrid Models in Quail".
 *
 * Every per-model number below was produced by running Quail itself
 * (fsdatalab/quail, commit 2a1fe4d2) for one NVIDIA H100 SXM: the cost-component
 * coefficients come from quail.cost.dense_decoder_cost, the pass size from
 * quail.cost.budgets.chunk_budget, and the retained-document pool from
 * budgets.hybrid_pool_pages (hybrid) or budgets.arena_pages (dense). Each model's
 * `test` entry is Quail's speed_of_light for a fixed work record and is checked
 * against this file's roofline in tests.
 */
(function (root) {
  'use strict';

  const DATA = {"page":16,"beta":3350000000000.0,"models":{"qwen3.5-4b-bf16":{"name":"qwen3.5-4b-bf16","label":"Qwen3.5-4B (bf16)","hybrid":true,"chunk":116508,"kappa":32768.0,"ckpt":51511296,"layers":32,"components":[{"name":"attn_proj","pi":857000000000000.0,"ft":2608332800.0,"fp":0.0,"bp":2608332800.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"mlp","pi":857000000000000.0,"ft":4529848320.0,"fp":0.0,"bp":4529848320.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"attention","pi":989500000000000.0,"ft":0.0,"fp":131072.0,"bp":0.0,"bw":32768.0,"br":32768.0,"bs":0.0},{"name":"gdn_core","pi":989500000000000.0,"ft":140509184.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"state","pi":989500000000000.0,"ft":0.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":51511296.0},{"name":"conv","pi":67000000000000.0,"ft":1572864.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":0.0}],"pool":35548299264.0,"test":{"work":[1200000.0,800000000.0,1200000.0,70000.0,985,1017],"sol":10.330444396461443}},"qwen3.5-4b-fp8":{"name":"qwen3.5-4b-fp8","label":"Qwen3.5-4B (fp8)","hybrid":true,"chunk":116508,"kappa":32768.0,"ckpt":51511296,"layers":32,"components":[{"name":"attn_proj","pi":1629000000000000.0,"ft":587202560.0,"fp":0.0,"bp":293601280.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"mlp","pi":1629000000000000.0,"ft":4529848320.0,"fp":0.0,"bp":2264924160.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"attention","pi":989500000000000.0,"ft":0.0,"fp":131072.0,"bp":0.0,"bw":32768.0,"br":32768.0,"bs":0.0},{"name":"gdn_proj","pi":857000000000000.0,"ft":2021130240.0,"fp":0.0,"bp":2021130240,"bw":0,"br":0,"bs":0},{"name":"gdn_core","pi":989500000000000.0,"ft":140509184.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"state","pi":989500000000000.0,"ft":0.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":51511296.0},{"name":"conv","pi":67000000000000.0,"ft":1572864.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":0.0}],"pool":38070124544.0,"test":{"work":[1200000.0,800000000.0,1200000.0,70000.0,985,1017],"sol":6.934845565162058}},"qwen3.5-27b-fp8":{"name":"qwen3.5-27b-fp8","label":"Qwen3.5-27B (fp8)","hybrid":true,"chunk":61680,"kappa":65536.0,"ckpt":153944064,"layers":64,"components":[{"name":"attn_proj","pi":1629000000000000.0,"ft":14475591680.0,"fp":0.0,"bp":7237795840.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"mlp","pi":1629000000000000.0,"ft":34225520640.0,"fp":0.0,"bp":17112760320.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"attention","pi":989500000000000.0,"ft":0.0,"fp":393216.0,"bp":0.0,"bw":65536.0,"br":65536.0,"bs":0.0},{"name":"gdn_core","pi":989500000000000.0,"ft":421527552.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"state","pi":989500000000000.0,"ft":0.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":153944064.0},{"name":"conv","pi":67000000000000.0,"ft":3932160.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":0.0}],"pool":12696158208.0,"test":{"work":[1200000.0,800000000.0,1200000.0,70000.0,985,1017],"sol":36.86712627655859}},"qwen3.6-35b-a3b-fp8":{"name":"qwen3.6-35b-a3b-fp8","label":"Qwen3.6-35B-A3B (fp8)","hybrid":true,"chunk":32768,"kappa":20480.0,"ckpt":64389120,"layers":40,"components":[{"name":"attn_proj","pi":1629000000000000.0,"ft":2566389760.0,"fp":0.0,"bp":1283194880.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"mlp","pi":1629000000000000.0,"ft":2264924160.0,"fp":0.0,"bp":32338083840.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"attention","pi":989500000000000.0,"ft":0.0,"fp":163840.0,"bp":0.0,"bw":20480.0,"br":20480.0,"bs":0.0},{"name":"router","pi":857000000000000.0,"ft":41943040.0,"fp":0.0,"bp":41943040,"bw":0,"br":0,"bs":0},{"name":"gdn_core","pi":989500000000000.0,"ft":175636480.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"state","pi":989500000000000.0,"ft":0.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":64389120.0},{"name":"conv","pi":67000000000000.0,"ft":1966080.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":0.0}],"pool":25766789120.0,"test":{"work":[1200000.0,800000000.0,1200000.0,70000.0,985,1017],"sol":4.0368652099644375}},"qwen3.8-27b-fp8":{"name":"qwen3.8-27b-fp8","label":"Qwen3.8-27B (fp8)","hybrid":true,"chunk":61680,"kappa":65536.0,"ckpt":153944064,"layers":64,"components":[{"name":"attn_proj","pi":1629000000000000.0,"ft":14475591680.0,"fp":0.0,"bp":7237795840.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"mlp","pi":1629000000000000.0,"ft":34225520640.0,"fp":0.0,"bp":17112760320.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"attention","pi":989500000000000.0,"ft":0.0,"fp":393216.0,"bp":0.0,"bw":65536.0,"br":65536.0,"bs":0.0},{"name":"gdn_core","pi":989500000000000.0,"ft":421527552.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"state","pi":989500000000000.0,"ft":0.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":153944064.0},{"name":"conv","pi":67000000000000.0,"ft":3932160.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":0.0}],"pool":12696158208.0,"test":{"work":[1200000.0,800000000.0,1200000.0,70000.0,985,1017],"sol":36.86712627655859}},"lfm2-2.6b-bf16":{"name":"lfm2-2.6b-bf16","label":"LFM2-2.6B (bf16)","hybrid":true,"chunk":99864,"kappa":16384.0,"ckpt":180224,"layers":30,"components":[{"name":"attn_proj","pi":857000000000000.0,"ft":905969664.0,"fp":0.0,"bp":905969664.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"mlp","pi":857000000000000.0,"ft":3963617280.0,"fp":0.0,"bp":3963617280.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"attention","pi":989500000000000.0,"ft":0.0,"fp":65536.0,"bp":0.0,"bw":16384.0,"br":16384.0,"bs":0.0},{"name":"state","pi":989500000000000.0,"ft":0.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":180224.0},{"name":"conv","pi":67000000000000.0,"ft":270336.0,"fp":0.0,"bp":0.0,"bw":0.0,"br":0.0,"bs":0.0}],"pool":51772391424.0,"test":{"work":[1200000.0,800000000.0,1200000.0,70000.0,985,1017],"sol":6.876492834790419}},"qwen3-4b-fp8":{"name":"qwen3-4b-fp8","label":"Qwen3-4B (fp8)","hybrid":false,"chunk":110376,"kappa":147456.0,"ckpt":0,"layers":36,"components":[{"name":"attn_proj","pi":1629000000000000.0,"ft":1887436800.0,"fp":0.0,"bp":943718400.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"mlp","pi":1629000000000000.0,"ft":5379194880.0,"fp":0.0,"bp":2689597440.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"attention","pi":989500000000000.0,"ft":0.0,"fp":589824.0,"bp":0.0,"bw":147456.0,"br":147456.0,"bs":0.0}],"pool":53414461440.0,"test":{"work":[1200000.0,800000000.0,1200000.0,70000.0,985,1017],"sol":5.829817809921301}},"qwen3-32b-fp8":{"name":"qwen3-32b-fp8","label":"Qwen3-32B (fp8)","hybrid":false,"chunk":41943,"kappa":262144.0,"ckpt":0,"layers":64,"components":[{"name":"attn_proj","pi":1629000000000000.0,"ft":12079595520.0,"fp":0.0,"bp":6039797760.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"mlp","pi":1629000000000000.0,"ft":50331648000.0,"fp":0.0,"bp":25165824000.0,"bw":0.0,"br":0.0,"bs":0.0},{"name":"attention","pi":989500000000000.0,"ft":0.0,"fp":2097152.0,"bp":0.0,"bw":262144.0,"br":262144.0,"bs":0.0}],"pool":27883732992.0,"test":{"work":[1200000.0,800000000.0,1200000.0,70000.0,985,1017],"sol":47.67065795645438}}}};
  const GB = 1e9;
  const MIB = 1048576;
  const PAGE = DATA.page;
  const MODELS = DATA.models;
  const HYBRIDS = Object.keys(MODELS).filter((k) => MODELS[k].hybrid);
  const ALL = Object.keys(MODELS);

  // ── formatting ─────────────────────────────────────────────────────────
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const int = (n) => Math.round(n).toLocaleString('en-US');
  const dec = (n, d) => n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  const fmtTime = (s) => {
    if (s === 0) return '0 s';
    if (s >= 1) return dec(s, 3) + ' s';
    if (s >= 1e-3) return dec(s * 1e3, 1) + ' ms';
    return dec(s * 1e6, 1) + ' µs';
  };
  const fmtSci = (x) => {
    if (x === 0) return '0';
    const e = Math.floor(Math.log10(x));
    return dec(x / Math.pow(10, e), 2) + '×10<sup>' + e + '</sup>';
  };
  const numField = (label, attrs) => '<label class="pg-field"><span>' + label + '</span><input class="pg-num" type="number" inputmode="decimal" step="any" ' + attrs + '></label>';
  const modelSelect = (attr, keys, sel) =>
    '<select class="pg-select" ' + attr + ' aria-label="Model">' +
    keys.map((k) => '<option value="' + k + '"' + (k === sel ? ' selected' : '') + '>' + esc(MODELS[k].label) + (MODELS[k].hybrid ? '' : ' (dense)') + '</option>').join('') +
    '</select>';

  // ── Checkpoint Playground ─────────────────────────────────────────────
  const modelDefaults = (k) => ({ pool: MODELS[k].pool / GB, ckpt: MODELS[k].ckpt / MIB, kv: MODELS[k].kappa });
  // A retained document holds ceil(l/page) pages of KV plus one checkpoint (hybrids).
  const docBytes = (m, l, page) => Math.ceil(l / page) * page * m.kv + m.ckpt * MIB;
  const slots = (m, l, page) => Math.floor((m.pool * GB) / docBytes(m, l, page));

  const L_MIN = 100;
  const L_MAX = 32000;
  const toPos = (l) => (Math.log(l) - Math.log(L_MIN)) / (Math.log(L_MAX) - Math.log(L_MIN));
  const fromPos = (p) => Math.round(Math.exp(Math.log(L_MIN) + p * (Math.log(L_MAX) - Math.log(L_MIN))));

  function modelBlock(side, key) {
    const d = modelDefaults(key);
    return (
      '<div class="hp-model"><div class="hp-model-head"><span class="pg-h-small">' + (side === 'a' ? 'Model' : 'Compare with') + '</span>' +
      modelSelect('data-sel="' + side + '"', side === 'a' ? HYBRIDS : ALL, key) + '</div>' +
      '<div class="hp-fields">' +
      numField('Pool (GB)', 'min="0.001" data-f="' + side + '.pool" value="' + d.pool + '"') +
      numField('Checkpoint (MiB)', 'min="0" data-f="' + side + '.ckpt" value="' + d.ckpt + '"') +
      numField('KV (B/token)', 'min="1" data-f="' + side + '.kv" value="' + d.kv + '"') +
      '</div></div>'
    );
  }

  function capacityPlot(A, B, l, page) {
    const W = 640, Ht = 210, padL = 44, padR = 14, padT = 10, padB = 28;
    const ymin = 1, ymax = 10000;
    const lx = (v) => padL + toPos(v) * (W - padL - padR);
    const ly = (v) => padT + (1 - (Math.log(Math.min(ymax, Math.max(ymin, v))) - Math.log(ymin)) / (Math.log(ymax) - Math.log(ymin))) * (Ht - padT - padB);
    const pts = [];
    for (let i = 0; i <= 80; i++) pts.push(fromPos(i / 80));
    const line = (m) => pts.map((v, i) => (i ? 'L' : 'M') + lx(v).toFixed(1) + ',' + ly(slots(m, v, page)).toFixed(1)).join(' ');
    let svg = '<svg class="pg-plot-svg" viewBox="0 0 ' + W + ' ' + Ht + '" role="img" aria-label="Documents retained as document length grows">';
    [1, 10, 100, 1000, 10000].forEach((y) => {
      svg += '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + ly(y).toFixed(1) + '" y2="' + ly(y).toFixed(1) + '" class="pg-axis"/>';
      svg += '<text x="' + (padL - 6) + '" y="' + (ly(y) + 3.5).toFixed(1) + '" text-anchor="end" class="pg-axis-label">' + int(y) + '</text>';
    });
    [100, 1000, 10000].forEach((x) => {
      svg += '<text x="' + lx(x).toFixed(1) + '" y="' + (Ht - 8) + '" text-anchor="middle" class="pg-axis-label">' + int(x) + '</text>';
    });
    svg += '<path d="' + line(B) + '" class="hp-line-d"/><path d="' + line(A) + '" class="hp-line-h"/>';
    const lc = Math.min(L_MAX, Math.max(L_MIN, l));
    svg += '<line x1="' + lx(lc).toFixed(1) + '" x2="' + lx(lc).toFixed(1) + '" y1="' + padT + '" y2="' + (Ht - padB) + '" class="hp-marker"/>';
    svg += '<circle cx="' + lx(lc).toFixed(1) + '" cy="' + ly(slots(A, lc, page)).toFixed(1) + '" r="4.5" class="hp-dot-h"/>';
    svg += '<circle cx="' + lx(lc).toFixed(1) + '" cy="' + ly(slots(B, lc, page)).toFixed(1) + '" r="4.5" class="hp-dot-d"/>';
    return svg + '</svg>';
  }

  function initCapacity(el) {
    const DEF = { a: 'qwen3.5-4b-bf16', b: 'qwen3-4b-fp8' };
    el.innerHTML =
      '<div class="pg-bar"><span class="pg-title">Checkpoint Playground <span class="pg-sub">documents retained in HBM on one H100</span></span>' +
      '<div class="pg-bar-actions"><button type="button" class="pg-btn" data-action="reset">Reset</button></div></div>' +
      '<div class="pg-body">' +
      '<div class="hp-controls"><label class="pg-field hp-slider"><span>Document length (tokens)</span><input class="pg-range" type="range" min="0" max="1000" step="1" data-range aria-label="Document length, log scale"></label>' +
      numField('Tokens <i>ℓ</i>', 'min="1" data-f="len" value="1000"') + numField('Page (tokens)', 'min="1" data-f="page" value="' + PAGE + '"') + '</div>' +
      '<div class="hp-models">' + modelBlock('a', DEF.a) + modelBlock('b', DEF.b) + '</div>' +
      '<div class="pg-error" data-out="err" hidden></div>' +
      '<div class="pg-tiles hp-tiles">' +
      '<div class="pg-tile pg-tile-main"><span class="pg-tile-label" data-out="alabel"></span><span class="pg-tile-value" data-out="a"></span><span class="pg-verdict" data-out="vs"></span></div>' +
      '<div class="pg-tile"><span class="pg-tile-label" data-out="blabel"></span><span class="pg-tile-value" data-out="b"></span><span class="pg-verdict" data-out="bsub"></span></div></div>' +
      '<div class="hp-split"><div class="pg-label-row"><span class="pg-h-small">Pool of the first model</span><span class="pg-verdict" data-out="share"></span></div>' +
      '<div class="hp-bar"><span class="hp-bar-ckpt" data-out="barckpt"></span><span class="hp-bar-kv" data-out="barkv"></span></div>' +
      '<div class="hp-bar-key"><span><i class="hp-sw hp-sw-ckpt"></i>Checkpoints <b data-out="ckptgb"></b></span><span><i class="hp-sw hp-sw-kv"></i>KV <b data-out="kvgb"></b></span></div></div>' +
      '<div class="pg-plot"><div class="pg-plot-head"><span class="pg-h-small">Documents retained vs. document length</span>' +
      '<span class="pg-keys"><span><i class="hp-sw hp-sw-line-h"></i><span data-out="akey"></span></span><span><i class="hp-sw hp-sw-line-d"></i><span data-out="bkey"></span></span></span></div>' +
      '<div data-out="plot"></div><div class="pg-plot-msg" data-out="cross"></div></div></div>';
    const range = el.querySelector('[data-range]');
    const out = (k) => el.querySelector('[data-out="' + k + '"]');
    const f = {};
    el.querySelectorAll('[data-f]').forEach((i) => (f[i.dataset.f] = i));
    const sel = { a: el.querySelector('[data-sel="a"]'), b: el.querySelector('[data-sel="b"]') };
    const read = () => {
      const g = (k) => (f[k].value.trim() === '' ? NaN : Number(f[k].value));
      const mk = (s) => ({ pool: g(s + '.pool'), ckpt: g(s + '.ckpt'), kv: g(s + '.kv') });
      const A = mk('a'), B = mk('b'), len = Math.floor(g('len')), page = g('page');
      const ok = (m) => m.pool > 0 && m.ckpt >= 0 && m.kv > 0;
      if (!(len >= 1) || !(Number.isInteger(page) && page >= 1) || !ok(A) || !ok(B)) return null;
      return { A, B, len, page };
    };
    function render(from) {
      const r = read();
      out('err').hidden = !!r;
      if (!r) {
        out('err').textContent = 'Enter positive numbers in every field (checkpoint may be 0, page must be a whole number).';
        return;
      }
      const { A, B, len, page } = r;
      if (from !== 'range') range.value = Math.round(toPos(Math.min(L_MAX, Math.max(L_MIN, len))) * 1000);
      range.style.setProperty('--p', (Number(range.value) / 10).toFixed(1) + '%');
      range.style.setProperty('--fill', 'var(--red)');
      const an = MODELS[sel.a.value].label, bn = MODELS[sel.b.value].label;
      const sa = slots(A, len, page), sb = slots(B, len, page);
      out('alabel').textContent = an + ' retains';
      out('blabel').textContent = bn + ' retains';
      out('a').textContent = int(sa) + ' docs';
      out('b').textContent = int(sb) + ' docs';
      out('vs').textContent = sb > 0 ? dec(sa / sb, 2) + '× the other model'
        : sa > 0 ? 'the other model holds none' : 'neither model fits a document';
      out('vs').classList.toggle('is-gain', sa > sb);
      out('bsub').textContent = int(docBytes(B, len, page) / 1e6) + ' MB per document';
      const share = (A.ckpt * MIB) / docBytes(A, len, page);
      const ckptGB = (sa * A.ckpt * MIB) / GB;
      const frac = Math.min(100, (ckptGB / A.pool) * 100);
      out('share').textContent = A.ckpt > 0 ? 'checkpoint is ' + dec(share * 100, 0) + '% of each retained document' : 'no checkpoint';
      out('barckpt').style.width = frac.toFixed(2) + '%';
      out('barkv').style.width = (100 - frac).toFixed(2) + '%';
      out('ckptgb').textContent = dec(ckptGB, 2) + ' GB';
      out('kvgb').textContent = dec(Math.max(0, A.pool - ckptGB), 2) + ' GB';
      out('akey').textContent = an;
      out('bkey').textContent = bn;
      out('plot').innerHTML = capacityPlot(A, B, len, page);
      const difference = sa - sb;
      out('cross').textContent = difference === 0
        ? 'Both models hold the same number of documents at ' + int(len) + ' tokens.'
        : (difference > 0 ? an : bn) + ' holds ' + int(Math.abs(difference)) + ' more documents at ' + int(len) + ' tokens.';
    }
    const load = (s) => {
      const d = modelDefaults(sel[s].value);
      f[s + '.pool'].value = d.pool;
      f[s + '.ckpt'].value = d.ckpt;
      f[s + '.kv'].value = d.kv;
    };
    range.addEventListener('input', () => { f.len.value = fromPos(Number(range.value) / 1000); render('range'); });
    el.addEventListener('input', (e) => { if (e.target.dataset.f) render(e.target.dataset.f === 'len' ? 'num' : undefined); });
    ['a', 'b'].forEach((s) => sel[s].addEventListener('change', () => { load(s); render(); }));
    el.querySelector('[data-action="reset"]').addEventListener('click', () => {
      sel.a.value = DEF.a; sel.b.value = DEF.b; load('a'); load('b');
      f.len.value = 1000; f.page.value = PAGE; render();
    });
    render();
  }

  // ── Cost-Model Playground ─────────────────────────────────────────────
  const TERMS = {
    mlp: ['<i>T</i><sub>mlp</sub>', 'MLP'],
    attn_proj: ['<i>T</i><sub>proj</sub>', 'projections'],
    gdn_proj: ['<i>T</i><sub>gdn proj</sub>', 'GDN projections (bf16)'],
    router: ['<i>T</i><sub>router</sub>', 'MoE router'],
    gdn_core: ['<i>T</i><sub>gdn</sub>', 'gated delta rule'],
    attention: ['<i>T</i><sub>attn</sub>', 'full attention'],
    state: ['<i>T</i><sub>state</sub>', 'checkpoint movement'],
    conv: ['<i>T</i><sub>conv</sub>', 'short convolution'],
  };
  // work = {n, A, W, R, UW, UR}; K = passes.
  function roofline(m, w, K) {
    const terms = [];
    let total = 0;
    m.components.forEach((c) => {
      const F = c.ft * w.n + c.fp * w.A;
      const B = c.bp * K + c.bw * w.W + c.br * w.R + c.bs * (w.UW + w.UR);
      if (!F && !B) return;
      const tc = c.pi && F ? F / c.pi : 0;
      const tm = B / DATA.beta;
      const T = Math.max(tc, tm);
      total += T;
      terms.push({ name: c.name, F, B, pi: F ? c.pi : 0, tc, tm, T });
    });
    return { terms, total };
  }
  function docCost(m, kind, p, q) {
    const r = p + q;
    const w = kind === 'scan' ? { n: r, A: (r * (r + 1)) / 2, W: r, R: 0, UW: 1, UR: 0 } : { n: q, A: q * p + (q * (q + 1)) / 2, W: q, R: p, UW: 0, UR: 1 };
    return roofline(m, w, w.n / m.chunk).total;
  }
  function evalOrder(m, order, g) {
    const p = g.pre + g.len;
    let docs = g.N;
    const tot = { n: 0, A: 0, W: 0, R: 0, UW: 0, UR: 0 };
    const stages = order.map((f, j) => {
      let w;
      if (j === 0) {
        const r = p + f.q;
        const reuseCount = g.N - 1;
        const prefix = reuseCount ? g.pre + g.shared / reuseCount : 0;
        const read = reuseCount * prefix;
        const n = g.N * r - read;
        // Approximate shared prefixes by their mean length among later documents.
        const pairs = (g.N * r * (r + 1) - reuseCount * prefix * (prefix + 1)) / 2;
        w = { n, A: pairs, W: n, R: read, UW: g.N, UR: reuseCount };
      } else {
        w = { n: docs * f.q, A: docs * (f.q * p + (f.q * (f.q + 1)) / 2), W: docs * f.q, R: docs * p, UW: 0, UR: docs };
      }
      const st = { f, docsIn: docs, passed: docs * f.s, tokens: w.n };
      docs *= f.s;
      for (const k in tot) tot[k] += w[k];
      return st;
    });
    const K = Math.ceil(tot.n / m.chunk);
    const rf = roofline(m, tot, K);
    return { stages, work: tot, K, terms: rf.terms, total: rf.total };
  }
  function permutations(items) {
    if (items.length <= 1) return [items.slice()];
    const out = [];
    items.forEach((x, i) => permutations(items.slice(0, i).concat(items.slice(i + 1))).forEach((r) => out.push([x].concat(r))));
    return out;
  }

  const SALES = {
    N: 985, len: 1162, pre: 18, shared: 10853,
    filters: [{ name: 'Discount offered', s: 104 / 985, q: 78 }, { name: 'Customer commits', s: 33 / 985, q: 83 }],
  };
  const CHIP = ['#C41230', '#555555', '#8B2332', '#777777', '#A6192E', '#444444', '#9B555F', '#666666'];
  const freshState = () => {
    const st = { nextId: 1, model: 'qwen3.5-4b-bf16', N: SALES.N, len: SALES.len, pre: SALES.pre, shared: SALES.shared, filters: [] };
    SALES.filters.forEach((f) => st.filters.push(Object.assign({ id: st.nextId++ }, f)));
    return st;
  };
  function salesRow(f, idx, total) {
    const color = CHIP[(f.id - 1) % CHIP.length];
    return (
      '<li class="pg-row" data-id="' + f.id + '" style="--c:' + color + '">' +
      '<span class="pg-handle"><span class="pg-badge">' + (idx + 1) + '</span></span>' +
      '<input class="pg-name" type="text" data-field="name" value="' + esc(f.name) + '" maxlength="60" autocomplete="off" aria-label="Predicate">' +
      '<div class="pg-s"><input class="pg-range" type="range" min="0" max="1" step="0.005" value="' + f.s + '" data-field="s" aria-label="Selectivity">' +
      '<input class="pg-num" type="number" inputmode="decimal" min="0" max="1" step="any" data-field="s" value="' + f.s + '" aria-label="Selectivity value"></div>' +
      '<input class="pg-num pg-q" type="number" inputmode="numeric" min="1" step="1" data-field="q" value="' + f.q + '" aria-label="Instruction length in tokens">' +
      '<div class="pg-io"><span data-io></span><span class="pg-io-bar"><i class="pg-io-in"></i><i class="pg-io-out"></i></span></div>' +
      '<span class="pg-tools">' +
      '<button type="button" class="pg-icon" data-move="-1" aria-label="Move earlier"' + (idx === 0 ? ' disabled' : '') + '>&lsaquo;</button>' +
      '<button type="button" class="pg-icon" data-move="1" aria-label="Move later"' + (idx === total - 1 ? ' disabled' : '') + '>&rsaquo;</button>' +
      '<button type="button" class="pg-icon pg-remove" data-remove="1" aria-label="Remove filter"' + (total === 1 ? ' disabled' : '') + '>&times;</button></span></li>'
    );
  }

  function initSales(el) {
    let st = freshState();
    el.innerHTML =
      '<div class="pg-bar"><span class="pg-title">Cost-Model Playground <span class="pg-sub">speed-of-light estimate on one H100</span></span>' +
      '<div class="pg-bar-actions">' + modelSelect('data-model', ALL, st.model) +
      '<button type="button" class="pg-btn" data-action="reset">Reset</button><button type="button" class="pg-btn pg-btn-primary" data-action="apply">Apply ordering rule</button></div></div>' +
      '<div class="pg-body"><div class="pg-workload hp-work4">' +
      numField('Documents <i>N</i>', 'min="1" data-g="N" value="' + st.N + '"') +
      numField('Average doc length', 'min="1" data-g="len" value="' + st.len + '"') +
      numField('Preamble <i>q</i><sub>pre</sub>', 'min="0" data-g="pre" value="' + st.pre + '"') +
      numField('Shared prefix tokens', 'min="0" data-g="shared" value="' + st.shared + '"') + '</div>' +
      '<div class="pg-label-row"><span class="pg-h">Your filter order</span></div>' +
      '<div class="pg-cols" aria-hidden="true"><span></span><span>Predicate</span><span>Selectivity <i>s</i></span><span><i>q</i><sub>i</sub></span><span>Docs</span><span></span></div>' +
      '<ol class="pg-rows" data-rows aria-label="Filters in the order you entered them"></ol>' +
      '<div class="pg-library"><button type="button" class="pg-btn" data-action="add">+ Add a filter</button></div>' +
      '<div class="pg-results" data-results aria-live="polite"></div></div>';
    const rowsEl = el.querySelector('[data-rows]');
    const resEl = el.querySelector('[data-results]');
    const modelEl = el.querySelector('[data-model]');
    const num = (v) => (String(v).trim() === '' ? NaN : Number(v));
    const paint = (r) => r.style.setProperty('--p', (Number(r.value) * 100).toFixed(1) + '%');
    const drawRows = () => {
      rowsEl.innerHTML = st.filters.map((f, i) => salesRow(f, i, st.filters.length)).join('');
      rowsEl.querySelectorAll('.pg-range').forEach(paint);
    };
    function readAll() {
      const g = { N: num(st.N), len: num(st.len), pre: num(st.pre), shared: num(st.shared) };
      if (!(Number.isInteger(g.N) && g.N >= 1)) return { error: 'Documents must be a whole number of at least 1.' };
      if (!(g.len >= 1)) return { error: 'Average document length must be at least 1 token.' };
      if (!(g.pre >= 0)) return { error: 'The preamble must be 0 or more tokens.' };
      if (!(g.shared >= 0 && g.shared <= (g.N - 1) * g.len)) {
        return { error: 'Shared prefix tokens cannot exceed the document tokens after the first document.' };
      }
      const filters = [];
      for (const f of st.filters) {
        const s = num(f.s), q = num(f.q);
        const name = f.name.trim() || 'Filter ' + f.id;
        if (!(s >= 0 && s <= 1)) return { error: 'Selectivity of "' + name + '" must be between 0 and 1.' };
        if (!(Number.isInteger(q) && q >= 1)) return { error: 'Instruction length of "' + name + '" must be a whole number of tokens, 1 or more.' };
        filters.push({ id: f.id, name, s, q, color: CHIP[(f.id - 1) % CHIP.length] });
      }
      return { g, filters };
    }
    function update() {
      const r = readAll();
      if (r.error) { resEl.innerHTML = '<div class="pg-error">' + esc(r.error) + '</div>'; return; }
      const { g, filters } = r;
      const m = MODELS[st.model];
      const p = g.pre + g.len;
      filters.forEach((f) => {
        f.ask = docCost(m, 'ask', p, f.q);
        f.rank = f.s >= 1 ? Infinity : f.ask / (1 - f.s);
      });
      const ranked = filters.slice().sort((a, b) => a.rank - b.rank || filters.indexOf(a) - filters.indexOf(b));
      const yours = evalOrder(m, filters, g);
      const rule = evalOrder(m, ranked, g);
      const same = filters.every((f, i) => f === ranked[i]);
      const perms = filters.length > 1 && filters.length <= 6 ? permutations(filters) : null;
      const best = perms ? perms.map((o) => ({ o, e: evalOrder(m, o, g) })).sort((a, b) => a.e.total - b.e.total)[0] : null;
      yours.stages.forEach((s) => {
        const li = rowsEl.querySelector('[data-id="' + s.f.id + '"]');
        if (!li) return;
        li.querySelector('[data-io]').textContent = int(s.docsIn) + ' → ' + int(s.passed);
        li.querySelector('.pg-io-in').style.width = (s.docsIn / g.N) * 100 + '%';
        li.querySelector('.pg-io-out').style.width = (s.passed / g.N) * 100 + '%';
      });
      const gain = yours.total - rule.total;
      const verdict = filters.length === 1 ? 'Single filter' : same ? 'Your order matches the ordering rule' : Math.abs(gain) < 1e-9 ? 'Both orders have the same SoL estimate' : gain > 0 ? 'The rule lowers the SoL estimate by ' + fmtTime(gain) + ' (' + dec((gain / yours.total) * 100, 2) + '%)' : 'Your order is lower by ' + fmtTime(-gain) + ' (' + dec((-gain / rule.total) * 100, 2) + '%)';
      const chips = ranked.map((f) => '<li class="pg-chip" style="--c:' + f.color + '"><span class="pg-chip-name">' + esc(f.name) + '</span></li>').join('');
      const bestNote = best ? (best.e.total < rule.total - 1e-9 ? 'Fastest of all ' + int(perms.length) + ' orders: ' + best.o.map((f) => esc(f.name)).join(' → ') + ' at ' + fmtTime(best.e.total) + '.' : 'The rule\'s order is the fastest of all ' + int(perms.length) + ' orders.') : '';
      const rows = yours.terms.map((x) => {
        const t = TERMS[x.name] || [esc(x.name), ''];
        return '<tr><td class="pg-left">' + t[0] + '</td><td class="pg-left">' + t[1] + '</td><td>' + (x.F ? fmtSci(x.F) : '—') + '</td><td>' + (x.pi ? dec(x.pi / 1e12, 1) + ' TFLOP/s' : '—') + '</td><td>' + (x.F ? fmtTime(x.tc) : '0') + '</td><td>' + (x.B ? dec(x.B / 1e9, 1) + ' GB' : '0') + '</td><td>' + (x.B ? fmtTime(x.tm) : '0') + '</td><td>' + fmtTime(x.T) + '</td><td class="pg-left">' + (x.tc >= x.tm ? 'compute' : 'memory') + '</td></tr>';
      }).join('');
      const stageRows = yours.stages.map((s, j) => '<tr><td>' + (j + 1) + (j === 0 ? ' (scan)' : ' (ask)') + '</td><td class="pg-left"><span class="pg-dot-c" style="--c:' + s.f.color + '"></span>' + esc(s.f.name) + '</td><td>' + int(s.docsIn) + '</td><td>' + int(s.passed) + '</td><td>' + int(s.tokens) + '</td></tr>').join('');
      const rankRows = ranked.map((f, i) => '<tr><td>' + (i + 1) + '</td><td class="pg-left"><span class="pg-dot-c" style="--c:' + f.color + '"></span>' + esc(f.name) + '</td><td>' + dec(f.ask * 1e6, 1) + '</td><td>' + dec(f.s, 3) + '</td><td>' + (isFinite(f.rank) ? dec(f.rank * 1e6, 1) : '∞') + '</td></tr>').join('');
      const w = yours.work;
      resEl.innerHTML =
        '<div class="pg-label-row"><span class="pg-h">Ordering rule</span></div><ol class="pg-chain">' + chips + '</ol>' +
        '<div class="pg-tiles"><div class="pg-tile pg-tile-main"><span class="pg-tile-label">SoL estimate, ordering rule</span><span class="pg-tile-value">' + fmtTime(rule.total) + '</span><span class="pg-verdict">' + int(rule.work.n) + ' tokens computed</span></div>' +
        '<div class="pg-tile"><span class="pg-tile-label">SoL estimate, your order</span><span class="pg-tile-value">' + fmtTime(yours.total) + '</span><span class="pg-verdict' + (!same && gain > 1e-9 ? ' is-gain' : '') + '">' + verdict + '</span></div></div>' +
        (bestNote ? '<div class="pg-note">' + bestNote + '</div>' : '') +
        '<details class="pg-details"><summary>Show the math (your order)</summary>' +
        '<div class="pg-h-small pg-mt">Stages</div><div class="pg-scroll"><table class="pg-table"><thead><tr><th class="pg-left">Stage</th><th class="pg-left">Filter</th><th>Documents asked</th><th>Passed</th><th>Tokens computed</th></tr></thead><tbody>' + stageRows + '</tbody></table></div>' +
        '<div class="pg-note">n = ' + int(w.n) + ', A = ' + int(w.A) + ', W = ' + int(w.W) + ', R = ' + int(w.R) + ', U<sub>W</sub> = ' + int(w.UW) + ', U<sub>R</sub> = ' + int(w.UR) + ', K = ' + yours.K + ' passes of up to ' + int(m.chunk) + ' tokens.</div>' +
        '<div class="pg-h-small pg-mt">Roofline terms</div><div class="pg-scroll"><table class="pg-table"><thead><tr><th class="pg-left">Term</th><th class="pg-left">Prices</th><th><i>F</i></th><th><i>Π</i></th><th><i>F</i>/<i>Π</i></th><th><i>B</i></th><th><i>B</i>/<i>β</i></th><th><i>T</i><sub>j</sub></th><th class="pg-left">Bound</th></tr></thead><tbody>' + rows +
        '<tr class="pg-total"><td class="pg-left" colspan="7"><b>Total</b></td><td><b>' + fmtTime(yours.total) + '</b></td><td></td></tr></tbody></table></div>' +
        '<div class="pg-h-small pg-mt">Ranks (microseconds per document)</div><div class="pg-scroll"><table class="pg-table"><thead><tr><th>#</th><th class="pg-left">Filter</th><th>ask</th><th><i>s</i></th><th>ask / (1 − <i>s</i>)</th></tr></thead><tbody>' + rankRows + '</tbody></table></div></details>';
    }
    modelEl.addEventListener('change', () => { st.model = modelEl.value; update(); });
    el.addEventListener('input', (e) => {
      const t = e.target;
      if (t.dataset.g) { st[t.dataset.g] = t.value; update(); }
      else if (t.dataset.field) {
        const li = t.closest('[data-id]');
        const f = st.filters.find((x) => x.id === Number(li.dataset.id));
        f[t.dataset.field] = t.value;
        if (t.dataset.field === 's') {
          li.querySelectorAll('[data-field="s"]').forEach((i) => { if (i !== t) i.value = t.value; });
          paint(li.querySelector('.pg-range'));
        }
        update();
      }
    });
    el.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      const li = b.closest('[data-id]');
      if (b.dataset.move && li) {
        const i = st.filters.findIndex((x) => x.id === Number(li.dataset.id));
        const j = i + Number(b.dataset.move);
        if (j >= 0 && j < st.filters.length) { const [x] = st.filters.splice(i, 1); st.filters.splice(j, 0, x); drawRows(); update(); }
      } else if (b.dataset.remove && li && st.filters.length > 1) {
        st.filters = st.filters.filter((x) => x.id !== Number(li.dataset.id)); drawRows(); update();
      } else if (b.dataset.action === 'add' && st.filters.length < 8) {
        const id = st.nextId++; st.filters.push({ id, name: 'Filter ' + id, s: 0.5, q: 80 }); drawRows(); update();
      } else if (b.dataset.action === 'apply') {
        const r = readAll();
        if (r.error) return;
        const m = MODELS[st.model], p = r.g.pre + r.g.len;
        const rank = (x) => { const s = Number(x.s); return s >= 1 ? Infinity : docCost(m, 'ask', p, Number(x.q)) / (1 - s); };
        st.filters.sort((a, b2) => rank(a) - rank(b2)); drawRows(); update();
      } else if (b.dataset.action === 'reset') {
        st = freshState(); modelEl.value = st.model;
        el.querySelectorAll('[data-g]').forEach((i) => (i.value = st[i.dataset.g]));
        drawRows(); update();
      }
    });
    drawRows();
    update();
  }

  function init() {
    document.querySelectorAll('[data-hybrid-capacity]').forEach(initCapacity);
    document.querySelectorAll('[data-hybrid-sales4]').forEach(initSales);
  }
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
  }
  root.HybridPlayground = { slots, roofline, evalOrder, MODELS, DATA };
})(typeof window !== 'undefined' ? window : globalThis);
