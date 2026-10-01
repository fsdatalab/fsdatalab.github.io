/*
 * Interactive Speed-of-Light calculator for chains of AI-powered filters.
 *
 * Implements the cost model of "How to Cost Your AI-Powered Filters"
 * (Sections 3-5) for one fixed configuration: Qwen3-4B (FP8 weights, BF16
 * attention and KV) on an NVIDIA H100 SXM.
 */
(function (root) {
  'use strict';

  // ── Hardware (Table 1) and model (Table 2) ─────────────────────────────
  const HW = {
    pi8: 1.979e15, // FLOP/s, FP8 GEMMs
    pi16: 989.5e12, // FLOP/s, BF16 attention
    beta: 3.35e12, // bytes/s, HBM bandwidth
    capacity: 80e9, // bytes, HBM capacity
  };
  const LLM = {
    layers: 36,
    dModel: 2560,
    nHeads: 32,
    nKv: 8,
    dHead: 128,
    dMlp: 9728,
    vocab: 151936,
    bw: 1, // bytes per weight (FP8)
    bkv: 2, // bytes per KV element (BF16)
  };

  const P_PROJ = LLM.layers * 2 * LLM.dModel * LLM.dHead * (LLM.nHeads + LLM.nKv);
  const P_MLP = 3 * LLM.layers * LLM.dModel * LLM.dMlp;
  const ATTN_FLOPS_PER_CMP = 4 * LLM.nHeads * LLM.dHead * LLM.layers;
  const B_KV = 2 * LLM.bkv * LLM.layers * LLM.nKv * LLM.dHead; // bytes/token
  const CHUNK = Math.floor((Math.pow(2, 31) - 1) / (2 * LLM.dMlp));
  // Section 4.1 / Table 4: resident weights include embeddings and FP8 scales.
  const RESIDENT_WEIGHT_BYTES = 4.5e9;
  const USABLE_HBM_BYTES = 0.95 * HW.capacity;
  const ACTIVATION_BYTES = 2 * CHUNK * 32 * LLM.dModel;

  // ── Roofline pieces ────────────────────────────────────────────────────
  function roof(compute, memory) {
    return { compute, memory, t: Math.max(compute, memory), bound: memory > compute ? 'memory' : 'compute' };
  }

  // Latency of one forward-pass stage that processes n tokens, makes A
  // attention comparisons, and writes W / reads R KV tokens (Sections 3.2-3.5).
  function stageTimes(n, A, W, R) {
    if (n <= 0) {
      const z = roof(0, 0);
      return { proj: z, attn: z, mlp: z, total: 0, passes: 0 };
    }
    const K = Math.ceil(n / CHUNK);
    const proj = roof((2 * P_PROJ * n) / HW.pi8, (LLM.bw * P_PROJ * K) / HW.beta);
    const mlp = roof((2 * P_MLP * n) / HW.pi8, (LLM.bw * P_MLP * K) / HW.beta);
    const attn = roof((ATTN_FLOPS_PER_CMP * A) / HW.pi16, (B_KV * (W + R)) / HW.beta);
    return { proj, attn, mlp, total: proj.t + attn.t + mlp.t, passes: K };
  }

  // Per-document ask / scan cost used for ranking (Section 5.1): projections and
  // MLP at their compute time, attention at max(compute, KV traffic).
  function docCosts(p, q) {
    const cost = (n, A, kvTokens) =>
      (2 * P_PROJ * n) / HW.pi8 +
      (2 * P_MLP * n) / HW.pi8 +
      Math.max((ATTN_FLOPS_PER_CMP * A) / HW.pi16, (B_KV * kvTokens) / HW.beta);
    const r = p + q;
    return {
      ask: cost(q, q * p + (q * (q + 1)) / 2, p + q),
      scan: cost(r, (r * (r + 1)) / 2, r),
    };
  }

  // S(pi) = N scan_{pi1} + sum_{j>=2} ask_{pi_j} N prod_{k<j} s_{pi_k}  (Section 4.2)
  function orderCost(order, N) {
    let total = N * order[0].scan;
    let docs = N * order[0].s;
    for (let j = 1; j < order.length; j++) {
      total += order[j].ask * docs;
      docs *= order[j].s;
    }
    return total;
  }

  // Sort by rank, then try each filter as the first one (Section 4.3).
  function chooseOrder(filters, N) {
    const ranked = filters.slice().sort((a, b) => a.rank - b.rank || a.index - b.index);
    const candidates = filters.map((f) => {
      const order = [f].concat(ranked.filter((g) => g !== f));
      return { first: f, order, cost: orderCost(order, N) };
    });
    let best = candidates[0];
    for (const c of candidates) {
      const rankPos = (x) => ranked.indexOf(x.first);
      if (c.cost < best.cost - 1e-15 || (Math.abs(c.cost - best.cost) <= 1e-15 && rankPos(c) < rankPos(best))) best = c;
    }
    return { ranked, candidates, best };
  }

  // Stage-by-stage work of a fixed order (Section 4.2, Table 7), and its
  // Speed-of-Light: Quail sums each stage's tokens, attention comparisons,
  // and KV traffic first, then applies the roofline once per component to
  // the totals (Section 5.2, Table 8) rather than adding each stage's own
  // roofline latency. Per-stage times are still kept for the breakdown
  // table to show each filter's arithmetic and memory bottlenecks.
  function evaluateOrder(order, cfg) {
    const { N, len, qpre } = cfg;
    const p = qpre + len;
    const stages = [];
    let expected = N;
    let nTot = 0, ATot = 0, WTot = 0, RTot = 0;
    order.forEach((f, j) => {
      const q = f.q;
      let docsIn, n, A, W, R;
      if (j === 0) {
        const r = p + q;
        docsIn = N;
        n = N * r;
        A = (N * r * (r + 1)) / 2;
        W = n;
        R = 0;
      } else {
        docsIn = expected;
        n = docsIn * q;
        A = docsIn * (q * p + (q * (q + 1)) / 2);
        W = n;
        R = docsIn * p;
      }
      stages.push(Object.assign({ filter: f, docsIn, n, A, W, R }, stageTimes(n, A, W, R)));
      nTot += n;
      ATot += A;
      WTot += W;
      RTot += R;
      expected *= f.s;
    });
    const chain = stageTimes(nTot, ATot, WTot, RTot);
    return {
      stages,
      proj: chain.proj.t,
      attn: chain.attn.t,
      mlp: chain.mlp.t,
      total: chain.total,
    };
  }

  function kvBatching(cfg) {
    const p = cfg.qpre + cfg.len;
    const perDoc = B_KV * p;
    const free = USABLE_HBM_BYTES - RESIDENT_WEIGHT_BYTES - ACTIVATION_BYTES;
    const perBatch = Math.floor(free / perDoc);
    return {
      perDocBytes: perDoc,
      freeBytes: free,
      perBatch,
      batches: perBatch > 0 ? Math.ceil(cfg.N / perBatch) : Infinity,
    };
  }

  // filters: [{name, s, q}] in the order the user entered them.
  function solve(cfg, filterInputs) {
    const p = cfg.qpre + cfg.len;
    const filters = filterInputs.map((f, index) => {
      const { ask, scan } = docCosts(p, f.q);
      return Object.assign({}, f, {
        index,
        ask,
        scan,
        rank: f.s >= 1 ? Infinity : ask / (1 - f.s),
      });
    });
    const { ranked, candidates, best } = chooseOrder(filters, cfg.N);
    return {
      filters,
      ranked,
      candidates,
      best,
      optimal: evaluateOrder(best.order, cfg),
      entered: evaluateOrder(filters, cfg),
      kv: kvBatching(cfg),
    };
  }

  function permutations(items) {
    if (items.length <= 1) return [items.slice()];
    const out = [];
    items.forEach((x, i) => {
      permutations(items.slice(0, i).concat(items.slice(i + 1))).forEach((rest) => out.push([x].concat(rest)));
    });
    return out;
  }

  // Exact SoL of every ordering (only sensible for a handful of filters).
  function allOrders(filters, cfg) {
    return permutations(filters).map((order) => ({ order, total: evaluateOrder(order, cfg).total }));
  }

  const model = { HW, LLM, P_PROJ, P_MLP, B_KV, CHUNK, stageTimes, permutations, allOrders, docCosts, orderCost, chooseOrder, evaluateOrder, kvBatching, solve };

  if (typeof module === 'object' && module.exports) module.exports = model;
  root.FilterChainModel = model;

  // ── Playground UI ──────────────────────────────────────────────────────
  if (typeof document === 'undefined') return;

  const MAX_FILTERS = 8;
  const MAX_BRUTE_FORCE = 6; // 720 orderings
  // One accent per filter so a row can be followed into the optimal order.
  const COLORS = ['#C41230', '#555555', '#8B2332', '#777777', '#A6192E', '#444444', '#9B555F', '#666666'];
  // The three IMDB predicates from the post's worked example (Section 5),
  // plus a blank slot for a custom filter.
  const PRESETS = [
    { name: 'mentions a positive aspect', s: '0.4856', q: '51' },
    { name: 'discusses the ending', s: '0.2273', q: '45' },
    { name: 'mentions a named actor', s: '0.6123', q: '49' },
    { name: 'custom', s: '0.5', q: '45', custom: true },
  ];
  // The playground's only scenario: the IMDB conjunction of Section 5,
  // entered in the reverse of the order it was written in (F3, F2, F1),
  // so the ranked-order fix in the "Optimal order" lane is visible at load.
  const DEFAULT_WORK = { N: '5000', len: '298.8466', qpre: '2' };
  const DEFAULT_FILTERS = [
    ['mentions a named actor', '0.6123', '49'],
    ['discusses the ending', '0.2273', '45'],
    ['mentions a positive aspect', '0.4856', '51'],
  ];
  let instances = 0;

  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

  function fmtTime(s) {
    if (s < 0) return '\u2212' + fmtTime(-s);
    if (s >= 1) return s.toFixed(2) + ' s';
    if (s >= 1e-3) return (s * 1e3).toFixed(2) + ' ms';
    return (s * 1e6).toFixed(1) + ' µs';
  }
  const fmtMs = (s) => (s * 1e3).toFixed(3);
  const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
  const ordinal = (n) => n + (['th', 'st', 'nd', 'rd'][(n % 100 > 10 && n % 100 < 14) || n % 10 > 3 ? 0 : n % 10] || 'th');
  const reduceMotion = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function newFilter(state, name, s, q, custom) {
    const id = state.nextId++;
    return { id, name: custom ? 'Filter ' + id : name, s, q, color: COLORS[(id - 1) % COLORS.length] };
  }

  function defaultState() {
    const state = Object.assign({ nextId: 1, filters: [] }, DEFAULT_WORK);
    DEFAULT_FILTERS.forEach((f) => state.filters.push(newFilter(state, f[0], f[1], f[2])));
    return state;
  }

  function paintRange(range) {
    const lo = Number(range.min);
    const hi = Number(range.max);
    const pct = hi > lo ? Math.min(100, Math.max(0, ((Number(range.value) - lo) / (hi - lo)) * 100)) : 0;
    range.style.setProperty('--p', pct + '%');
  }

  function rowHTML(f, idx, total) {
    return (
      '<li class="pg-row" data-id="' + f.id + '" style="--c:' + f.color + '">' +
      '<span class="pg-handle" title="Drag to reorder"><span class="pg-grip" aria-hidden="true"></span><span class="pg-badge">' + (idx + 1) + '</span></span>' +
      '<input class="pg-name" type="text" data-field="name" value="' + esc(f.name) + '" maxlength="60" autocomplete="off" aria-label="Predicate">' +
      '<div class="pg-s" data-pair><input class="pg-range" type="range" min="0" max="1" step="0.005" value="' + esc(f.s) + '" aria-label="Selectivity">' +
      '<input class="pg-num" type="number" inputmode="decimal" min="0" max="1" step="any" data-field="s" value="' + esc(f.s) + '" aria-label="Selectivity value"></div>' +
      '<input class="pg-num pg-q" type="number" inputmode="numeric" min="1" step="1" data-field="q" value="' + esc(f.q) + '" aria-label="Instruction length in tokens">' +
      '<div class="pg-io"><span data-io></span><span class="pg-io-bar"><i class="pg-io-in"></i><i class="pg-io-out"></i></span></div>' +
      '<span class="pg-tools">' +
      '<button type="button" class="pg-icon" data-move="-1" aria-label="Move earlier"' + (idx === 0 ? ' disabled' : '') + '>&lsaquo;</button>' +
      '<button type="button" class="pg-icon" data-move="1" aria-label="Move later"' + (idx === total - 1 ? ' disabled' : '') + '>&rsaquo;</button>' +
      '<button type="button" class="pg-icon pg-remove" data-remove="1" aria-label="Remove filter"' + (total === 1 ? ' disabled' : '') + '>&times;</button>' +
      '</span></li>'
    );
  }

  function presetHTML(p, i) {
    return (
      '<li class="pg-preset' + (p.custom ? ' is-custom' : '') + '" data-preset="' + i + '" tabindex="0" role="button" aria-label="Add ' + esc(p.name) + '">' +
      (p.custom ? '+ ' : '<span class="pg-grip" aria-hidden="true"></span>') + esc(p.name) + '</li>'
    );
  }

  function field(label, attrs) {
    return '<label class="pg-field"><span>' + label + '</span><input class="pg-num" type="number" inputmode="decimal" step="any" ' + attrs + '></label>';
  }

  function skeleton(state) {
    return (
      '<div class="pg-bar"><span class="pg-title">Filter Playground <span class="pg-sub">Qwen3-4B-fp8 on H100, IMDB conjunction (Section 5)</span></span>' +
      '<div class="pg-bar-actions">' +
      '<button type="button" class="pg-btn" data-action="reset">Reset</button>' +
      '<button type="button" class="pg-btn" data-action="shuffle">Shuffle</button>' +
      '<button type="button" class="pg-btn pg-btn-primary" data-action="apply">Apply ordering rule</button></div></div>' +
      '<div class="pg-body">' +
      '<div class="pg-workload">' +
      field('Documents <i>N</i>', 'min="1" data-global="N" value="' + esc(state.N) + '"') +
      field('Average doc length', 'min="1" data-global="len" value="' + esc(state.len) + '"') +
      field('Preamble <i>q</i><sub>pre</sub>', 'min="0" data-global="qpre" value="' + esc(state.qpre) + '"') +
      '</div>' +
      '<div class="pg-label-row"><span class="pg-h">Your filter order</span></div>' +
      '<div class="pg-cols" aria-hidden="true"><span></span><span>Predicate</span><span>Selectivity <i>s</i></span><span><i>q</i><sub>i</sub></span><span>Docs</span><span></span></div>' +
      '<ol class="pg-rows" aria-label="Filters in the order you entered them"></ol>' +
      '<div class="pg-library"><span class="pg-h-small">Drag in</span><ul class="pg-presets">' + PRESETS.map(presetHTML).join('') + '</ul></div>' +
      '<div class="pg-results" aria-live="polite"></div></div>'
    );
  }

  function parseConfig(state) {
    const errors = {};
    const num = (str) => (String(str).trim() === '' ? NaN : Number(str));
    const cfg = { N: num(state.N), len: num(state.len), qpre: num(state.qpre) };
    if (!(Number.isInteger(cfg.N) && cfg.N >= 1 && cfg.N <= 1e8)) errors['g:N'] = 'Documents must be a whole number of at least 1.';
    if (!(cfg.len >= 1 && cfg.len <= 1e6)) errors['g:len'] = 'Average document length must be at least 1 token.';
    if (!(Number.isInteger(cfg.qpre) && cfg.qpre >= 0 && cfg.qpre <= 1e5)) errors['g:qpre'] = 'q_pre must be a whole number of tokens, 0 or more.';
    const filters = state.filters.map((f, i) => {
      const s = num(f.s);
      const q = num(f.q);
      const name = f.name.trim() || 'Filter ' + (i + 1);
      if (!(s >= 0 && s <= 1)) errors[f.id + ':s'] = 'Selectivity of "' + name + '" must be between 0 and 1.';
      if (!(Number.isInteger(q) && q >= 1 && q <= 1e5)) errors[f.id + ':q'] = 'Instruction length for "' + name + '" must be a whole number of tokens, 1 or more.';
      return { id: f.id, name, s, q, color: f.color };
    });
    return { cfg, filters, errors };
  }

  // Animate elements from their previous position to their new one (FLIP).
  function flip(root, selector, key, mutate, duration) {
    const before = {};
    root.querySelectorAll(selector).forEach((n) => (before[n.dataset[key]] = n.getBoundingClientRect()));
    mutate();
    if (reduceMotion()) return;
    root.querySelectorAll(selector).forEach((n) => {
      const b = before[n.dataset[key]];
      if (!b || !n.animate) return;
      const a = n.getBoundingClientRect();
      const dx = b.left - a.left;
      const dy = b.top - a.top;
      if (Math.abs(dx) < 1 && Math.abs(dy) < 1) return;
      n.animate([{ transform: 'translate(' + dx + 'px,' + dy + 'px)' }, { transform: 'none' }], { duration: duration || 520, easing: 'cubic-bezier(.2,.8,.2,1)' });
    });
  }

  // Count a number up or down to its new value.
  function tween(node, from, to) {
    if (reduceMotion() || from === undefined || Math.abs(from - to) < 1e-12) {
      node.textContent = fmtTime(to);
      return;
    }
    node.textContent = fmtTime(from);
    // Frames are paused in background tabs, so always land on the final value.
    setTimeout(() => {
      if (node.isConnected) node.textContent = fmtTime(to);
    }, 480);
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / 420);
      const e = 1 - Math.pow(1 - t, 3);
      if (t < 1 && node.isConnected) {
        node.textContent = fmtTime(from + (to - from) * e);
        requestAnimationFrame(step);
      } else if (node.isConnected) node.textContent = fmtTime(to);
    };
    requestAnimationFrame(step);
  }

  // Dot plot of the SoL of every ordering of the chain.
  function orderingPlot(all, yoursKey, ruleKey) {
    const W = 640, padL = 14, padR = 14, step = 9;
    const totals = all.map((a) => a.total);
    const lo = Math.min.apply(null, totals);
    const hi = Math.max.apply(null, totals);
    const span = hi - lo || 1;
    const x = (v) => padL + ((v - lo) / span) * (W - padL - padR);
    const lanes = [];
    const placed = all
      .map((a) => ({ a, cx: x(a.total) }))
      .sort((p, q) => p.cx - q.cx)
      .map((d) => {
        let lane = lanes.findIndex((last) => d.cx - last >= 9);
        if (lane === -1) lane = lanes.length;
        lanes[lane] = d.cx;
        return Object.assign(d, { lane });
      });
    const H = 16 + lanes.length * step + 30;
    const axisY = H - 28;
    const dots = placed.map((d) => Object.assign(d, { cy: axisY - 12 - d.lane * step }));
    const key = (order) => order.map((f) => f.id).join('-');
    const yours = dots.find((d) => key(d.a.order) === yoursKey);
    const rule = dots.find((d) => key(d.a.order) === ruleKey);
    const best = dots.slice().sort((p, q) => p.a.total - q.a.total)[0];
    let svg = '<svg class="pg-plot-svg" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Latency of every ordering of the chain">';
    svg += '<line x1="' + padL + '" y1="' + axisY + '" x2="' + (W - padR) + '" y2="' + axisY + '" class="pg-axis"/>';
    const orderAttr = (order) => order.map((f) => f.id).join(',');
    const dotTitle = (d) => esc(d.a.order.map((f) => f.name).join(' → ')) + ': ' + fmtTime(d.a.total) + ' (click to apply)';
    dots.forEach((d) => {
      svg += '<circle cx="' + d.cx.toFixed(1) + '" cy="' + d.cy + '" r="3.4" class="pg-dot" tabindex="0" role="button" data-order="' + orderAttr(d.a.order) + '"><title>' + dotTitle(d) + '</title></circle>';
    });
    if (yours) svg += '<circle cx="' + yours.cx.toFixed(1) + '" cy="' + yours.cy + '" r="6.5" class="pg-dot-yours" tabindex="0" role="button" data-order="' + orderAttr(yours.a.order) + '"><title>' + dotTitle(yours) + '</title></circle>';
    if (rule) svg += '<circle cx="' + rule.cx.toFixed(1) + '" cy="' + rule.cy + '" r="4.6" class="pg-dot-rule" tabindex="0" role="button" data-order="' + orderAttr(rule.a.order) + '"><title>' + dotTitle(rule) + '</title></circle>';
    if (best && best !== rule) svg += '<circle cx="' + best.cx.toFixed(1) + '" cy="' + best.cy + '" r="4.6" class="pg-dot-best" tabindex="0" role="button" data-order="' + orderAttr(best.a.order) + '"><title>' + dotTitle(best) + '</title></circle>';
    svg += '<text x="' + padL + '" y="' + (axisY + 16) + '" class="pg-axis-label">' + fmtTime(lo) + '</text>';
    svg += '<text x="' + (W - padR) + '" y="' + (axisY + 16) + '" text-anchor="end" class="pg-axis-label">' + fmtTime(hi) + '</text>';
    return svg + '</svg>';
  }

  function mount(el) {
    const uid = 'pg-filters-' + ++instances;
    const state = defaultState();
    el.innerHTML = skeleton(state);
    const rowsEl = el.querySelector('.pg-rows');
    const presetsEl = el.querySelector('.pg-presets');
    const resultsEl = el.querySelector('.pg-results');
    const lastValue = {};
    let lastSol = null;

    function renderRows() {
      const total = state.filters.length;
      rowsEl.innerHTML = state.filters.map((f, i) => rowHTML(f, i, total)).join('');
      rowsEl.querySelectorAll('.pg-range').forEach(paintRange);
      presetsEl.classList.toggle('is-full', total >= MAX_FILTERS);
    }

    function refreshRows() {
      const total = state.filters.length;
      [].forEach.call(rowsEl.children, (li, i) => {
        li.querySelector('.pg-badge').textContent = i + 1;
        li.querySelector('[data-move="-1"]').disabled = i === 0;
        li.querySelector('[data-move="1"]').disabled = i === total - 1;
        li.querySelector('.pg-remove').disabled = total === 1;
      });
    }

    function markErrors(errors) {
      el.querySelectorAll('input.pg-num').forEach((input) => {
        const row = input.closest('.pg-row');
        const key = input.dataset.global ? 'g:' + input.dataset.global : (row ? row.dataset.id : '') + ':' + input.dataset.field;
        const bad = key in errors;
        input.classList.toggle('is-invalid', bad);
        input.setAttribute('aria-invalid', bad ? 'true' : 'false');
      });
    }

    // expected survivors after each stage
    function survivors(cfg, order) {
      let alive = cfg.N;
      return order.map((f) => (alive *= f.s));
    }

    function stageRow(s, j) {
      const cell = (r) => fmtTime(r.t) + (r.bound === 'memory' && r.t > 0 ? ' <span class="pg-tag">memory-bound</span>' : '');
      return (
        '<tr><td>' + (j + 1) + '</td><td class="pg-left"><span class="pg-dot-c" style="--c:' + s.filter.color + '"></span>' + esc(s.filter.name) + '</td>' +
        '<td>' + fmtInt(s.docsIn) + '</td><td>' + fmtInt(s.n) + '</td>' +
        '<td>' + cell(s.proj) + '</td><td>' + cell(s.attn) + '</td><td>' + cell(s.mlp) + '</td>' +
        '<td><strong>' + fmtTime(s.total) + '</strong></td></tr>'
      );
    }

    function resultsHTML(cfg, sol) {
      const m = sol.filters.length;
      const opt = sol.optimal;
      const ent = sol.entered;
      const same = sol.best.order.every((f, i) => f.index === i);
      const gain = ent.total - opt.total;
      const gainPct = ent.total > 0 ? (gain / ent.total) * 100 : 0;

      const chips = sol.best.order
        .map((f, j) => {
          const moved = f.index !== j;
          return (
            '<li class="pg-chip' + (moved ? ' is-moved' : '') + '" data-chip="' + f.id + '" style="--c:' + f.color + '">' +
            '<span class="pg-badge">' + (j + 1) + '</span><span class="pg-chip-name">' + esc(f.name) + '</span>' +
            (moved ? '<span class="pg-chip-was">was ' + ordinal(f.index + 1) + '</span>' : '') + '</li>'
          );
        })
        .join('');

      const verdict = m === 1 ? 'Single filter'
        : same ? 'Your order matches the ordering rule'
        : Math.abs(gain) < 1e-12 ? 'Both orders have the same SoL estimate'
        : gain > 0 ? 'The rule lowers the SoL estimate by ' + fmtTime(gain) + ' (' + gainPct.toFixed(2) + '%)'
        : 'Your order has a lower SoL estimate by ' + fmtTime(-gain) + ' (' + (-gainPct).toFixed(2) + '%)';

      let plot = '';
      if (m > 1 && m <= MAX_BRUTE_FORCE) {
        const all = allOrders(sol.filters, cfg);
        const rank = 1 + all.filter((a) => a.total < opt.total - 1e-12).length;
        const bestTotal = Math.min.apply(null, all.map((a) => a.total));
        const key = (order) => order.map((f) => f.id).join('-');
        const msg =
          rank === 1
            ? 'The rule’s order is the fastest of all ' + fmtInt(all.length) + '.'
            : 'The rule’s order ranks ' + ordinal(rank) + ' of ' + fmtInt(all.length) + '; the best is ' + (((opt.total - bestTotal) / opt.total) * 100).toFixed(2) + '% quicker.';
        plot =
          '<div class="pg-plot"><div class="pg-plot-head"><span class="pg-h-small">All ' + fmtInt(all.length) + ' orderings</span>' +
          '<span class="pg-keys"><span><i class="pg-key pg-key-yours"></i>yours</span><span><i class="pg-key pg-key-rule"></i>rule</span>' +
          (rank === 1 ? '' : '<span><i class="pg-key pg-key-best"></i>best</span>') + '</span></div>' +
          orderingPlot(all, key(sol.filters), key(sol.best.order)) +
          '<div class="pg-plot-msg">' + msg + '</div></div>';
      }

      let math = '';
      if (m > 1) {
        const rankRows = sol.ranked
          .map(
            (f, i) =>
              '<tr><td>' + (i + 1) + '</td><td class="pg-left"><span class="pg-dot-c" style="--c:' + f.color + '"></span>' + esc(f.name) + '</td><td>' +
              fmtMs(f.ask) + '</td><td>' + fmtMs(f.scan) + '</td><td>' + (isFinite(f.rank) ? fmtMs(f.rank) : '&infin;') + '</td></tr>'
          )
          .join('');
        const candRows = sol.candidates
          .slice()
          .sort((a, b) => a.cost - b.cost)
          .map(
            (c) =>
              '<tr' + (c === sol.best ? ' class="is-best"' : '') + '><td class="pg-left">' + c.order.map((f) => esc(f.name)).join(' &rarr; ') + '</td><td>' + fmtTime(c.cost) + '</td></tr>'
          )
          .join('');
        math =
          '<div class="pg-h-small pg-mt">Rank filters, ms per document</div>' +
          '<div class="pg-scroll"><table class="pg-table"><thead><tr><th>#</th><th class="pg-left">Filter</th><th>ask</th><th>scan</th><th>ask / (1 &minus; <i>s</i>)</th></tr></thead><tbody>' + rankRows + '</tbody></table></div>' +
          '<div class="pg-h-small pg-mt">Candidate orders, each filter tried first</div>' +
          '<div class="pg-scroll"><table class="pg-table"><thead><tr><th class="pg-left">Order</th><th>S(&pi;)</th></tr></thead><tbody>' + candRows + '</tbody></table></div>';
      }
      const kv = sol.kv;
      const kvNote =
        kv.perBatch < 1
          ? 'One document’s prefix KV (' + (kv.perDocBytes / 1e9).toFixed(2) + ' GB) does not fit in the free HBM (' + (kv.freeBytes / 1e9).toFixed(2) + ' GB).'
          : 'KV per document ' + (kv.perDocBytes / 1e9).toFixed(3) + ' GB; ' + (kv.freeBytes / 1e9).toFixed(2) + ' GB free, so ' + fmtInt(kv.perBatch) + ' documents per batch (' + fmtInt(kv.batches) + (kv.batches === 1 ? ' batch' : ' batches') + ').';
      math +=
        '<div class="pg-h-small pg-mt">Stage costs, ordering rule</div>' +
        '<div class="pg-scroll"><table class="pg-table"><thead><tr><th>#</th><th class="pg-left">Filter</th><th>Docs in</th><th>Tokens</th><th><i>T</i><sub>proj</sub></th><th><i>T</i><sub>attn</sub></th><th><i>T</i><sub>mlp</sub></th><th>Stage</th></tr></thead><tbody>' +
        opt.stages.map(stageRow).join('') +
        '<tr class="pg-total"><td></td><td class="pg-left">Conjunction SoL</td><td></td><td>' + fmtInt(opt.stages.reduce((a, s) => a + s.n, 0)) + '</td><td>' + fmtTime(opt.proj) + '</td><td>' + fmtTime(opt.attn) + '</td><td>' + fmtTime(opt.mlp) + '</td><td><strong>' + fmtTime(opt.total) + '</strong></td></tr>' +
        '</tbody></table></div><div class="pg-note">The total applies the roofline to combined work, so it can be lower than the sum of stage times.</div><div class="pg-note">' + kvNote + '</div>';

      return (
        '<div class="pg-label-row"><span class="pg-h">Ordering rule</span></div>' +
        '<ol class="pg-chain">' + chips + '</ol>' +
        '<div class="pg-tiles">' +
        '<div class="pg-tile pg-tile-main"><span class="pg-tile-label">SoL estimate, ordering rule</span><span class="pg-tile-value" data-count="opt"></span></div>' +
        '<div class="pg-tile"><span class="pg-tile-label">SoL estimate, your order</span><span class="pg-tile-value" data-count="ent"></span>' +
        '<span class="pg-verdict' + (!same && gain > 1e-12 ? ' is-gain' : '') + '">' + verdict + '</span></div>' +
        '</div>' +
        plot +
        '<details class="pg-details"><summary>Show the math</summary>' + math + '</details>'
      );
    }

    function updateRowReadouts(cfg, sol) {
      const out = survivors(cfg, sol.filters);
      sol.filters.forEach((f, j) => {
        const li = rowsEl.querySelector('[data-id="' + f.id + '"]');
        if (!li) return;
        const st = sol.entered.stages[j];
        li.querySelector('[data-io]').textContent = fmtInt(st.docsIn) + ' → ' + fmtInt(out[j]);
        li.querySelector('.pg-io-in').style.width = (st.docsIn / cfg.N) * 100 + '%';
        li.querySelector('.pg-io-out').style.width = (out[j] / cfg.N) * 100 + '%';
      });
    }

    function update() {
      const { cfg, filters, errors } = parseConfig(state);
      markErrors(errors);
      const keys = Object.keys(errors);
      if (keys.length) {
        resultsEl.innerHTML = '<div class="pg-error">' + esc(errors[keys[0]]) + '</div>';
        lastSol = null;
        return;
      }
      const sol = solve(cfg, filters);
      const open = [].map.call(resultsEl.querySelectorAll('details'), (d) => d.open);
      flip(resultsEl, '[data-chip]', 'chip', () => {
        resultsEl.innerHTML = resultsHTML(cfg, sol);
        resultsEl.querySelectorAll('details').forEach((d, i) => {
          if (open[i]) d.open = true;
        });
      });
      resultsEl.querySelectorAll('[data-count]').forEach((n) => {
        const to = n.dataset.count === 'opt' ? sol.optimal.total : sol.entered.total;
        tween(n, lastValue[n.dataset.count], to);
        lastValue[n.dataset.count] = to;
      });
      updateRowReadouts(cfg, sol);
      lastSol = sol;
    }

    function reorderTo(ids) {
      flip(rowsEl, '.pg-row', 'id', () => {
        state.filters.sort((a, b) => ids.indexOf(String(a.id)) - ids.indexOf(String(b.id)));
        renderRows();
      }, 480);
      update();
    }

    function syncOrderFromDom() {
      const ids = [].map.call(rowsEl.children, (li) => li.dataset.id);
      state.filters.sort((a, b) => ids.indexOf(String(a.id)) - ids.indexOf(String(b.id)));
      refreshRows();
      update();
    }

    function addFilter(presetIndex, at) {
      if (state.filters.length >= MAX_FILTERS) return null;
      const p = PRESETS[presetIndex];
      const f = newFilter(state, p.name, p.s, p.q, p.custom);
      state.filters.splice(at === undefined ? state.filters.length : at, 0, f);
      renderRows();
      update();
      return f;
    }

    function loadDefault() {
      Object.assign(state, defaultState());
      ['N', 'len', 'qpre'].forEach((k) => (el.querySelector('[data-global="' + k + '"]').value = state[k]));
      flip(rowsEl, '.pg-row', 'id', renderRows, 380);
      update();
    }

    // ── events ───────────────────────────────────────────────────────────
    function applyInput(target) {
      if (target.dataset.global) state[target.dataset.global] = target.value;
      else if (target.dataset.field) {
        const row = target.closest('.pg-row');
        const f = row && state.filters.find((x) => String(x.id) === row.dataset.id);
        if (f) f[target.dataset.field] = target.value;
      }
    }

    el.addEventListener('input', (e) => {
      const t = e.target;
      if (t.classList.contains('pg-range')) {
        const num = t.closest('[data-pair]').querySelector('.pg-num');
        paintRange(t);
        num.value = t.value;
        applyInput(num);
        update();
      } else if (t.classList.contains('pg-num')) {
        const pair = t.closest('[data-pair]');
        const range = pair && pair.querySelector('.pg-range');
        if (range && String(t.value).trim() !== '' && isFinite(Number(t.value))) {
          range.value = t.value;
          paintRange(range);
        }
        applyInput(t);
        update();
      } else if (t.classList.contains('pg-name')) {
        applyInput(t);
        update();
      }
    });

    el.addEventListener('click', (e) => {
      const dot = e.target.closest('[data-order]');
      if (dot) {
        reorderTo(dot.dataset.order.split(','));
        return;
      }
      const btn = e.target.closest('button');
      const row = e.target.closest('.pg-row');
      const preset = e.target.closest('.pg-preset');
      if (preset && !btn) {
        const f = addFilter(Number(preset.dataset.preset));
        const input = f && rowsEl.querySelector('[data-id="' + f.id + '"] .pg-name');
        if (input) input.focus();
        return;
      }
      if (!btn) return;
      const action = btn.dataset.action;
      if (action === 'reset') {
        loadDefault();
      } else if (action === 'apply' && lastSol) {
        reorderTo(lastSol.best.order.map((f) => String(f.id)));
      } else if (action === 'shuffle' && state.filters.length > 1) {
        const before = state.filters.map((f) => f.id).join();
        let ids;
        do {
          ids = state.filters.map((f) => String(f.id));
          for (let i = ids.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [ids[i], ids[j]] = [ids[j], ids[i]];
          }
        } while (ids.join() === before);
        reorderTo(ids);
      } else if (row && btn.dataset.remove && state.filters.length > 1) {
        state.filters = state.filters.filter((f) => String(f.id) !== row.dataset.id);
        renderRows();
        update();
      } else if (row && btn.dataset.move) {
        const i = state.filters.findIndex((f) => String(f.id) === row.dataset.id);
        const j = i + Number(btn.dataset.move);
        if (j < 0 || j >= state.filters.length) return;
        flip(rowsEl, '.pg-row', 'id', () => {
          state.filters.splice(j, 0, state.filters.splice(i, 1)[0]);
          renderRows();
        }, 300);
        update();
        const again = rowsEl.querySelector('[data-id="' + row.dataset.id + '"] [data-move="' + btn.dataset.move + '"]');
        if (again && !again.disabled) again.focus();
      }
    });

    el.addEventListener('keydown', (e) => {
      const preset = e.target.closest && e.target.closest('.pg-preset');
      if (preset && (e.key === 'Enter' || e.key === ' ')) {
        e.preventDefault();
        preset.click();
      }
      const dot = e.target.closest && e.target.closest('[data-order]');
      if (dot && (e.key === 'Enter' || e.key === ' ')) {
        e.preventDefault();
        reorderTo(dot.dataset.order.split(','));
      }
    });

    renderRows();
    update();

    // ── drag and drop (SortableJS); the buttons remain as the fallback ──
    if (window.Sortable) {
      window.Sortable.create(rowsEl, {
        group: { name: uid, pull: false, put: true },
        draggable: '.pg-row',
        handle: '.pg-handle',
        animation: 200,
        easing: 'cubic-bezier(.2,.8,.2,1)',
        ghostClass: 'is-ghost',
        chosenClass: 'is-chosen',
        dragClass: 'is-drag',
        delay: 120,
        delayOnTouchOnly: true,
        onStart: () => el.classList.add('is-dragging'),
        onEnd: () => el.classList.remove('is-dragging'),
        onUpdate: syncOrderFromDom,
        onAdd: (evt) => {
          const index = Number(evt.item.dataset.preset);
          evt.item.remove();
          const f = addFilter(index, evt.newIndex);
          const input = f && PRESETS[index].custom && rowsEl.querySelector('[data-id="' + f.id + '"] .pg-name');
          if (input) {
            input.focus();
            input.select();
          }
        },
      });
      window.Sortable.create(presetsEl, {
        group: { name: uid, pull: 'clone', put: false },
        sort: false,
        animation: 150,
        ghostClass: 'is-ghost',
        dragClass: 'is-drag',
        delay: 120,
        delayOnTouchOnly: true,
        onStart: () => el.classList.add('is-dragging'),
        onEnd: () => el.classList.remove('is-dragging'),
      });
      el.classList.add('has-dnd');
    }
  }

  function init() {
    document.querySelectorAll('[data-filter-chain-calculator]').forEach(mount);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})(typeof window !== 'undefined' ? window : globalThis);
