import "./style.css";
import { createChart, LineStyle, CrosshairMode } from "lightweight-charts";

// ------------------------------------------------------------------ helpers
const $ = (s, r = document) => r.querySelector(s);
const app = $("#app");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toLocaleString("it-IT", { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (v, d = 2) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(d)}%`);
const sgn = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");
const money = (v) => (v === null || v === undefined ? "–" : `${num(v, 2)} $`);
const dt = (s) => (s ? new Date(s).toLocaleString("it-IT", { timeZone: "UTC", dateStyle: "short", timeStyle: "short" }) + " UTC" : "–");
const short = (m) => String(m || "").replace("mistralai/", "");
let timers = [];
let charts = [];
let ws = null;
let listeners = [];
function listen(ev, fn) { window.addEventListener(ev, fn); listeners.push([ev, fn]); }

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let t = await r.text();
    try { t = JSON.parse(t).detail ?? t; } catch {}
    throw new Error(typeof t === "string" ? t : JSON.stringify(t));
  }
  return r.json();
}

function chartBase(el, opts = {}) {
  const c = createChart(el, {
    autoSize: true,
    layout: { background: { color: "#151b23" }, textColor: "#8b97a6", fontSize: 11 },
    grid: { vertLines: { color: "#1e2631" }, horzLines: { color: "#1e2631" } },
    rightPriceScale: { borderColor: "#2a3340" },
    timeScale: { borderColor: "#2a3340", timeVisible: true, secondsVisible: false },
    crosshair: { mode: CrosshairMode.Normal },
    localization: { locale: "it-IT", timeFormatter: (t) => new Date(t * 1000).toISOString().slice(0, 16).replace("T", " ") },
    ...opts,
  });
  charts.push(c);
  return c;
}

function cleanup() {
  timers.forEach(clearInterval);
  timers = [];
  listeners.forEach(([e, f]) => window.removeEventListener(e, f));
  listeners = [];
  charts.forEach((c) => { try { c.remove(); } catch {} });
  charts = [];
}

function every(ms, fn) { timers.push(setInterval(fn, ms)); }

function kpi(label, value, sub = "", cls = "") {
  return `<div class="card kpi"><div class="label">${label}</div><div class="value ${cls}">${value}</div><div class="sub">${sub}</div></div>`;
}

function table(cols, rows, opts = {}) {
  if (!rows || !rows.length) return `<div class="note">${opts.empty || "Nessun dato."}</div>`;
  const head = cols.map((c) => `<th class="${c.num ? "num" : ""}">${c.h}</th>`).join("");
  const body = rows.map((r, i) => `<tr class="${opts.click ? "click" : ""}" data-i="${i}">${cols.map((c) => {
    const v = c.f ? c.f(r) : esc(r[c.k]);
    return `<td class="${c.num ? "num" : ""} ${c.cls ? c.cls(r) : ""}">${v}</td>`;
  }).join("")}</tr>`).join("");
  return `<div class="scroll" style="${opts.maxh ? `max-height:${opts.maxh}px` : ""}"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function bindRows(container, rows, fn) {
  container.querySelectorAll("tr.click").forEach((tr) => tr.addEventListener("click", () => fn(rows[+tr.dataset.i])));
}

// ------------------------------------------------------------------ routing
const routes = {
  dashboard: ["Dashboard", pageDashboard],
  chart: ["Grafico", pageChart],
  trades: ["Operazioni", pageTrades],
  model: ["Qualità modello", pageModel],
  backtest: ["Backtest", pageBacktest],
  research: ["Ricerca", pageResearch],
  settings: ["Impostazioni", pageSettings],
};

function renderNav(cur) {
  $("#nav").innerHTML = Object.entries(routes).map(([k, [t]]) => `<a href="#${k}" class="${k === cur ? "active" : ""}">${t}</a>`).join("");
}

async function route() {
  cleanup();
  const [name, query] = (location.hash.slice(1) || "dashboard").split("?");
  const params = new URLSearchParams(query || "");
  const r = routes[name] || routes.dashboard;
  renderNav(routes[name] ? name : "dashboard");
  app.innerHTML = `<div class="note">Caricamento…</div>`;
  try { await r[1](params); } catch (e) { app.innerHTML = `<div class="card err">${esc(e.message)}</div>`; }
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ header status (websocket)
function headerStatus(st) {
  if (!st) return;
  const hb = st.trader_heartbeat ? (Date.now() - new Date(st.trader_heartbeat).getTime()) / 1000 : null;
  const live = st.live;
  const fOk = st.forecaster && st.forecaster.ok;
  const lastC = (st.candles || []).map((c) => new Date(c.last).getTime()).sort()[0];
  const ctl = st.control || {};
  $("#hdr-status").innerHTML = [
    `<span class="badge ${hb !== null && hb < 90 ? "ok" : "bad"}">trader ${hb === null ? "offline" : hb < 90 ? "attivo" : `fermo da ${Math.round(hb)}s`}</span>`,
    `<span class="badge ${fOk ? "ok" : "bad"}">forecaster ${fOk ? short(st.forecaster.default_model) : "down"}</span>`,
    lastC ? `<span class="badge">ultima candela ${dt(lastC)}</span>` : "",
    ctl.paused ? `<span class="badge warn">IN PAUSA</span>` : "",
    live && live.halted ? `<span class="badge bad">KILL-SWITCH</span>` : "",
  ].join("");
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.type === "update") {
      headerStatus(d.status);
      window.dispatchEvent(new CustomEvent("status", { detail: d.status }));
    }
  };
  ws.onclose = () => setTimeout(connectWS, 3000);
}

// ------------------------------------------------------------------ equity chart
async function equityChart(el, runId) {
  const d = await api(`/api/runs/${runId}/equity`);
  const c = chartBase(el);
  const eq = c.addLineSeries({ color: "#38bdf8", lineWidth: 2, title: "Bot" });
  const bm = c.addLineSeries({ color: "#8b97a6", lineWidth: 1, lineStyle: LineStyle.Dashed, title: "Buy & Hold" });
  eq.setData(d.points.map((p) => ({ time: p.t, value: p.equity })));
  bm.setData(d.benchmark.map((p) => ({ time: p.t, value: p.value })));
  c.timeScale().fitContent();
  return { chart: c, eq, bm, data: d };
}

// ------------------------------------------------------------------ dashboard
async function pageDashboard() {
  const st = await api("/api/status");
  headerStatus(st);
  const L = st.live;
  if (!L) {
    app.innerHTML = `<div class="card"><h2>Nessuna sessione live</h2><div class="note">Il trader non ha ancora creato la sessione di paper trading. Controlla che i servizi <code>trader</code> e <code>forecaster</code> siano in esecuzione.</div></div>`;
    every(10000, route);
    return;
  }
  const strat = L.config.strategy;
  app.innerHTML = `
  <div class="row" style="justify-content:space-between;margin-bottom:12px">
    <div><h2 style="margin:0">Sessione live #${L.id} — ${esc(L.name)}</h2>
      <div class="note">Avviata ${dt(L.started_at)} · ultima barra elaborata ${dt(L.last_bar)} · strategia <b>${esc(strat.name)}</b> <code>${esc(JSON.stringify(strat.params))}</code> · modello ${esc(short(L.config.model))}</div></div>
    <div class="row">
      ${st.control.paused ? `<button data-a="resume" class="primary">Riprendi</button>` : `<button data-a="pause">Pausa (no nuovi ordini)</button>`}
      <button data-a="flatten" class="danger">Chiudi tutte le posizioni</button>
      ${L.halted ? `<button data-a="reset_halt">Reset kill-switch</button>` : ""}
      <button data-a="reset_run" class="danger">Nuova sessione</button>
    </div>
  </div>
  ${L.halted ? `<div class="card err" style="margin-bottom:12px">Kill-switch attivo: ${esc(L.halt_reason)}</div>` : ""}
  <div class="grid kpis" id="kpis"></div>
  <div class="grid cols3" style="margin-top:14px">
    <div class="card"><h3>Equity vs Buy &amp; Hold (equal-weight)</h3>
      <div class="legend"><span><i style="background:#38bdf8"></i>Bot</span><span><i style="background:#8b97a6"></i>Buy &amp; Hold</span></div>
      <div id="eq" class="chart"></div></div>
    <div class="card"><h3>Posizioni aperte</h3><div id="pos"></div><h3 style="margin-top:16px">Ordini in attesa (prossima apertura)</h3><div id="pend"></div></div>
  </div>
  <div class="grid cols2" style="margin-top:14px">
    <div class="card"><h3>Ultimi segnali operativi</h3><div id="sig"></div></div>
    <div class="card"><h3>Eventi</h3><div id="ev"></div></div>
  </div>`;
  app.querySelectorAll("button[data-a]").forEach((b) => b.addEventListener("click", async () => {
    const a = b.dataset.a;
    const msg = { flatten: "Chiudere tutte le posizioni alla prossima apertura?", reset_run: "Chiudere la sessione attuale e ripartire da capitale iniziale?" }[a];
    if (msg && !confirm(msg)) return;
    await api(`/api/control/${a}`, { method: "POST" });
    setTimeout(route, 500);
  }));
  const eqc = await equityChart($("#eq"), L.id);
  const fill = async (s) => {
    const live = s.live;
    const pts = eqc.data.points;
    const bmk = eqc.data.benchmark;
    const bhRet = bmk.length ? bmk[bmk.length - 1].value / bmk[0].value - 1 : null;
    $("#kpis").innerHTML = [
      kpi("Equity", money(live.equity), `capitale iniziale ${money(live.initial_cash)}`),
      kpi("P&amp;L totale", money(live.pnl), pct(live.return), sgn(live.pnl)),
      kpi("Buy &amp; Hold stesso periodo", pct(bhRet), "riferimento da battere", sgn(bhRet)),
      kpi("Drawdown", pct(live.drawdown), `picco ${money(live.peak)}`, live.drawdown < -0.05 ? "neg" : ""),
      kpi("Trade chiusi", live.n_trades, `win rate ${pct(live.win_rate, 0)}`),
      kpi("Commissioni pagate", money(live.fees_paid), `realizzato ${money(live.realized_pnl)}`),
    ].join("");
    $("#pos").innerHTML = table([
      { h: "Simbolo", k: "symbol" },
      { h: "Lato", f: (r) => (r.qty > 0 ? `<span class="pos">LONG</span>` : `<span class="neg">SHORT</span>`) },
      { h: "Qtà", num: 1, f: (r) => num(Math.abs(r.qty), 5) },
      { h: "Prezzo medio", num: 1, f: (r) => num(r.avg_price) },
      { h: "Ultimo", num: 1, f: (r) => num(r.last) },
      { h: "P&amp;L non real.", num: 1, f: (r) => `<span class="${sgn(r.unrealized)}">${num(r.unrealized)}</span>` },
      { h: "Stop / TP", num: 1, f: (r) => `${num(r.stop)} / ${num(r.take_profit)}` },
      { h: "Barre", num: 1, k: "bars" },
    ], live.positions, { empty: "Nessuna posizione aperta (flat)." });
    $("#pend").innerHTML = table([
      { h: "Simbolo", k: "symbol" }, { h: "Qtà", num: 1, f: (r) => `<span class="${sgn(r.qty)}">${num(r.qty, 5)}</span>` },
      { h: "Motivo", f: (r) => esc(r.reason) },
    ], live.pending_orders, { empty: "Nessun ordine in coda." });
  };
  await fill(st);
  const loadLists = async () => {
    const [sig, ev] = await Promise.all([api(`/api/runs/${L.id}/signals?limit=40&actionable=true`), api("/api/events?limit=40")]);
    $("#sig").innerHTML = table([
      { h: "Ora", f: (r) => dt(r.ts) }, { h: "Simbolo", k: "symbol" },
      { h: "Azione", f: (r) => `<b class="${r.action === "LONG" ? "pos" : r.action === "SHORT" ? "neg" : ""}">${r.action}</b>` },
      { h: "P(su)", num: 1, f: (r) => pct(r.p_up, 0) }, { h: "P(giù)", num: 1, f: (r) => pct(r.p_down, 0) },
      { h: "Motivo", f: (r) => `<span class="muted">${esc(r.reason)}</span>` },
    ], sig, { empty: "Ancora nessun segnale operativo.", maxh: 340 });
    $("#ev").innerHTML = table([
      { h: "Ora", f: (r) => dt(r.ts) }, { h: "", f: (r) => `<span class="badge ${r.level === "error" ? "bad" : r.level === "warning" ? "warn" : ""}">${r.source}</span>` },
      { h: "Messaggio", f: (r) => `<span style="white-space:normal">${esc(r.message)}</span>` },
    ], ev, { maxh: 340 });
  };
  await loadLists();
  const onStatus = async (e) => {
    const s = e.detail;
    if (!s.live || s.live.id !== L.id) return route();
    if (s.live.last_bar !== L.last_bar) { L.last_bar = s.live.last_bar; return route(); }
    await fill(s);
  };
  listen("status", onStatus);
  every(60000, loadLists);
}

// ------------------------------------------------------------------ chart page
async function pageChart(params) {
  const [meta, cfg, st] = await Promise.all([api("/api/meta"), api("/api/config"), api("/api/status")]);
  const sym = params.get("symbol") || cfg.symbols[0];
  const model = params.get("model") || cfg.model;
  app.innerHTML = `
  <div class="row" style="margin-bottom:12px">
    <label>Simbolo <select id="sym">${cfg.symbols.map((s) => `<option ${s === sym ? "selected" : ""}>${s}</option>`).join("")}</select></label>
    <label>Modello <select id="mdl">${meta.models.map((m) => `<option value="${m}" ${m === model ? "selected" : ""}>${short(m)}</option>`).join("")}</select></label>
    <div class="note" style="max-width:700px">Candele orarie reali. A destra il <b>ventaglio</b> previsto da Timestral per le prossime ${cfg.horizon} ore (mediana, 25–75%, 5–95%). Sulle barre passate: banda 10–90% prevista un'ora prima. Frecce: ordini eseguiti dal bot live.</div>
  </div>
  <div class="card"><div class="legend">
    <span><i style="background:#f59e0b"></i>mediana</span><span><i style="background:#a78bfa"></i>25–75%</span><span><i style="background:#64748b"></i>5–95%</span><span><i style="background:#38bdf8;opacity:.5"></i>10–90% (1 ora prima)</span>
  </div><div id="c" class="chart tall"></div></div>
  <div class="card" style="margin-top:14px"><h3>Copertura recente della banda 10–90% a 1 ora</h3><div id="cov" class="note"></div></div>`;
  $("#sym").onchange = () => (location.hash = `chart?symbol=${encodeURIComponent($("#sym").value)}&model=${encodeURIComponent($("#mdl").value)}`);
  $("#mdl").onchange = $("#sym").onchange;
  const [candles, fc] = await Promise.all([api(`/api/candles?symbol=${encodeURIComponent(sym)}&limit=600`), api(`/api/forecasts/latest?symbol=${encodeURIComponent(sym)}&model=${encodeURIComponent(model)}&n_hist=600`)]);
  const c = chartBase($("#c"));
  const cs = c.addCandlestickSeries({ upColor: "#22c55e", downColor: "#f05252", borderVisible: false, wickUpColor: "#22c55e", wickDownColor: "#f05252" });
  cs.setData(candles.map((k) => ({ time: k.t, open: k.o, high: k.h, low: k.l, close: k.c })));
  const line = (color, width, style, data) => { const s = c.addLineSeries({ color, lineWidth: width, lineStyle: style, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }); s.setData(data); return s; };
  const minT = candles.length ? candles[0].t : 0;
  const hist = fc.history.filter((h) => h.t >= minT);
  line("rgba(56,189,248,.45)", 1, LineStyle.Solid, hist.map((h) => ({ time: h.t, value: h.q10 })));
  line("rgba(56,189,248,.45)", 1, LineStyle.Solid, hist.map((h) => ({ time: h.t, value: h.q90 })));
  if (fc.future.length && candles.length) {
    const lastC = candles[candles.length - 1];
    const anchor = (k) => [{ time: fc.origin, value: lastC.c }, ...fc.future.map((f) => ({ time: f.t, value: f[k] }))];
    if (fc.origin === lastC.t) {
      line("#64748b", 1, LineStyle.Dashed, anchor("q05")); line("#64748b", 1, LineStyle.Dashed, anchor("q95"));
      line("#a78bfa", 1, LineStyle.Solid, anchor("q25")); line("#a78bfa", 1, LineStyle.Solid, anchor("q75"));
      line("#f59e0b", 2, LineStyle.Solid, anchor("q50"));
    }
  }
  // fill markers from live orders
  if (st.live) {
    const orders = await api(`/api/runs/${st.live.id}/orders?limit=1000`);
    const step = 3600;
    const mk = orders.filter((o) => o.status === "filled" && o.symbol === sym).map((o) => ({
      time: Math.floor(new Date(o.fill_ts).getTime() / 1000 / step) * step,
      position: o.side === "buy" ? "belowBar" : "aboveBar", color: o.side === "buy" ? "#22c55e" : "#f05252",
      shape: o.side === "buy" ? "arrowUp" : "arrowDown", text: `${o.side} ${num(o.qty, 4)}`,
    })).filter((m) => m.time >= minT).sort((a, b) => a.time - b.time);
    cs.setMarkers(mk);
  }
  c.timeScale().fitContent();
  // coverage of last N one-step forecasts
  const byT = new Map(candles.map((k) => [k.t, k.c]));
  const ev = hist.filter((h) => byT.has(h.t));
  const inside = ev.filter((h) => byT.get(h.t) >= h.q10 && byT.get(h.t) <= h.q90).length;
  $("#cov").innerHTML = ev.length ? `${inside}/${ev.length} chiusure dentro la banda 10–90% prevista un'ora prima = <b>${pct(inside / ev.length, 1)}</b> (ideale ≈ 80%).` : "Servono previsioni storiche: verranno accumulate dal trader live e dai backtest.";
}

// ------------------------------------------------------------------ trades page
async function pageTrades(params) {
  const st = await api("/api/status");
  const runs = await api("/api/runs?limit=200");
  const rid = +(params.get("run") || (st.live && st.live.id) || (runs[0] && runs[0].id));
  if (!rid) { app.innerHTML = `<div class="card note">Nessuna sessione.</div>`; return; }
  const [tr, od, sg] = await Promise.all([api(`/api/runs/${rid}/trades?limit=1000`), api(`/api/runs/${rid}/orders?limit=500`), api(`/api/runs/${rid}/signals?limit=500`)]);
  app.innerHTML = `
  <div class="row" style="margin-bottom:12px"><label>Sessione <select id="run">${runs.map((r) => `<option value="${r.id}" ${r.id === rid ? "selected" : ""}>#${r.id} ${esc(r.kind)} · ${esc(r.name)}</option>`).join("")}</select></label></div>
  <div class="card"><h3>Trade chiusi (${tr.length})</h3>${table([
    { h: "Simbolo", k: "symbol" }, { h: "Lato", f: (r) => `<span class="${r.side === "long" ? "pos" : "neg"}">${r.side}</span>` },
    { h: "Entrata", f: (r) => dt(r.entry_ts) }, { h: "Px entrata", num: 1, f: (r) => num(r.entry_price) },
    { h: "Uscita", f: (r) => dt(r.exit_ts) }, { h: "Px uscita", num: 1, f: (r) => num(r.exit_price) },
    { h: "Qtà", num: 1, f: (r) => num(r.qty, 5) }, { h: "Barre", num: 1, k: "bars" },
    { h: "Fee", num: 1, f: (r) => num(r.fees) }, { h: "P&amp;L netto", num: 1, f: (r) => `<b class="${sgn(r.pnl)}">${num(r.pnl)}</b>` },
    { h: "Uscita per", k: "exit_reason" },
  ], tr, { empty: "Nessun trade chiuso." })}</div>
  <div class="grid cols2" style="margin-top:14px">
    <div class="card"><h3>Ordini</h3>${table([
      { h: "Creato", f: (r) => dt(r.created_ts) }, { h: "Simbolo", k: "symbol" }, { h: "Lato", k: "side" },
      { h: "Qtà", num: 1, f: (r) => num(r.qty, 5) }, { h: "Stato", k: "status" }, { h: "Px fill", num: 1, f: (r) => num(r.fill_price) },
      { h: "Motivo", f: (r) => `<span class="muted">${esc(r.reason)}</span>` },
    ], od)}</div>
    <div class="card"><h3>Segnali</h3>${table([
      { h: "Ora", f: (r) => dt(r.ts) }, { h: "Simbolo", k: "symbol" }, { h: "Azione", k: "action" },
      { h: "P(su)", num: 1, f: (r) => pct(r.p_up, 0) }, { h: "P(giù)", num: 1, f: (r) => pct(r.p_down, 0) },
      { h: "Mediana", num: 1, f: (r) => pct(r.exp_ret, 2) }, { h: "Ampiezza 10-90", num: 1, f: (r) => pct(r.width, 2) },
      { h: "Motivo", f: (r) => `<span class="muted">${esc(r.reason)}</span>` },
    ], sg)}</div>
  </div>`;
  $("#run").onchange = () => (location.hash = `trades?run=${$("#run").value}`);
}

// ------------------------------------------------------------------ model quality
function qualityTable(symbols) {
  const rows = [];
  for (const [s, v] of Object.entries(symbols)) {
    if (!v) continue;
    for (const h of ["h1", "h4", "h12", "h24"]) if (v[h]) rows.push({ s, h, ...v[h] });
  }
  return table([
    { h: "Simbolo", k: "s" }, { h: "Orizz.", k: "h" }, { h: "N", num: 1, k: "n" },
    { h: "Accuratezza direzione", num: 1, f: (r) => `<b class="${r.dir_pvalue < 0.05 ? "pos" : ""}">${pct(r.dir_acc, 1)}</b>` },
    { h: "p-value (vs 50%)", num: 1, f: (r) => num(r.dir_pvalue, 3) },
    { h: "Acc. top-20% sicure", num: 1, f: (r) => pct(r.dir_acc_top20_confident, 1) },
    { h: "Skill CRPS vs naive", num: 1, f: (r) => `<span class="${sgn(r.crps_skill)}">${pct(r.crps_skill, 1)}</span>` },
    { h: "Copertura 80%", num: 1, f: (r) => pct(r.coverage_80, 1) }, { h: "Copertura 90%", num: 1, f: (r) => pct(r.coverage_90, 1) },
    { h: "Vol ρ Timestral", num: 1, f: (r) => num(r.vol_spearman, 3) }, { h: "Vol ρ naive", num: 1, f: (r) => num(r.vol_spearman_naive, 3) },
    { h: "PIT", f: (r) => `<div class="bars" style="height:22px;width:90px">${(r.pit_hist || []).map((x) => `<div style="height:${(100 * x) / Math.max(...r.pit_hist)}%"></div>`).join("")}</div>` },
  ], rows);
}

async function pageModel(params) {
  const days = +(params.get("days") || 30);
  const meta = await api("/api/meta");
  const cfg = await api("/api/config");
  const model = params.get("model") || cfg.model;
  app.innerHTML = `
  <div class="row" style="margin-bottom:12px">
    <label>Modello <select id="mdl">${meta.models.map((m) => `<option value="${m}" ${m === model ? "selected" : ""}>${short(m)}</option>`).join("")}</select></label>
    <label>Finestra <select id="days">${[7, 30, 90, 365].map((d) => `<option ${d === days ? "selected" : ""} value="${d}">ultimi ${d} giorni</option>`).join("")}</select></label>
  </div>
  <div class="card"><h3>Qualità previsioni (dati reali, nessun trading)</h3><div id="q" class="note">Calcolo…</div>
  <p class="note">Accuratezza direzione: quante volte il segno della mediana prevista coincide con il movimento reale (50% = moneta). Skill CRPS &gt; 0 = distribuzione migliore del random walk. Copertura: % di valori reali dentro l'intervallo (ideale 80% / 90%). Vol ρ: correlazione tra ampiezza prevista e movimento assoluto (previsione di volatilità). PIT piatto = ben calibrato.</p></div>
  <div class="card" style="margin-top:14px"><h3>Report di valutazione salvati</h3><div id="reps"></div></div>`;
  const go = () => (location.hash = `model?model=${encodeURIComponent($("#mdl").value)}&days=${$("#days").value}`);
  $("#mdl").onchange = go; $("#days").onchange = go;
  api(`/api/model-quality?days=${days}&model=${encodeURIComponent(model)}`).then((d) => ($("#q").innerHTML = qualityTable(d.symbols))).catch((e) => ($("#q").innerHTML = `<span class="err">${esc(e.message)}</span>`));
  const reps = await api("/api/reports?kind=forecast_eval");
  $("#reps").innerHTML = table([{ h: "#", k: "id" }, { h: "Nome", k: "name" }, { h: "Creato", f: (r) => dt(r.created_at) }], reps, { click: true, empty: "Nessun report." });
  bindRows($("#reps"), reps, async (r) => {
    const full = await api(`/api/reports/${r.id}`);
    $("#reps").innerHTML = `<button id="back">← elenco</button><h2 style="margin-top:10px">${esc(full.name)}</h2>${qualityTable(full.result.symbols)}`;
    $("#back").onclick = route;
  });
}

// ------------------------------------------------------------------ backtest
function metricsTable(m) {
  const rows = [
    ["Rendimento totale", pct(m.total_return), sgn(m.total_return)], ["Buy & Hold stesso periodo", pct(m.bh_total_return), sgn(m.bh_total_return)],
    ["Extra-rendimento vs B&H", pct(m.excess_return_vs_bh), sgn(m.excess_return_vs_bh)], ["CAGR", pct(m.cagr)],
    ["Sharpe", num(m.sharpe), sgn(m.sharpe)], ["Sharpe B&H", num(m.bh_sharpe)], ["Sortino", num(m.sortino)],
    ["Max drawdown", pct(m.max_drawdown), "neg"], ["Max drawdown B&H", pct(m.bh_max_drawdown)], ["Volatilità annua", pct(m.ann_vol)],
    ["Trade", m.n_trades], ["Win rate", pct(m.win_rate, 1)], ["Profit factor", num(m.profit_factor)],
    ["Durata media (barre)", num(m.avg_bars_held, 1)], ["Commissioni pagate", money(m.fees_paid)], ["Tempo a mercato", pct(m.time_in_market, 1)],
    ["p-value rendimento medio > 0 (t-test)", num(m.mean_ret_pvalue, 3)],
  ];
  if (m.mc && m.mc.p_value !== null && m.mc.p_value !== undefined) {
    rows.push(["Monte Carlo: percentile vs timing casuale", pct(m.mc.percentile / 100, 1)]);
    rows.push(["Monte Carlo: p-value", num(m.mc.p_value, 3), m.mc.p_value < 0.05 ? "pos" : ""]);
    rows.push(["Monte Carlo: Sharpe 95° percentile nullo", num(m.mc.null_p95)]);
  }
  if (m.halted) rows.push(["Kill-switch", esc(m.halted), "neg"]);
  return `<table>${rows.map(([a, b, c]) => `<tr><td class="muted">${a}</td><td class="num ${c || ""}">${b}</td></tr>`).join("")}</table>`;
}

function mcHist(m) {
  if (!m.mc || !m.mc.hist) return "";
  const max = Math.max(...m.mc.hist);
  const edges = m.mc.hist_edges;
  const actual = m.mc.actual_sharpe_vec;
  const pos = Math.min(100, Math.max(0, ((actual - edges[0]) / (edges[edges.length - 1] - edges[0])) * 100));
  return `<h3 style="margin-top:14px">Distribuzione Sharpe con timing casuale (1000 simulazioni)</h3>
  <div style="position:relative"><div class="bars" style="height:70px">${m.mc.hist.map((x) => `<div style="height:${(100 * x) / max}%;background:#475569"></div>`).join("")}</div>
  <div style="position:absolute;left:${pos}%;top:-4px;bottom:0;width:2px;background:#f59e0b" title="strategia"></div></div>
  <div class="row note" style="justify-content:space-between"><span>${num(edges[0])}</span><span style="color:#f59e0b">strategia: ${num(actual)} (Sharpe su calcolo vettoriale semplificato)</span><span>${num(edges[edges.length - 1])}</span></div>`;
}

async function showRun(container, rid) {
  const r = await api(`/api/runs/${rid}`);
  container.innerHTML = `<div class="row" style="justify-content:space-between"><h2>#${r.id} ${esc(r.name)}</h2><a href="#trades?run=${r.id}">vedi operazioni →</a></div>
  <div class="note" style="margin-bottom:10px"><code>${esc(JSON.stringify({ model: r.config.model, strategy: r.config.strategy, params: r.config.params, start: r.config.start, end: r.config.end }))}</code></div>
  <div class="grid cols3"><div><div class="legend"><span><i style="background:#38bdf8"></i>Strategia</span><span><i style="background:#8b97a6"></i>Buy &amp; Hold</span></div><div id="req" class="chart"></div>${mcHist(r.metrics)}</div><div>${metricsTable(r.metrics)}</div></div>`;
  await equityChart($("#req", container), rid);
}

async function pageBacktest(params) {
  const meta = await api("/api/meta");
  const cfg = await api("/api/config");
  const strategies = meta.strategies;
  app.innerHTML = `
  <div class="card"><h3>Nuovo backtest</h3>
    <div class="form">
      <label>Strategia <select id="bs">${Object.keys(strategies).map((s) => `<option>${s}</option>`).join("")}</select></label>
      <label>Modello previsioni <select id="bm">${meta.models.map((m) => `<option value="${m}" ${m === cfg.model ? "selected" : ""}>${short(m)}</option>`).join("")}</select></label>
      <label>Inizio (UTC) <input id="bstart" type="date" value="${meta.holdout_start}"></label>
      <label>Fine (vuoto = oggi) <input id="bend" type="date"></label>
      <label>Simboli <input id="bsym" value="${cfg.symbols.join(",")}"></label>
      <label>Fee / slippage per lato <input id="bcost" value="0.001 / 0.0003"></label>
    </div>
    <label style="margin-top:12px">Parametri strategia (JSON) <textarea id="bp"></textarea></label>
    <div class="row" style="margin-top:10px"><button class="primary" id="go">Avvia backtest</button>
      <span class="note">Usa le previsioni in cache: se mancano per il periodo/modello scelto, avvia prima "Genera previsioni" nella pagina Ricerca. I prezzi sono reali; ordini eseguiti all'apertura della barra successiva, con fee e slippage.</span></div>
    <div id="berr" class="err"></div>
  </div>
  <div class="card" style="margin-top:14px"><h3>Job</h3><div id="jobs"></div></div>
  <div class="card" style="margin-top:14px"><h3>Backtest salvati</h3><div id="runs"></div></div>
  <div class="card" style="margin-top:14px" id="detail"><div class="note">Seleziona un backtest per vedere i dettagli.</div></div>`;
  const setP = () => ($("#bp").value = JSON.stringify(strategies[$("#bs").value], null, 1));
  $("#bs").onchange = setP; setP();
  $("#go").onclick = async () => {
    $("#berr").textContent = "";
    try {
      const [fee, slip] = $("#bcost").value.split("/").map((x) => parseFloat(x));
      const p = { strategy: $("#bs").value, model: $("#bm").value, params: JSON.parse($("#bp").value), start: $("#bstart").value,
        end: $("#bend").value || null, symbols: $("#bsym").value.split(",").map((s) => s.trim()).filter(Boolean), fee, slippage: slip, mc: 1000,
        feature_set: cfg.feature_set, context_len: cfg.context_len };
      await api("/api/jobs", { method: "POST", body: JSON.stringify({ kind: "backtest", params: p }) });
      loadJobs();
    } catch (e) { $("#berr").textContent = e.message; }
  };
  const loadJobs = async () => {
    const jobs = await api("/api/jobs?limit=15");
    $("#jobs").innerHTML = table([
      { h: "#", k: "id" }, { h: "Tipo", k: "kind" }, { h: "Stato", f: (r) => `<span class="badge ${r.status === "done" ? "ok" : r.status === "error" ? "bad" : "warn"}">${r.status}</span>` },
      { h: "Avanzamento", f: (r) => `<div class="progress"><div style="width:${(r.progress || 0) * 100}%"></div></div>` },
      { h: "Messaggio", f: (r) => `<span class="muted" style="white-space:normal">${esc((r.message || "").slice(0, 160))}</span>` },
      { h: "Risultato", f: (r) => (r.result && r.result.run_id ? `<a href="#backtest?run=${r.result.run_id}">run #${r.result.run_id}</a>` : "") },
      { h: "Creato", f: (r) => dt(r.created_at) },
    ], jobs, { maxh: 260 });
  };
  const loadRuns = async () => {
    const runs = await api("/api/runs?kind=backtest&limit=200");
    $("#runs").innerHTML = table([
      { h: "#", k: "id" }, { h: "Nome", k: "name" }, { h: "Periodo", f: (r) => `${(r.config.start || "").slice(0, 10)} → ${(r.config.end || "oggi").slice(0, 10)}` },
      { h: "Rend.", num: 1, f: (r) => `<span class="${sgn(r.metrics.total_return)}">${pct(r.metrics.total_return)}</span>` },
      { h: "B&amp;H", num: 1, f: (r) => pct(r.metrics.bh_total_return) },
      { h: "Sharpe", num: 1, f: (r) => num(r.metrics.sharpe) }, { h: "Max DD", num: 1, f: (r) => pct(r.metrics.max_drawdown) },
      { h: "Trade", num: 1, f: (r) => r.metrics.n_trades }, { h: "MC p", num: 1, f: (r) => num(r.metrics.mc && r.metrics.mc.p_value, 3) },
    ], runs, { click: true, maxh: 320 });
    bindRows($("#runs"), runs, (r) => (location.hash = `backtest?run=${r.id}`));
  };
  await Promise.all([loadJobs(), loadRuns()]);
  every(4000, loadJobs);
  every(15000, loadRuns);
  if (params.get("run")) { await showRun($("#detail"), +params.get("run")); $("#detail").scrollIntoView(); }
}

// ------------------------------------------------------------------ research (suite reports)
async function pageResearch(params) {
  const meta = await api("/api/meta");
  const reps = await api("/api/reports?kind=suite");
  const rid = +(params.get("report") || (reps[0] && reps[0].id) || 0);
  app.innerHTML = `
  <div class="grid cols2">
    <div class="card"><h3>Studio completo (tune → validation → holdout)</h3>
      <div class="note">Per ogni modello prova ~70 configurazioni sul periodo di tuning, porta la migliore di ogni famiglia sul periodo di validazione (con test Monte Carlo e Deflated Sharpe) e la esegue <b>una sola volta</b> sull'holdout successivo al rilascio di Timestral (${meta.holdout_start}). Confronto con buy &amp; hold, SMA e lo stesso approccio con previsione naive.</div>
      <div class="form" style="margin-top:10px">
        <label>Modelli <input id="sm" value="naive-rw,mistralai/Timestral-5M,mistralai/Timestral-28M,mistralai/Timestral-123M"></label>
        <label>Tuning <input id="st" value="2025-01-01:2026-01-01"></label>
        <label>Validazione <input id="sv" value="2026-01-01:${meta.holdout_start}"></label>
        <label>Holdout <input id="sh" value="${meta.holdout_start}:"></label>
      </div>
      <div class="row" style="margin-top:10px"><button class="primary" id="suite">Avvia studio</button><button id="gen">Genera previsioni mancanti</button><button id="evalb">Valuta previsioni</button></div>
      <div id="serr" class="err"></div></div>
    <div class="card"><h3>Report</h3><div id="reps"></div></div>
  </div>
  <div id="rep" style="margin-top:14px"></div>`;
  const models = () => $("#sm").value.split(",").map((s) => s.trim()).filter(Boolean);
  const rng = (v) => { const [a, b] = v.split(":"); return [a, b || null]; };
  const post = async (kind, p) => { try { await api("/api/jobs", { method: "POST", body: JSON.stringify({ kind, params: p }) }); location.hash = "backtest"; } catch (e) { $("#serr").textContent = e.message; } };
  $("#suite").onclick = () => post("suite", { models: models(), tune: rng($("#st").value), validation: rng($("#sv").value), holdout: rng($("#sh").value), mc: 1000 });
  $("#gen").onclick = () => post("generate", { models: models(), start: "2024-12-01" });
  $("#evalb").onclick = () => post("evaluate", { models: models(), start: meta.holdout_start });
  $("#reps").innerHTML = table([{ h: "#", k: "id" }, { h: "Nome", k: "name" }, { h: "Creato", f: (r) => dt(r.created_at) }], reps, { click: true, empty: "Nessuno studio ancora." });
  bindRows($("#reps"), reps, (r) => (location.hash = `research?report=${r.id}`));
  if (!rid) return;
  const rep = (await api(`/api/reports/${rid}`)).result;
  const V = rep.verdict || { candidates: [] };
  const chk = { sharpe_gt_1: "Sharpe > 1", beats_buy_hold_sharpe: "batte B&H (Sharpe)", beats_sma_sharpe: "batte SMA", beats_naive_forecaster_same_strategy: "batte previsione naive",
    "mc_p_below_0.05": "Monte Carlo p < 0.05", validation_also_positive: "positiva anche in validazione", enough_trades_200: "≥ 200 trade" };
  const cols = [
    { h: "Modello", f: (r) => esc(short(r.model)) }, { h: "Strategia", k: "strategy" },
    { h: "Parametri", f: (r) => `<code style="font-size:11px">${esc(JSON.stringify(r.params))}</code>` },
    { h: "Rend.", num: 1, f: (r) => `<span class="${sgn(r.total_return)}">${pct(r.total_return)}</span>` }, { h: "B&amp;H", num: 1, f: (r) => pct(r.bh_total_return) },
    { h: "Sharpe", num: 1, f: (r) => `<b class="${sgn(r.sharpe)}">${num(r.sharpe)}</b>` }, { h: "Max DD", num: 1, f: (r) => pct(r.max_drawdown) },
    { h: "Trade", num: 1, k: "n_trades" }, { h: "Fee", num: 1, f: (r) => num(r.fees_paid, 0) },
    { h: "MC p", num: 1, f: (r) => (r.mc_pvalue === undefined ? "" : `<span class="${r.mc_pvalue < 0.05 ? "pos" : ""}">${num(r.mc_pvalue, 3)}</span>`) },
  ];
  const gcol = { h: "Rend. lordo (senza costi)", num: 1, f: (r) => (r.gross_return === undefined ? "" : `<span class="${sgn(r.gross_return)}">${pct(r.gross_return)}</span>`) };
  const holdCols = [...cols, gcol, { h: "", f: (r) => (r.run_id ? `<a href="#backtest?run=${r.run_id}">dettagli</a>` : "") }];
  const valCols = [...cols, gcol, { h: "Sharpe tune", num: 1, f: (r) => num(r.tune_sharpe) }, { h: "DSR", num: 1, f: (r) => num(r.dsr, 3) }];
  const tuneTop = [...rep.tune].sort((a, b) => b.sharpe - a.sharpe).slice(0, 20);
  const pr = rep.periods;
  $("#rep").innerHTML = `
  <div class="card verdict ${V.any_passed ? "yes" : "no"}"><h2>${V.any_passed ? "Almeno una configurazione supera TUTTI i criteri sull'holdout" : "Nessuna configurazione supera tutti i criteri sull'holdout"}</h2>
    <div class="note">Periodi: tuning ${pr.tune.join(" → ")} · validazione ${pr.validation.join(" → ")} · holdout ${pr.holdout[0]} → ${pr.holdout[1] || "oggi"} · ${rep.tune.length} configurazioni provate · ${num(rep.seconds / 60, 1)} min</div>
    ${V.candidates.map((c) => `<div style="margin-top:8px"><b>${esc(short(c.model))} · ${esc(c.strategy)}</b> — Sharpe holdout ${num(c.holdout_sharpe)}, rendimento ${pct(c.holdout_return)}<br>${Object.entries(c.checks).map(([k, v]) => `<span class="check ${v ? "pos" : "neg"}">${v ? "✔" : "✘"} ${chk[k] || k}</span>`).join("")}</div>`).join("")}
  </div>
  <div class="card" style="margin-top:14px"><h3>Holdout (eseguito una sola volta)</h3>${table(holdCols, rep.holdout)}
    <h3 style="margin-top:14px">Baseline holdout</h3>${table(cols.slice(1, -1).concat([{ h: "", f: (r) => (r.run_id ? `<a href="#backtest?run=${r.run_id}">dettagli</a>` : "") }]), rep.baselines.holdout || [])}</div>
  <div class="card" style="margin-top:14px"><h3>Validazione (migliore per famiglia scelta sul tuning)</h3>${table(valCols, rep.validation)}
    <h3 style="margin-top:14px">Baseline validazione</h3>${table(cols.slice(1, -1), rep.baselines.validation || [])}</div>
  <div class="card" style="margin-top:14px"><h3>Tuning — top 20 di ${rep.tune.length}</h3>${table(cols.slice(0, -1), tuneTop)}
    <h3 style="margin-top:14px">Baseline tuning</h3>${table(cols.slice(1, -1), rep.baselines.tune || [])}</div>
  <div class="card" style="margin-top:14px"><h3>Copertura previsioni in cache</h3><pre class="note">${esc(JSON.stringify(rep.coverage, null, 1))}</pre></div>`;
}

// ------------------------------------------------------------------ settings
async function pageSettings() {
  const [meta, cfg] = await Promise.all([api("/api/meta"), api("/api/config")]);
  app.innerHTML = `
  <div class="card"><h3>Configurazione del bot live (paper)</h3>
    <div class="form">
      <label>Strategia <select id="s">${Object.keys(meta.strategies).map((s) => `<option ${s === cfg.strategy.name ? "selected" : ""}>${s}</option>`).join("")}</select></label>
      <label>Modello <select id="m">${meta.models.map((m) => `<option value="${m}" ${m === cfg.model ? "selected" : ""}>${short(m)}</option>`).join("")}</select></label>
      <label>Simboli <input id="sy" value="${cfg.symbols.join(",")}"></label>
      <label>Feature set <select id="fs">${["close", "cov"].map((f) => `<option ${f === cfg.feature_set ? "selected" : ""}>${f}</option>`).join("")}</select></label>
      <label>Contesto (barre) <input id="cl" type="number" value="${cfg.context_len}"></label>
      <label>Orizzonte previsione <input id="hz" type="number" value="${cfg.horizon}"></label>
      <label>Fee per lato <input id="fee" type="number" step="0.0001" value="${cfg.fee}"></label>
      <label>Slippage per lato <input id="sl" type="number" step="0.0001" value="${cfg.slippage}"></label>
      <label>Capitale iniziale (nuove sessioni) <input id="ic" type="number" value="${cfg.initial_cash}"></label>
    </div>
    <div class="grid cols2" style="margin-top:12px">
      <label>Parametri strategia (JSON) <textarea id="sp">${esc(JSON.stringify(cfg.strategy.params, null, 1))}</textarea></label>
      <label>Rischio (JSON) <textarea id="rk">${esc(JSON.stringify(cfg.risk, null, 1))}</textarea></label>
    </div>
    <div class="row" style="margin-top:10px"><button class="primary" id="save">Salva</button><button id="def">Parametri di default della strategia</button><span id="msg" class="note"></span></div>
    <p class="note">Le modifiche si applicano dalla prossima candela: posizioni e cassa restano. Cambiare strategia azzera solo lo stato interno della strategia (con warm-up automatico).
      <b>Rischio</b>: <code>max_frac_per_symbol</code> (null = 1/N simboli), <code>rebalance_band</code>, <code>daily_loss_limit</code> (stop operatività fino al giorno dopo), <code>max_drawdown_halt</code> (kill-switch: chiude tutto e si ferma fino a reset manuale).
      La modalità è sempre <b>paper</b>: nessuna chiave API reale, nessun ordine verso exchange.</p>
  </div>`;
  $("#def").onclick = () => ($("#sp").value = JSON.stringify(meta.strategies[$("#s").value], null, 1));
  $("#s").onchange = $("#def").onclick;
  $("#save").onclick = async () => {
    try {
      const body = { symbols: $("#sy").value.split(",").map((s) => s.trim()).filter(Boolean), model: $("#m").value, feature_set: $("#fs").value,
        context_len: +$("#cl").value, horizon: +$("#hz").value, initial_cash: +$("#ic").value, fee: +$("#fee").value, slippage: +$("#sl").value,
        strategy: { name: $("#s").value, params: JSON.parse($("#sp").value) }, risk: JSON.parse($("#rk").value) };
      await api("/api/config", { method: "PUT", body: JSON.stringify(body) });
      $("#msg").innerHTML = `<span class="pos">Salvato.</span>`;
    } catch (e) { $("#msg").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  };
}

connectWS();
route();
