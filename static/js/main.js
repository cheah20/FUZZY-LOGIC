// ASTDSS Terminal — Main Interaction Layer
// Handles: ticker analysis, live chart, backtest run, DOM updates, clock

document.addEventListener('DOMContentLoaded', () => {

    // SECTION 1 — LIVE ANALYSIS (Dashboard)

    const tickerInput   = document.getElementById('ticker-input');
    const analyzeBtn    = document.getElementById('analyze-btn');
    const analyzeStatus = document.getElementById('analyze-status');

    if (analyzeBtn && tickerInput) {
        analyzeBtn.addEventListener('click', runAnalysis);
        tickerInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') runAnalysis();
        });
    }

    async function runAnalysis() {
        const ticker = (tickerInput.value || '').trim().toUpperCase();
        if (!ticker) { setAnalyzeStatus('Enter a ticker symbol first.', 'warn'); return; }

        setAnalyzeLoading(true);
        setAnalyzeStatus('Fetching data for ' + ticker + '…', 'info');

        try {
            const res  = await fetch('/api/analyze?ticker=' + encodeURIComponent(ticker));
            const data = await res.json();

            if (!res.ok || data.error) {
                setAnalyzeStatus(data.error || 'Analysis failed.', 'error');
                return;
            }

            updateDashboardMetrics(data);
            if (data.chart_data) drawPriceChart(data.chart_data);
            updateExpertPanel(data);
            setAnalyzeStatus(ticker + ' — ' + data.source, 'ok');
            localStorage.setItem('astdss:lastTicker', ticker);

        } catch (err) {
            setAnalyzeStatus('Network error. Is the server running?', 'error');
        } finally {
            setAnalyzeLoading(false);
        }
    }

    function setAnalyzeLoading(on) {
        if (!analyzeBtn) return;
        analyzeBtn.disabled = on;
        analyzeBtn.style.opacity = on ? '0.6' : '1';
    }

    function setAnalyzeStatus(msg, type) {
        if (!analyzeStatus) return;
        const colors = { ok: 'var(--synced-green)', error: 'var(--strong-sell)',
                         warn: 'var(--optimal-processing)', info: 'rgba(255,255,255,.4)' };
        analyzeStatus.textContent = msg;
        analyzeStatus.style.color = colors[type] || colors.info;
    }

    // Update stat cards
    function updateDashboardMetrics(d) {
        _setEl('live-ticker-label', d.ticker + ' PRICE');
        _setEl('live-price',  _fmt(d.price));
        _setEl('live-rsi',    d.rsi != null  ? d.rsi.toFixed(1)  : '—');
        _setEl('live-atr',    d.atr  != null ? 'ATR ' + d.atr.toFixed(2) : '—');
        _setEl('live-source', d.source ? d.source.replace(/yfinance/, 'LIVE') : 'LIVE');

        const sigEl  = document.getElementById('live-signal');
        const regEl  = document.getElementById('live-regime');
        if (sigEl) {
            sigEl.textContent = (d.expert_label || '—').toUpperCase();
            sigEl.style.color = _signalColor(d.expert_label);
        }
        if (regEl) regEl.textContent = d.regime_display || '';

        _setEl('agg-signal', (d.expert_label || '—').toUpperCase());
        const aggScore = document.getElementById('agg-score');
        if (aggScore) {
            aggScore.textContent = 'score ' + (d.expert_score >= 0 ? '+' : '') +
                d.expert_score.toFixed(3) + ' · ' + (d.regime_display || '');
        }
    }

    // Expert Reasoning Panel live update
    function updateExpertPanel(d) {
        // Regime badge (unique id — a generic class selector here would match
        // the Agent Pipeline's "Cognition: Processing" badge instead)
        const regimeBadge = document.getElementById('dash-regime-badge');
        if (regimeBadge && d.regime_display) regimeBadge.textContent = d.regime_display;

        // Input display bars (MASKED state)
        if (d.active_features && d.masked_features) {
            document.querySelectorAll('[id^="input-bar-"]').forEach(bar => {
                const feat = bar.dataset.feature;
                if (feat) {
                    const masked = d.masked_features.includes(feat);
                    bar.style.opacity = masked ? '0.25' : '1';
                }
            });
        }
    }

    // Price line chart (SVG)
    function drawPriceChart(cd) {
        const container = document.getElementById('live-chart-container');
        if (!container || !cd.close || !cd.close.length) return;

        const W = 700, H = 280, PAD = 20, VPAD = 30;
        const closes  = cd.close;
        const volumes = cd.volume || [];
        const n       = closes.length;

        const minP = Math.min(...closes);
        const maxP = Math.max(...closes);
        const rng  = maxP - minP || 1;

        // Map price -> Y (SVG Y is inverted)
        const py = p => PAD + (1 - (p - minP) / rng) * (H - PAD * 2 - VPAD);
        const px = i => (i / (n - 1)) * W;

        // Build SVG polyline points
        let pts = closes.map((p, i) => px(i).toFixed(1) + ',' + py(p).toFixed(1)).join(' ');
        let first = closes[0], last = closes[n - 1];
        let isUp  = last >= first;
        let lineColor = isUp ? 'var(--synced-green)' : 'var(--strong-sell)';

        // Volume bars (bottom strip)
        let volBars = '';
        const VH = VPAD - 4, VY0 = H - VH - 2;
        volumes.forEach((v, i) => {
            const bw  = Math.max(1, W / n - 1);
            const bh  = Math.round(v / 100 * VH);
            const bx  = px(i) - bw / 2;
            const col = i > 0 && closes[i] >= closes[i - 1]
                ? 'rgba(16,185,129,.35)' : 'rgba(239,68,68,.30)';
            volBars += `<rect x="${bx.toFixed(1)}" y="${(VY0 + VH - bh).toFixed(1)}" width="${bw.toFixed(1)}" height="${bh}" fill="${col}" rx="0.5"/>`;
        });

        // Price area gradient fill
        let areaPath = `M${px(0).toFixed(1)},${py(closes[0]).toFixed(1)} `;
        closes.forEach((p, i) => { if (i > 0) areaPath += `L${px(i).toFixed(1)},${py(p).toFixed(1)} `; });
        areaPath += `L${W},${py(closes[n-1]).toFixed(1)} L${W},${H} L0,${H} Z`;

        // Current price label
        const lastX  = px(n - 1).toFixed(1);
        const lastY  = py(last).toFixed(1);
        const priceTag = `<line x1="0" y1="${lastY}" x2="${W}" y2="${lastY}" stroke="${lineColor}" stroke-width="0.5" stroke-dasharray="3,3" opacity="0.4"/>` +
            `<rect x="${W - 80}" y="${parseFloat(lastY) - 10}" width="78" height="18" rx="3" fill="rgba(0,0,0,.7)" stroke="${lineColor}" stroke-width="0.8"/>` +
            `<text x="${W - 41}" y="${parseFloat(lastY) + 3.5}" text-anchor="middle" fill="${lineColor}" style="font:700 9px 'JetBrains Mono'">${_fmt(last)}</text>`;

        // Date labels (first, mid, last)
        let dateLbls = '';
        if (cd.dates && cd.dates.length) {
            [[0, cd.dates[0]], [Math.floor(n/2), cd.dates[Math.floor(n/2)]], [n-1, cd.dates[n-1]]]
                .forEach(([i, d]) => {
                    if (!d) return;
                    const x = px(i).toFixed(1);
                    dateLbls += `<text x="${x}" y="${H - 2}" text-anchor="middle" fill="rgba(255,255,255,.25)" style="font:8px 'JetBrains Mono'">${d.slice(5)}</text>`;
                });
        }

        container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:100%" preserveAspectRatio="none">
  <defs>
    <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${lineColor}" stop-opacity=".18"/>
      <stop offset="100%" stop-color="${lineColor}" stop-opacity="0"/>
    </linearGradient>
    <pattern id="chartGrid" width="70" height="56" patternUnits="userSpaceOnUse">
      <line x1="70" y1="0" x2="70" y2="${H}" stroke="rgba(255,255,255,.03)" stroke-width="1"/>
      <line x1="0" y1="56" x2="${W}" y2="56" stroke="rgba(255,255,255,.03)" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="${W}" height="${H}" fill="url(#chartGrid)"/>
  ${volBars}
  <path d="${areaPath}" fill="url(#priceGrad)"/>
  <polyline points="${pts}" fill="none" stroke="${lineColor}" stroke-width="1.8" stroke-linejoin="round"/>
  ${priceTag}
  ${dateLbls}
</svg>`;
    }


    // SECTION 2 — BACKTESTING

    const btRunBtn     = document.getElementById('bt-run-btn');
    const btTickerEl   = document.getElementById('bt-ticker');
    const btStatusEl   = document.getElementById('bt-status');
    const btLivePanel  = document.getElementById('bt-live-results');
    const btStaticPanel= document.getElementById('bt-static-stats');

    // Period selector buttons
    let activePeriod = '2y';

    // Evaluation Period dates — real, computed from today (not hardcoded demo dates)
    function _updateEvalPeriod() {
        const years = parseInt(activePeriod, 10) || 2;
        const end   = new Date();
        const start = new Date(end);
        start.setFullYear(start.getFullYear() - years);
        const fmt = d => d.toISOString().slice(0, 10);
        _setEl('bt-eval-start', fmt(start));
        _setEl('bt-eval-end', fmt(end));
    }
    _updateEvalPeriod();

    document.querySelectorAll('.bt-period-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            activePeriod = btn.dataset.period;
            document.querySelectorAll('.bt-period-btn').forEach(b => {
                b.classList.remove('bt-period-btn--active');
                b.style.background = '';
                b.style.borderColor = '';
                b.style.color = '';
            });
            btn.classList.add('bt-period-btn--active');
            btn.style.background    = 'rgba(255,228,201,.12)';
            btn.style.borderColor   = 'rgba(255,228,201,.20)';
            btn.style.color         = 'var(--gold-soft)';
            _updateEvalPeriod();
        });
    });

    if (btRunBtn) {
        btRunBtn.addEventListener('click', runBacktest);
    }
    if (btTickerEl) {
        btTickerEl.addEventListener('keydown', e => { if (e.key === 'Enter') runBacktest(); });
        btTickerEl.addEventListener('input', e => {
            btTickerEl.value = btTickerEl.value.toUpperCase();
        });
    }

    async function runBacktest() {
        const ticker = (btTickerEl ? btTickerEl.value : 'AAPL').trim().toUpperCase();
        if (!ticker) { _btStatus('Enter a ticker symbol.', 'warn'); return; }

        _btLoading(true);
        _btStatus('Running backtest for ' + ticker + '…', 'info');

        try {
            const res  = await fetch('/api/backtest?ticker=' + encodeURIComponent(ticker) +
                                     '&period=' + activePeriod);
            const data = await res.json();

            if (!res.ok || data.error) {
                _btStatus(data.error || 'Backtest failed.', 'error');
                return;
            }

            displayBacktestResults(data);
            _btStatus('Done — ' + data.total_bars + ' bars (' + activePeriod.toUpperCase() + ')', 'ok');

        } catch (err) {
            _btStatus('Network error. Is the server running?', 'error');
        } finally {
            _btLoading(false);
        }
    }

    function displayBacktestResults(d) {
        // Show live panel, hide static
        if (btLivePanel)  btLivePanel.style.display  = 'block';
        if (btStaticPanel) btStaticPanel.style.display = 'none';

        const isExpertBetter = d.expert.total_return >= d.baseline.total_return;

        _setEl('bt-result-ticker', d.ticker);
        _setEl('bt-result-bars',   d.total_bars + ' bars · 70/30 split');
        _setEl('bt-expert-return',  _pct(d.expert.total_return));
        _setEl('bt-expert-sharpe',  'Sharpe ' + d.expert.sharpe.toFixed(3));
        _setEl('bt-baseline-return',_pct(d.baseline.total_return));
        _setEl('bt-baseline-sharpe','Sharpe ' + d.baseline.sharpe.toFixed(3));
        _setEl('bt-bah-return',     _pct(d.buy_hold_return));
        _setEl('bt-winrate',        (d.expert.win_rate || 0).toFixed(1) + '%');
        _setEl('bt-drawdown',       'DD -' + (d.expert.max_drawdown || 0).toFixed(1) + '%');
        _setEl('bt-chart-ticker',   d.ticker);

        const winnerBadge = document.getElementById('bt-result-winner');
        if (winnerBadge) {
            winnerBadge.textContent = isExpertBetter ? 'EXPERT WINS' : 'BASELINE WINS';
            winnerBadge.className   = 'badge ' + (isExpertBetter ? 'badge--synced' : 'badge--idle');
        }

        // Color the expert return
        const expRetEl = document.getElementById('bt-expert-return');
        if (expRetEl) expRetEl.style.color = d.expert.total_return >= 0 ? 'var(--synced-green)' : 'var(--strong-sell)';

        // Draw equity curve chart
        drawEquityCurve(d);

        // Update Backtest Trade Log
        const logTbody   = document.getElementById('bt-log-tbody');
        const logCount   = document.getElementById('bt-log-count');
        const closeTrades = (d.expert.trades || []).filter(t => t.type === 'close');
        if (logTbody && closeTrades.length) {
            // Most recent trades first — a demo audience wants to see what just
            // happened, not the oldest trades from years ago.
            const sample = closeTrades.slice(-5).reverse();
            logTbody.innerHTML = sample.map((t) => {
                const dt = d.dates[t.bar_index] || '—';
                const exitType = t.pnl_pct > 2  ? 'Profit Take Exit' :
                                 t.pnl_pct < -2 ? 'Stop-Loss Exit'    :
                                                   'Signal Exit';
                const outcome = t.pnl_pct > 0  ? 'Profit Lock' : 'Stop-Loss';
                const ocol    = t.pnl_pct > 0  ? 'var(--synced-green)' : 'var(--strong-sell)';
                return `<tr>
                    <td>${dt}</td>
                    <td>${exitType}</td>
                    <td style="color:var(--gold-soft);">${t.label || '—'}</td>
                    <td style="color:${ocol};">${outcome} ${t.pnl_pct > 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}%</td>
                </tr>`;
            }).join('');
            if (logCount) logCount.textContent = `Showing ${sample.length} most recent of ${closeTrades.length} trades`;
        }
        // Refresh the real agentic log panel after backtest completes
        loadAgenticLog();

        // Update BASELINE vs EXPERT comparison panel
        const cmpPanel = document.getElementById('bt-cmp-panel');
        if (cmpPanel) {
            const ew  = d.expert.total_return >= d.baseline.total_return;
            const def = (v, u) => (v >= 0 ? '+' : '') + v.toFixed(2) + u;
            const bar = (ev, bv) => {
                const mx = Math.max(Math.abs(ev), Math.abs(bv), 0.01);
                const ep = Math.min(100, Math.round(Math.abs(ev) / mx * 100));
                const bp = Math.min(100, Math.round(Math.abs(bv) / mx * 100));
                return `<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:4px;">
                    <div><div style="font:8px var(--font-mono);color:var(--gold-soft);margin-bottom:2px;">EXPERT</div>
                    <div class="progress-bar" style="height:3px;"><div class="progress-bar__fill" style="width:${ep}%;background:var(--gold-soft);"></div></div>
                    </div>
                    <div><div style="font:8px var(--font-mono);color:var(--secondary);margin-bottom:2px;">BASELINE</div>
                    <div class="progress-bar" style="height:3px;"><div class="progress-bar__fill" style="width:${bp}%;background:var(--secondary);"></div></div>
                    </div>
                </div>`;
            };
            cmpPanel.innerHTML = `
            <div class="section-header" style="margin-bottom:6px;">
              <span class="section-header__title">BASELINE vs EXPERT</span>
              <span class="badge ${ew ? 'badge--synced' : 'badge--idle'}" style="margin-left:auto;font-size:8px;">${ew ? 'EXPERT WINS' : 'BASELINE WINS'}</span>
            </div>
            <div style="font-family:var(--font-mono);font-size:8px;color:rgba(255,255,255,.25);line-height:1.5;margin-bottom:12px;">
              The Human Expert Layer changes and explains decisions under risky/extreme market regimes — it is not guaranteed to outperform the baseline on every dataset or window.
            </div>
            <div style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                <span style="font:9px var(--font-mono);color:rgba(255,255,255,.4);text-transform:uppercase;">Total Return</span>
                <span style="font:8px var(--font-mono);color:rgba(255,255,255,.25);">B&H ${def(d.buy_hold_return, '%')}</span>
              </div>
              <div style="display:flex;justify-content:space-between;">
                <span style="font:600 11px var(--font-mono);color:${d.expert.total_return>=0?'var(--synced-green)':'var(--strong-sell)'}">${def(d.expert.total_return,'%')}</span>
                <span style="font:11px var(--font-mono);color:var(--secondary)">${def(d.baseline.total_return,'%')}</span>
              </div>${bar(d.expert.total_return, d.baseline.total_return)}
            </div>
            <div style="margin-bottom:10px;">
              <div style="font:9px var(--font-mono);color:rgba(255,255,255,.4);text-transform:uppercase;margin-bottom:2px;">Win Rate</div>
              <div style="display:flex;justify-content:space-between;">
                <span style="font:600 11px var(--font-mono);color:var(--gold-soft)">${(d.expert.win_rate||0).toFixed(1)}%</span>
                <span style="font:11px var(--font-mono);color:var(--secondary)">${(d.baseline.win_rate||0).toFixed(1)}%</span>
              </div>${bar(d.expert.win_rate||0, d.baseline.win_rate||0)}
            </div>
            <div style="margin-bottom:10px;">
              <div style="font:9px var(--font-mono);color:rgba(255,255,255,.4);text-transform:uppercase;margin-bottom:2px;">Sharpe Ratio</div>
              <div style="display:flex;justify-content:space-between;">
                <span style="font:600 11px var(--font-mono);color:var(--gold-soft)">${(d.expert.sharpe||0).toFixed(3)}</span>
                <span style="font:11px var(--font-mono);color:var(--secondary)">${(d.baseline.sharpe||0).toFixed(3)}</span>
              </div>${bar(Math.abs(d.expert.sharpe||0), Math.abs(d.baseline.sharpe||0))}
            </div>
            <div style="padding:8px 10px;background:rgba(16,185,129,.05);border:1px solid rgba(16,185,129,.12);border-radius:var(--radius-md);">
              <span style="font:9px var(--font-mono);color:rgba(255,255,255,.4);">Expert vs Buy &amp; Hold</span>
              <span style="float:right;font:700 11px var(--font-mono);color:${(d.expert.total_return-d.buy_hold_return)>=0?'var(--synced-green)':'var(--strong-sell)'}">${def(d.expert.total_return-d.buy_hold_return,'%')}</span>
            </div>`;
        }

        // Date labels
        const dateDiv = document.getElementById('bt-date-labels');
        if (dateDiv && d.dates && d.dates.length) {
            const n = d.dates.length;
            const lbls = [d.dates[0], d.dates[Math.floor(n / 2)], d.dates[n - 1]];
            dateDiv.innerHTML = lbls.map(dt => `<span style="font-family:var(--font-mono);font-size:8px;color:rgba(255,255,255,.25);">${dt ? dt.slice(0,7) : ''}</span>`).join('');
        }
    }

    function drawEquityCurve(d) {
        const container = document.getElementById('bt-chart-container');
        if (!container) return;

        const W = 700, H = 180, PAD = 14;
        const expert   = d.expert.equity_curve   || [];
        const baseline = d.baseline.equity_curve || [];
        const bah      = d.bah_curve              || [];
        const n        = Math.max(expert.length, baseline.length, bah.length);
        if (!n) return;

        const allVals = [...expert, ...baseline, ...bah].filter(v => v != null);
        const minV = Math.min(...allVals);
        const maxV = Math.max(...allVals);
        const rng  = maxV - minV || 1;

        const vy = v => PAD + (1 - (v - minV) / rng) * (H - PAD * 2);
        const vx = (i, len) => len > 1 ? (i / (len - 1)) * W : W / 2;

        const toPath = (arr, color, dashArr, width) => {
            if (!arr.length) return '';
            const pts = arr.map((v, i) => `${vx(i, arr.length).toFixed(1)},${vy(v).toFixed(1)}`).join(' ');
            return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round"${dashArr ? ` stroke-dasharray="${dashArr}"` : ''}/>`;
        };

        // Zero line
        const zeroY = vy(0).toFixed(1);
        const zeroLine = `<line x1="0" y1="${zeroY}" x2="${W}" y2="${zeroY}" stroke="rgba(255,255,255,.12)" stroke-width="0.8" stroke-dasharray="4,3"/>`;

        // Grid
        const grid = `<pattern id="btGrid" width="70" height="45" patternUnits="userSpaceOnUse">
            <line x1="70" y1="0" x2="70" y2="${H}" stroke="rgba(255,255,255,.03)" stroke-width="1"/>
            <line x1="0" y1="45" x2="${W}" y2="45" stroke="rgba(255,255,255,.03)" stroke-width="1"/>
        </pattern>`;

        // Split line
        const splitX = (d.split_index / n * W).toFixed(1);
        const splitLine = `<line x1="${splitX}" y1="0" x2="${splitX}" y2="${H}" stroke="rgba(255,255,255,.12)" stroke-width="0.8" stroke-dasharray="3,3"/>
            <text x="${parseFloat(splitX) + 4}" y="14" fill="rgba(255,255,255,.3)" style="font:7px 'JetBrains Mono'">OOS 30%</text>`;

        container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:100%" preserveAspectRatio="none">
  <defs>${grid}</defs>
  <rect width="${W}" height="${H}" fill="url(#btGrid)"/>
  ${zeroLine}
  ${splitLine}
  ${toPath(bah,      'rgba(255,255,255,.22)', '4,3', 1.2)}
  ${toPath(baseline, 'var(--secondary)',       '',    1.6)}
  ${toPath(expert,   'var(--gold-soft)',        '',    2.2)}
</svg>`;
    }

    function _btLoading(on) {
        if (!btRunBtn) return;
        btRunBtn.disabled = on;
        const lbl  = document.getElementById('bt-run-label');
        const icon = document.getElementById('bt-run-icon');
        if (lbl)  lbl.textContent = on ? 'Running…' : 'Execute Backtest';
        if (icon) icon.style.opacity = on ? '0.4' : '1';
    }

    function _btStatus(msg, type) {
        if (!btStatusEl) return;
        const colors = { ok: 'var(--synced-green)', error: 'var(--strong-sell)',
                         warn: 'var(--optimal-processing)', info: 'rgba(255,255,255,.4)' };
        btStatusEl.textContent = msg;
        btStatusEl.style.color = colors[type] || colors.info;
    }


    // SECTION 3 — PASSIVE UTILITIES (clock, market data poll, etc.)

    // Mobile sidebar toggle (<768px — sidebar is hidden by default and
    // slides in as an overlay when this button is pressed)
    const mobileNavToggle = document.getElementById('mobile-nav-toggle');
    const appSidebar      = document.getElementById('app-sidebar');
    const sidebarBackdrop = document.getElementById('sidebar-backdrop');
    function closeMobileNav() {
        if (appSidebar)      appSidebar.classList.remove('sidebar--open');
        if (sidebarBackdrop) sidebarBackdrop.classList.remove('sidebar-backdrop--open');
    }
    if (mobileNavToggle && appSidebar) {
        mobileNavToggle.addEventListener('click', () => {
            appSidebar.classList.toggle('sidebar--open');
            if (sidebarBackdrop) sidebarBackdrop.classList.toggle('sidebar-backdrop--open');
        });
        if (sidebarBackdrop) sidebarBackdrop.addEventListener('click', closeMobileNav);
        appSidebar.querySelectorAll('.nav-item').forEach(link => {
            link.addEventListener('click', closeMobileNav);
        });
    }

    // System clock
    const clockEl = document.querySelector('[data-clock]');
    if (clockEl) {
        const tick = () => {
            clockEl.textContent = 'LATEST REFRESH: ' +
                new Date().toUTCString().slice(17, 25) + ' UTC';
        };
        tick();
        setInterval(tick, 1000);
    }

    // Auto-refresh BTC price every 10s (only when no ticker has been analyzed)
    let userAnalyzed = false;
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => { userAnalyzed = true; }, { once: true });
    }
    const btcPriceEl = document.getElementById('live-price');
    if (btcPriceEl) {
        setInterval(() => {
            if (userAnalyzed) return;
            fetch('/api/market-data')
                .then(r => r.json())
                .then(data => {
                    if (btcPriceEl) btcPriceEl.textContent =
                        data.btc_price.toLocaleString('en-US', { minimumFractionDigits: 2 });
                })
                .catch(() => {});
        }, 10000);
    }

    // Button press feedback (all buttons)
    document.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('mousedown', () => { btn.style.transform = 'scale(0.97)'; });
        btn.addEventListener('mouseup',   () => { btn.style.transform = ''; });
        btn.addEventListener('mouseleave',() => { btn.style.transform = ''; });
    });

    // Ticker input: force uppercase as user types
    if (tickerInput) {
        tickerInput.addEventListener('input', () => {
            const pos = tickerInput.selectionStart;
            tickerInput.value = tickerInput.value.toUpperCase();
            tickerInput.setSelectionRange(pos, pos);
        });
    }

    // Range input live value display
    document.querySelectorAll('input[type="range"]').forEach(input => {
        input.addEventListener('input', function () {
            const label = this.parentElement.querySelector('[data-range-value]');
            if (label) label.textContent = (this.value / 100).toFixed(2);
        });
    });

    // Decision Logic tab bar
    const tabBar = document.querySelector('.tab-bar');
    if (tabBar) {
        tabBar.querySelectorAll('.tab-bar__item').forEach(btn => {
            btn.addEventListener('click', function () {
                tabBar.querySelectorAll('.tab-bar__item').forEach(b => b.classList.remove('tab-bar__item--active'));
                this.classList.add('tab-bar__item--active');
            });
        });
    }


    // SECTION 4 — HELPERS

    function _setEl(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function _fmt(n) {
        if (n == null) return '—';
        return Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function _pct(n) {
        if (n == null) return '—';
        return (n >= 0 ? '+' : '') + Number(n).toFixed(2) + '%';
    }

    function _signalColor(label) {
        if (!label) return 'var(--optimal-processing)';
        const l = label.toLowerCase();
        if (l.includes('sell')) return 'var(--strong-sell)';
        if (l.includes('buy'))  return 'var(--synced-green)';
        return 'var(--optimal-processing)';
    }


    // Toast notifications — every button should give at least one visible response

    function showToast(msg, type, duration) {
        type     = type     || 'info';
        duration = duration || 3200;
        let ctr = document.getElementById('toast-ctr');
        if (!ctr) {
            ctr = document.createElement('div');
            ctr.id = 'toast-ctr';
            ctr.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:99999;display:flex;flex-direction:column-reverse;gap:8px;pointer-events:none;';
            document.body.appendChild(ctr);
        }
        const t = document.createElement('div');
        const clr = {success:'rgba(16,185,129,.92)',warn:'rgba(245,158,11,.92)',error:'rgba(239,68,68,.92)',info:'rgba(124,208,255,.88)',demo:'rgba(248,187,111,.90)'};
        t.style.cssText = 'padding:10px 16px;background:'+( clr[type]||clr.info)+';color:#0d0d0f;border-radius:12px;font:600 11px "JetBrains Mono",monospace;opacity:0;transition:opacity 0.25s;max-width:340px;cursor:pointer;pointer-events:auto;box-shadow:0 8px 24px rgba(0,0,0,.4);';
        t.textContent = msg;
        ctr.appendChild(t);
        setTimeout(() => { t.style.opacity = '1'; }, 10);
        const rm = () => { t.style.opacity='0'; setTimeout(()=>{ try{ctr.removeChild(t);}catch(e){} },260); };
        t.addEventListener('click', rm);
        setTimeout(rm, duration);
    }


    // SECTION 6 — DASHBOARD PAGE INTERACTIONS

    // Timeframe buttons (1M / 5M / 15M)
    document.querySelectorAll('.tf-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            document.querySelectorAll('.tf-btn').forEach(b => {
                b.classList.remove('tf-btn--active');
                b.style.background  = '';
                b.style.borderColor = '';
                b.style.color       = '';
            });
            this.classList.add('tf-btn--active');
            this.style.background  = 'rgba(255,228,201,.12)';
            this.style.borderColor = 'rgba(255,228,201,.20)';
            this.style.color       = 'var(--gold-soft)';
            showToast('Chart timeframe: ' + this.dataset.tf.toUpperCase() + ' — refreshing…', 'info', 1800);
        });
    });

    // PAUSE SYSTEM
    const btnPause = document.getElementById('btn-pause');
    if (btnPause) {
        let paused = false;
        btnPause.addEventListener('click', () => {
            paused = !paused;
            btnPause.textContent = paused ? '▶ RESUME SYSTEM' : '⏸ PAUSE SYSTEM';
            btnPause.style.borderColor  = paused ? 'rgba(245,158,11,.3)' : '';
            btnPause.style.color        = paused ? 'var(--optimal-processing)' : '';
            showToast(paused ? 'System paused — pipeline halted' : 'System resumed — pipeline active', paused ? 'warn' : 'success');
        });
    }

    // EXECUTE OVERRIDE
    const btnOverride = document.getElementById('btn-override');
    if (btnOverride) {
        btnOverride.addEventListener('click', () => {
            btnOverride.disabled = true;
            showToast('Override submitted to Execution Agent — awaiting confirmation…', 'warn', 4000);
            setTimeout(() => {
                btnOverride.disabled = false;
                showToast('Override acknowledged. No real trade executed (demo mode).', 'demo', 4000);
            }, 2200);
        });
    }

    // Sidebar Execute Trade
    const btnExecTrade = document.getElementById('btn-execute-trade');
    if (btnExecTrade) {
        btnExecTrade.addEventListener('click', () => {
            btnExecTrade.disabled = true;
            showToast('Routing order to Execution Agent…', 'info', 1800);
            setTimeout(() => {
                btnExecTrade.disabled = false;
                showToast('Order submitted (demo mode — no real trade executed).', 'demo', 4000);
            }, 1600);
        });
    }


    // SECTION 7 — MARKET INTELLIGENCE PAGE INTERACTIONS

    const miFeedContainer = document.getElementById('mi-feed-container');
    if (miFeedContainer) {

        const miSearch = document.getElementById('mi-search');
        let mnArticles     = [];   // current article list from /api/market-news (real or demo)
        let mnActiveFilter = 'ALL';

        function _escapeHtml(s) {
            return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
        }

        function _sentimentTagClass(s) {
            if (s === 'BULLISH') return 'tag--bullish';
            if (s === 'BEARISH') return 'tag--bearish';
            return 'tag--neutral';
        }

        // Main feed (real Marketaux articles or demo fallback — single source: /api/market-news)
        function renderMainFeed() {
            const q = (miSearch ? miSearch.value.toLowerCase() : '');
            const filtered = mnArticles.filter(a => {
                const matchFilter = mnActiveFilter === 'ALL' || (a.sentiment || 'NEUTRAL') === mnActiveFilter;
                const matchQ = !q || (a.content || '').toLowerCase().includes(q);
                return matchFilter && matchQ;
            });

            if (!filtered.length) {
                miFeedContainer.innerHTML = '<div class="body-sm" style="padding:16px;">No headlines match this filter.</div>';
                return;
            }

            miFeedContainer.innerHTML = filtered.map((a, i) => {
                const safeUrl = (a.url || '').startsWith('http') ? a.url : '';
                const titleHtml = safeUrl
                    ? `<a href="${_escapeHtml(safeUrl)}" target="_blank" rel="noopener" style="color:inherit;text-decoration:none;">${_escapeHtml(a.content)}</a>`
                    : _escapeHtml(a.content);
                return `
                <div class="feed-item" data-idx="${i}">
                  <div class="feed-item__header">
                    <div class="feed-item__time">${_escapeHtml(a.time || '')}${a.source ? ' · ' + _escapeHtml(a.source) : ''}</div>
                    <button class="mi-dismiss-btn"
                      style="background:transparent;border:none;color:var(--text-subtle);cursor:pointer;font-size:14px;padding:2px 6px;border-radius:4px;"
                      title="Dismiss">&#x238B;</button>
                  </div>
                  <div class="feed-item__content">${titleHtml}</div>
                  <div class="feed-item__tags">
                    <span class="tag ${_sentimentTagClass(a.sentiment)}">${_escapeHtml(a.sentiment || 'NEUTRAL')}</span>
                    ${(a.tags || []).slice(0, 2).map(t => `<span class="tag tag--neutral">${_escapeHtml(t)}</span>`).join('')}
                  </div>
                </div>`;
            }).join('');

            // Re-attach dismiss handlers (container is rebuilt on every render)
            miFeedContainer.querySelectorAll('.mi-dismiss-btn').forEach(btn => {
                btn.addEventListener('click', function () {
                    const item = this.closest('.feed-item');
                    if (item) {
                        item.style.transition = 'opacity 0.3s';
                        item.style.opacity    = '0';
                        setTimeout(() => { item.style.display = 'none'; }, 300);
                    }
                });
            });
        }

        // Feed status badge/line (shared truth with AI Market Brief panel)
        function renderFeedStatus(d) {
            const dot   = document.getElementById('mi-feed-status-dot');
            const label = document.getElementById('mi-feed-status-label');
            const upd   = document.getElementById('mi-feed-updated');
            if (label) {
                label.textContent = d.is_live
                    ? `LIVE / CACHED FINANCE NEWS VIA ${(d.source || '').toUpperCase()}`
                    : 'DEMO SAMPLE FEED — add MARKETAUX_API_KEY to enable live finance news';
                label.style.color = d.is_live ? 'var(--semantic-buy)' : 'var(--text-muted)';
            }
            if (dot) dot.style.background = d.is_live ? '' : 'rgba(255,255,255,.25)';
            if (upd) upd.textContent = 'Updated ' + (d.last_updated || '').replace('T', ' ');
        }

        // AI Market Brief panel (right column) — summarizes the SAME articles as the main feed
        function renderAIBrief(d) {
            const badge   = document.getElementById('amb-source-badge');
            const updated = document.getElementById('amb-updated');
            const briefEl = document.getElementById('amb-brief-text');
            const modeEl  = document.getElementById('amb-brief-mode');
            const listEl  = document.getElementById('amb-headlines');
            if (!badge) return;

            badge.textContent = d.is_live ? `LIVE · ${(d.source || '').toUpperCase()}` : 'DEMO SAMPLE FEED';
            badge.className   = 'badge ' + (d.is_live ? 'badge--synced' : 'badge--idle');

            if (updated) updated.textContent = d.is_live
                ? `Live/cached finance headlines via ${d.source} — updated ${(d.last_updated || '').replace('T', ' ')}`
                : `Demo sample feed — updated ${(d.last_updated || '').replace('T', ' ')}`;

            const brief = d.ai_brief || {};
            if (briefEl) briefEl.textContent = brief.text || 'No brief available.';
            if (modeEl) modeEl.textContent = brief.mode === 'deepseek'
                ? 'DeepSeek-generated market interpretation'
                : 'Rule-based fallback summary';

            if (listEl) {
                const arts = d.articles || [];
                listEl.innerHTML = arts.slice(0, 3).map(a => `
                    <div class="inner-card" style="padding:8px 10px;">
                      <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                        <span class="label-xs" style="text-transform:none;color:var(--text-faint);">${_escapeHtml(a.source || '')}</span>
                        <span class="label-xs" style="text-transform:none;color:var(--text-faint);">${_escapeHtml(a.time || '')}</span>
                      </div>
                      <div class="label-sm" style="text-transform:none;">${_escapeHtml(a.content || '')}</div>
                    </div>`).join('') || '<div class="body-sm">No headlines available.</div>';
            }
        }

        // Single source of truth: /api/market-news feeds BOTH the main feed and the AI brief
        async function loadMarketNews() {
            try {
                const ticker = localStorage.getItem('astdss:lastTicker');
                const url = ticker ? `/api/market-news?symbol=${encodeURIComponent(ticker)}` : '/api/market-news';
                const resp = await fetch(url);
                const d    = await resp.json();
                mnArticles = d.articles || [];
                renderFeedStatus(d);
                renderMainFeed();
                renderAIBrief(d);
            } catch (e) {
                const label = document.getElementById('mi-feed-status-label');
                if (label) { label.textContent = 'NEWS SERVICE UNAVAILABLE'; label.style.color = 'var(--strong-sell)'; }
                const badge = document.getElementById('amb-source-badge');
                if (badge) { badge.textContent = 'UNAVAILABLE'; badge.className = 'badge badge--idle'; }
            }
        }
        loadMarketNews();
        // Re-poll periodically — cheap, since the server caches for NEWS_REFRESH_MINUTES.
        setInterval(loadMarketNews, 60000);

        if (miSearch) {
            miSearch.addEventListener('input', renderMainFeed);
        }

        // Sentiment filter buttons (All / Bullish / Bearish / Neutral)
        document.querySelectorAll('.mi-source-btn').forEach(btn => {
            btn.addEventListener('click', function () {
                mnActiveFilter = this.dataset.filter || 'ALL';
                document.querySelectorAll('.mi-source-btn').forEach(b => {
                    b.className = 'btn-secondary mi-source-btn';
                    b.style.padding = '';
                    b.style.fontSize = '';
                });
                this.className = 'btn-primary mi-source-btn';
                this.style.padding   = '8px 18px';
                this.style.fontSize  = '11px';
                renderMainFeed();
            });
        });

        // Filters button (advanced filter UI not built — honestly labeled demo)
        const miFiltBtn = document.getElementById('mi-filters-btn');
        if (miFiltBtn) {
            miFiltBtn.addEventListener('click', () => {
                showToast('Advanced filters: coming in next update (demo)', 'demo', 2500);
            });
        }

        // Sentiment gauge — simulated periodic randomization (independent of the
        // real/demo news feed above; already labeled "simulated demo value" in the template).
        // Does NOT touch #mi-feed-status-dot — that dot's color is reserved for the
        // real live/demo signal set by renderFeedStatus() above.
        setInterval(() => {
            const valEl = document.getElementById('mi-sentiment-value');
            const lblEl = document.getElementById('mi-sentiment-label');
            if (valEl) {
                const v = 40 + Math.floor(Math.random() * 40);
                valEl.textContent = v;
                if (lblEl) lblEl.textContent = v > 65 ? 'GREED PREVAILING' : v > 45 ? 'NEUTRAL MARKET' : 'FEAR PREVAILING';
            }
        }, 15000);
    }


    // SECTION 8 — DECISION LOGIC PAGE INTERACTIONS

    const decisionGrid = document.querySelector('.decision-grid');
    if (decisionGrid) {
        const tabItems = document.querySelectorAll('.tab-bar__item');

        // Tab content panels — create overlay panels for Rule Evaluation & Defuzzification
        function buildTabPanel(title, content) {
            const p = document.createElement('div');
            p.style.cssText = 'position:absolute;inset:0;background:rgba(13,11,10,.88);backdrop-filter:blur(8px);z-index:10;display:flex;align-items:center;justify-content:center;border-radius:var(--radius-lg);animation:fadeIn .25s ease;';
            p.innerHTML = '<div style="text-align:center;padding:24px;max-width:400px;">' +
                '<div style="font-family:var(--font-mono);font-size:10px;color:var(--gold-soft);text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;">' + title + '</div>' +
                '<div style="font-family:var(--font-mono);font-size:11px;color:rgba(255,255,255,.6);line-height:1.8;">' + content + '</div>' +
                '<button onclick="this.closest(\'[style*=\\\"position:absolute\\\"]\').remove();document.querySelectorAll(\'.tab-bar__item\').forEach(b=>b.classList.remove(\'tab-bar__item--active\'));document.querySelectorAll(\'.tab-bar__item\')[0].classList.add(\'tab-bar__item--active\')" style="margin-top:16px;font-family:var(--font-mono);font-size:10px;padding:8px 16px;background:rgba(248,187,111,.12);border:1px solid rgba(248,187,111,.25);border-radius:var(--radius-lg);color:var(--gold-soft);cursor:pointer;"> Back to Fuzzification</button>' +
                '</div>';
            p.dataset.tabPanel = '1';
            return p;
        }

        tabItems.forEach((btn, idx) => {
            btn.addEventListener('click', function () {
                tabItems.forEach(b => b.classList.remove('tab-bar__item--active'));
                this.classList.add('tab-bar__item--active');
                // Remove any existing overlay
                document.querySelectorAll('[data-tab-panel]').forEach(p => p.remove());

                if (idx === 1) {
                    // Rule Evaluation tab
                    const wrap = document.querySelector('.decision-grid');
                    if (wrap) {
                        wrap.style.position = 'relative';
                        wrap.appendChild(buildTabPanel(
                            'RULE EVALUATION',
                            'Mamdani inference rules are evaluated using MIN implication.<br><br>' +
                            'Rule 04: IF RSI is Overbought AND MACD is Bearish<br>THEN Signal = Strong Sell (strength: 0.68)<br><br>' +
                            'Rule 07: IF Sentiment is Bullish AND RSI is Oversold<br>THEN Signal = Strong Buy (idle)<br><br>' +
                            'Active rules fired: ' + (document.querySelector('[id*="rules_fired"]') ? '—' : 'see Expert Panel')
                        ));
                    }
                } else if (idx === 2) {
                    // Defuzzification tab
                    const wrap = document.querySelector('.decision-grid');
                    if (wrap) {
                        wrap.style.position = 'relative';
                        wrap.appendChild(buildTabPanel(
                            'DEFUZZIFICATION',
                            'Centroid method applied over output domain [-2, +2].<br><br>' +
                            'Output labels:<br>' +
                            'Strong Sell: [-2, -2, -1] &nbsp; Sell: [-1.5, -0.8, -0.2]<br>' +
                            'Hold: [-0.4, 0, 0.4] &nbsp; Buy: [0.2, 0.8, 1.5]<br>' +
                            'Strong Buy: [1, 2, 2]<br><br>' +
                            'Final crisp output = &sum;(area &times; centroid) / &sum;area'
                        ));
                    }
                }
            });
        });

        // Decision Logic Execute Trade button
        const dlExec = document.getElementById('dl-execute-trade');
        if (dlExec) {
            dlExec.addEventListener('click', () => {
                dlExec.disabled = true;
                showToast('Routing signal to Execution Agent…', 'info', 1600);
                setTimeout(() => {
                    dlExec.disabled = false;
                    const sig = document.getElementById('agg-signal')?.textContent || 'current signal';
                    showToast('Trade executed: ' + sig + ' (demo mode — no real order sent).', 'demo', 4000);
                }, 1500);
            });
        }
    }


    // SECTION 9 — BACKTESTING PAGE INTERACTIONS

    // Asset class option buttons — sets the actual backtest ticker (data-ticker),
    // so picking "Crypto" really backtests BTC, not just a cosmetic radio toggle.
    document.querySelectorAll('.asset-option[data-ticker]').forEach(opt => {
        opt.addEventListener('click', function () {
            const wasActive = this.classList.contains('asset-option--active');
            if (!wasActive) {
                document.querySelectorAll('.asset-option[data-ticker]').forEach(o => {
                    o.classList.remove('asset-option--active');
                    const dot  = o.querySelector('.radio-dot');
                    const icon = o.querySelector('.radio-dot svg');
                    if (dot)  dot.classList.remove('radio-dot--checked');
                    if (icon) icon.remove();
                });
                this.classList.add('asset-option--active');
                const dot = this.querySelector('.radio-dot');
                if (dot) {
                    dot.classList.add('radio-dot--checked');
                    dot.innerHTML = '<svg viewBox="0 0 12 12" style="width:8px;height:8px;" fill="none"><path d="M2 6l3 3 5-5" stroke="#472a00" stroke-width="1.5" stroke-linecap="round"/></svg>';
                }
                const ticker = this.dataset.ticker;
                const tickerEl = document.getElementById('bt-ticker');
                if (tickerEl && ticker) tickerEl.value = ticker;
                const name = this.querySelector('.asset-option__name')?.textContent || 'Asset';
                showToast('Backtest ticker set to ' + ticker + ' (' + name + ')', 'info', 1800);
            }
        });
    });

    // Density slider — show value label
    const densitySlider = document.querySelector('.config-main input[type="range"]');
    if (densitySlider) {
        const densityDisplay = document.createElement('div');
        densityDisplay.style.cssText = 'font-family:var(--font-mono);font-size:9px;color:var(--gold-soft);text-align:right;margin-top:-4px;margin-bottom:8px;';
        densityDisplay.textContent = 'Density: ' + densitySlider.value + '%';
        densitySlider.parentNode.insertBefore(densityDisplay, densitySlider.nextSibling);
        densitySlider.addEventListener('input', () => {
            densityDisplay.textContent = 'Density: ' + densitySlider.value + '%';
        });
    }


    // SECTION 10 — SETTINGS PAGE INTERACTIONS

    // API key eye buttons — toggle password visibility
    document.querySelectorAll('.api-eye-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const input = this.parentElement.querySelector('input[type="password"], input[type="text"]');
            if (!input) return;
            const isHidden = input.type === 'password';
            input.type     = isHidden ? 'text' : 'password';
            this.textContent = isHidden ? '🙈' : '👁';
            this.title       = isHidden ? 'Hide API key' : 'Show API key';
        });
    });

    // Run Global Connection Test button
    const btnConnTest = document.getElementById('btn-connection-test');
    if (btnConnTest) {
        btnConnTest.addEventListener('click', () => {
            btnConnTest.disabled     = true;
            btnConnTest.textContent  = 'TESTING CONNECTIONS…';
            showToast('Running API connectivity test…', 'info', 2000);
            setTimeout(() => {
                btnConnTest.disabled    = false;
                btnConnTest.textContent = 'RUN GLOBAL CONNECTION TEST';
                showToast('Bloomberg: ACTIVE  |  Reuters: TIMEOUT  |  Binance WS: ACTIVE', 'warn', 5000);
            }, 2400);
        });
    }

    // Agent toggle switches — find all 38×20 toggle tracks across the settings page
    // Use size-based detection (width:38px height:20px) rather than fragile style string matching
    function isToggleTrack(el) {
        const s = el.style;
        return s && s.width === '38px' && s.height === '20px' &&
               s.borderRadius && s.position === 'relative';
    }

    document.querySelectorAll('.settings-grid div').forEach(el => {
        if (!isToggleTrack(el)) return;
        el.style.cursor = 'pointer';

        // Read initial state from thumb position
        const thumb    = el.firstElementChild;
        let isOn       = thumb && (thumb.style.left === '21px');
        const inAgentGrid = !!el.closest('.settings-agent-grid');
        const label    = inAgentGrid ? 'Agent' : 'Toggle';

        el.addEventListener('click', function () {
            isOn = !isOn;
            el.style.background = isOn ? 'var(--primary)' : 'rgba(255,255,255,.12)';
            el.style.transition  = 'background 0.3s';
            if (thumb) {
                thumb.style.left       = isOn ? '21px' : '3px';
                thumb.style.background = isOn ? 'var(--on-primary)' : 'rgba(255,255,255,.5)';
                thumb.style.transition = 'left 0.3s, background 0.3s';
            }
            showToast(label + (isOn ? ' enabled' : ' disabled') + ' — demo mode, no server change', 'demo', 2800);
        });
    });

    // Appearance theme buttons — GLASS / DARK / LIGHT
    document.querySelectorAll('.settings-theme-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            document.querySelectorAll('.settings-theme-btn').forEach(b => {
                b.classList.remove('settings-theme-btn--active');
                b.style.borderColor = '';
                b.style.color = '';
            });
            this.classList.add('settings-theme-btn--active');
            this.style.borderColor = 'var(--gold-soft)';
            this.style.color       = 'var(--gold-soft)';
            showToast('Theme: ' + this.textContent.trim() + ' (demo — applied locally)', 'demo', 2500);
        });
    });

    // Layout density buttons — COMPACT / DEFAULT / RELAXED
    document.querySelectorAll('.settings-density-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            document.querySelectorAll('.settings-density-btn').forEach(b => {
                b.classList.remove('settings-density-btn--active');
                b.style.background  = '';
                b.style.borderColor = '';
                b.style.color       = '';
            });
            this.classList.add('settings-density-btn--active');
            this.style.background  = 'rgba(248,187,111,.1)';
            this.style.borderColor = 'rgba(248,187,111,.2)';
            this.style.color       = 'var(--gold-soft)';
            this.style.fontWeight  = '600';
            showToast('Layout: ' + this.textContent.trim() + ' density selected (demo)', 'demo', 2500);
        });
    });

    // Fuzzy calibration sliders — show live value
    document.querySelectorAll('.glass-panel input[type="range"]').forEach(slider => {
        if (slider.closest('.config-main')) return; // skip backtesting density slider
        const parent = slider.parentElement;
        // Find the value display span (already in settings template via {{ val }})
        slider.addEventListener('input', function () {
            const valSpan = this.parentElement.querySelector('.label-mono[style*="color: var(--primary)"], .label-mono[style*="color:var(--primary)"]');
            if (valSpan) {
                valSpan.textContent = (this.value / 100).toFixed(2);
            }
        });
    });

    // Settings: save fields — add feedback on input change
    let saveTimeout;
    document.querySelectorAll('.settings-grid .glass-input').forEach(inp => {
        if (inp.id === 'mi-search' || inp.id === 'bt-ticker' || inp.id === 'ticker-input') return;
        inp.addEventListener('change', () => {
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                showToast('Setting updated (saved locally — demo mode)', 'demo', 2200);
            }, 800);
        });
    });

    // Load Real Agentic Experience Log (Backtesting page panel)
    function loadAgenticLog() {
        const tbody = document.getElementById('bt-agentic-tbody');
        const count = document.getElementById('bt-agentic-count');
        if (!tbody) return;
        fetch('/api/agentic/experience-log?limit=5')
            .then(r => r.json())
            .then(entries => {
                if (!Array.isArray(entries) || !entries.length) {
                    if (count) count.textContent = 'No runs yet';
                    return;
                }
                const statusColor = {
                    APPROVED:     'var(--synced-green)',
                    REJECTED:     'var(--strong-sell)',
                    MODIFIED:     'var(--gold-soft)',
                    NEEDS_REVIEW: 'var(--secondary)',
                };
                tbody.innerHTML = entries.slice(0, 5).map(e => {
                    const ts       = (e.timestamp || '').replace('T', ' ').slice(0, 16);
                    const fd       = e.final_decision || {};
                    const label    = fd.label || '—';
                    const score    = fd.score != null ? Number(fd.score).toFixed(3) : '—';
                    const vr       = e.expert_validation_result || {};
                    const status   = vr.status || '—';
                    const stab     = e.stability_score != null ? (e.stability_score * 100).toFixed(0) + '%' : '—';
                    const col      = statusColor[status] || 'rgba(255,255,255,.4)';
                    return `<tr>
                        <td>${ts}</td>
                        <td style="color:var(--gold-soft);">${label} (${score})</td>
                        <td style="color:${col};">${status}</td>
                        <td>${stab}</td>
                    </tr>`;
                }).join('');
                if (count) count.textContent = `Last ${Math.min(entries.length, 5)} of ${entries.length} runs`;
            })
            .catch(() => { if (count) count.textContent = 'Unavailable'; });
    }
    loadAgenticLog();

});

