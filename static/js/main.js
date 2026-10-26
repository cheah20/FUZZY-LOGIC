// Page behaviour for every NeuralTrade page.

document.addEventListener('DOMContentLoaded', () => {

    // Dashboard: analyse a ticker

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
            setAnalyzeStatus(ticker + ' · ' + data.source, 'ok');
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
                         warn: 'var(--optimal-processing)', info: 'rgba(20, 26, 34, 0.45)' };
        analyzeStatus.textContent = msg;
        analyzeStatus.style.color = colors[type] || colors.info;
    }

    // price card and signal
    function updateDashboardMetrics(d) {
        _setEl('live-ticker-label', d.ticker);
        _setEl('live-price',  _fmt(d.price));
        _setEl('live-rsi',    d.rsi != null  ? d.rsi.toFixed(1)  : '—');
        _setEl('live-atr',    d.atr  != null ? 'ATR ' + d.atr.toFixed(2) : '—');
        _setEl('live-source', d.source ? d.source.replace(/yfinance/, 'Live') : 'Live');
        const pin = document.getElementById('decision-pin');
        if (pin && typeof d.expert_score === 'number') {
            const pos = Math.max(0, Math.min(100, (d.expert_score + 2) / 4 * 100));
            pin.style.left = pos.toFixed(1) + '%';
        }

        const sigEl  = document.getElementById('live-signal');
        const regEl  = document.getElementById('live-regime');
        if (sigEl) {
            sigEl.textContent = d.expert_label || '—';
            sigEl.style.color = _signalColor(d.expert_label);
        }
        if (regEl) regEl.textContent = d.regime_display || '';

        _setEl('agg-signal', (d.expert_label || '—').toUpperCase());
        const aggScore = document.getElementById('agg-score');
        if (aggScore) {
            aggScore.textContent = 'score ' + (d.expert_score >= 0 ? '+' : '') + d.expert_score.toFixed(3);
        }
    }

    // decision panel and bottom tabs
    function updateExpertPanel(d) {
        // regime badge by id; a class selector would also hit the pipeline badge
        const regimeBadge = document.getElementById('dash-regime-badge');
        if (regimeBadge && d.regime_display) regimeBadge.textContent = d.regime_display;

        // indicator rows, same data the server used for the first render
        const rows = document.getElementById('dash-inputs');
        if (rows && d.input_display) {
            rows.innerHTML = d.input_display.map(r => `
    <div class="input-row${r.active ? '' : ' is-masked'}">
      <div class="input-row__top">
        <span class="input-row__name">${r.label}${r.active ? '' : ' · Masked'}</span>
        <span class="input-row__val" style="color:${r.color}">${r.value}</span>
      </div>
      <div class="progress-bar"><div class="progress-bar__fill" style="width:${r.pct}%;background:${r.color};opacity:${r.active ? 1 : 0.35};"></div></div>
    </div>`).join('');
        }

        // bottom "Expert reasoning" tab
        const tags = (list, cls) => list && list.length
            ? list.map(f => `<span class="badge ${cls}" style="margin:0 4px 4px 0;">${f}</span>`).join('')
            : '<span style="color:var(--text-faint);">none</span>';
        _setEl('dash-rule', d.triggered_rule || '');
        if (typeof d.regime_confidence === 'number') _setEl('dash-conf', Math.round(d.regime_confidence * 100));
        const act = document.getElementById('dash-active');
        if (act) act.innerHTML = tags(d.active_features, 'badge--synced');
        const msk = document.getElementById('dash-masked');
        if (msk) msk.innerHTML = tags(d.masked_features, 'badge--idle');
        const flc = document.getElementById('dash-flc');
        if (flc && d.expert_label) {
            flc.innerHTML = `<b style="color:${_signalColor(d.expert_label)};">${d.expert_label}</b> ` +
                `<span style="color:var(--text-muted);">(${d.expert_score >= 0 ? '+' : ''}${d.expert_score.toFixed(2)})</span>`;
        }
    }

    // Price chart: candles when open/high/low are there, a line otherwise
    let lastChart = null;
    window.addEventListener('resize', () => { if (lastChart) drawPriceChart(lastChart); });

    function drawPriceChart(cd) {
        const container = document.getElementById('live-chart-container');
        if (!container || !cd.close || !cd.close.length) return;
        lastChart = cd;

        const box = container.getBoundingClientRect();
        const W = Math.round(box.width) || 1000, H = Math.round(box.height) || 400, AXIS = 72, VOL_H = 60, TOP = 18;
        const closes = cd.close;
        const n = closes.length;
        const hasOhlc = cd.open && cd.high && cd.low && cd.open.length === n;
        const highs = hasOhlc ? cd.high : closes;
        const lows = hasOhlc ? cd.low : closes;
        const volumes = cd.volume || [];
        const up = '#12a46b', down = '#e5443b', grid = '#eef0f3', label = '#a5adb8';

        const maxP = Math.max(...highs), minP = Math.min(...lows);
        const rng = maxP - minP || 1;
        const plotW = W - AXIS, plotH = H - VOL_H - TOP - 20;
        const step = plotW / n;
        const x = i => i * step + step / 2;
        const y = p => TOP + (maxP - p) / rng * plotH;

        let out = '';
        for (let g = 0; g <= 5; g++) {
            const gy = TOP + g * plotH / 5;
            const val = maxP - g * rng / 5;
            out += `<line x1="0" x2="${plotW}" y1="${gy.toFixed(1)}" y2="${gy.toFixed(1)}" stroke="${grid}"/>`;
            out += `<text x="${plotW + 8}" y="${(gy + 4).toFixed(1)}" fill="${label}" style="font:11px Roboto">${_fmt(val)}</text>`;
        }

        closes.forEach((c, i) => {
            const prev = i > 0 ? closes[i - 1] : c;
            const o = hasOhlc ? cd.open[i] : prev;
            const col = c >= o ? up : down;
            if (hasOhlc) {
                out += `<line x1="${x(i).toFixed(1)}" x2="${x(i).toFixed(1)}" y1="${y(cd.high[i]).toFixed(1)}" y2="${y(cd.low[i]).toFixed(1)}" stroke="${col}"/>`;
                const top = y(Math.max(o, c)), h = Math.max(1, Math.abs(y(o) - y(c)));
                out += `<rect x="${(x(i) - step * 0.32).toFixed(1)}" y="${top.toFixed(1)}" width="${(step * 0.64).toFixed(1)}" height="${h.toFixed(1)}" fill="${col}"/>`;
            }
            if (volumes[i] != null) {
                const vh = volumes[i] / 100 * VOL_H;
                out += `<rect x="${(x(i) - step * 0.32).toFixed(1)}" y="${(H - 20 - vh).toFixed(1)}" width="${(step * 0.64).toFixed(1)}" height="${vh.toFixed(1)}" fill="${col}" opacity="0.3"/>`;
            }
        });

        if (!hasOhlc) {
            const pts = closes.map((c, i) => x(i).toFixed(1) + ',' + y(c).toFixed(1)).join(' ');
            out += `<polyline points="${pts}" fill="none" stroke="${closes[n - 1] >= closes[0] ? up : down}" stroke-width="1.8"/>`;
        }

        // 20-day moving average
        let ma = '';
        for (let i = 19; i < n; i++) {
            const avg = closes.slice(i - 19, i + 1).reduce((a, b) => a + b, 0) / 20;
            ma += x(i).toFixed(1) + ',' + y(avg).toFixed(1) + ' ';
        }
        if (ma) out += `<polyline points="${ma}" fill="none" stroke="#f0a020" stroke-width="1.6"/>`;

        // last price tag on the axis
        const last = closes[n - 1], ly = y(last);
        const lastCol = last >= (hasOhlc ? cd.open[n - 1] : closes[n - 2] || last) ? up : down;
        out += `<line x1="0" x2="${plotW}" y1="${ly.toFixed(1)}" y2="${ly.toFixed(1)}" stroke="${lastCol}" stroke-dasharray="3 3" opacity="0.6"/>`;
        out += `<rect x="${plotW + 2}" y="${(ly - 10).toFixed(1)}" width="${AXIS - 4}" height="20" rx="2" fill="${lastCol}"/>`;
        out += `<text x="${plotW + 8}" y="${(ly + 4).toFixed(1)}" fill="#fff" style="font:500 11px Roboto">${_fmt(last)}</text>`;

        if (cd.dates && cd.dates.length) {
            [0, Math.floor(n / 3), Math.floor(2 * n / 3), n - 1].forEach(i => {
                if (cd.dates[i]) out += `<text x="${x(i).toFixed(1)}" y="${H - 4}" text-anchor="middle" fill="${label}" style="font:11px Roboto">${cd.dates[i].slice(5)}</text>`;
            });
        }

        container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:100%">${out}</svg>`;
    }


    // Backtesting

    const btRunBtn     = document.getElementById('bt-run-btn');
    const btTickerEl   = document.getElementById('bt-ticker');
    const btStatusEl   = document.getElementById('bt-status');
    const btLivePanel  = document.getElementById('bt-live-results');
    const btStaticPanel= document.getElementById('bt-static-stats');

    // 1Y / 2Y / 3Y buttons
    let activePeriod = '2y';

    // date range from today
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
            btn.style.background    = 'rgba(20, 26, 34, 0.108)';
            btn.style.borderColor   = 'rgba(20, 26, 34, 0.18)';
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
            _btStatus('Done: ' + data.total_bars + ' bars, ' + activePeriod.toUpperCase(), 'ok');

        } catch (err) {
            _btStatus('Network error. Is the server running?', 'error');
        } finally {
            _btLoading(false);
        }
    }

    function displayBacktestResults(d) {
        // show the real results, hide the example numbers
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

        // colour of the expert return
        const expRetEl = document.getElementById('bt-expert-return');
        if (expRetEl) expRetEl.style.color = d.expert.total_return >= 0 ? 'var(--synced-green)' : 'var(--strong-sell)';

        // equity chart
        drawEquityCurve(d);

        // trade log
        const logTbody   = document.getElementById('bt-log-tbody');
        const logCount   = document.getElementById('bt-log-count');
        const closeTrades = (d.expert.trades || []).filter(t => t.type === 'close');
        if (logTbody && closeTrades.length) {
            // newest trades first
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
        // reload the agentic log once the backtest is done
        loadAgenticLog();

        // baseline vs expert panel
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
            <div style="font-family:var(--font-mono);font-size:8px;color:rgba(20, 26, 34, 0.45);line-height:1.5;margin-bottom:12px;">
              The expert layer mainly changes decisions in extreme markets. It does not beat the baseline on every ticker or period.
            </div>
            <div style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                <span style="font:9px var(--font-mono);color:rgba(20, 26, 34, 0.45);text-transform:uppercase;">Total Return</span>
                <span style="font:8px var(--font-mono);color:rgba(20, 26, 34, 0.45);">B&H ${def(d.buy_hold_return, '%')}</span>
              </div>
              <div style="display:flex;justify-content:space-between;">
                <span style="font:600 11px var(--font-mono);color:${d.expert.total_return>=0?'var(--synced-green)':'var(--strong-sell)'}">${def(d.expert.total_return,'%')}</span>
                <span style="font:11px var(--font-mono);color:var(--secondary)">${def(d.baseline.total_return,'%')}</span>
              </div>${bar(d.expert.total_return, d.baseline.total_return)}
            </div>
            <div style="margin-bottom:10px;">
              <div style="font:9px var(--font-mono);color:rgba(20, 26, 34, 0.45);text-transform:uppercase;margin-bottom:2px;">Win Rate</div>
              <div style="display:flex;justify-content:space-between;">
                <span style="font:600 11px var(--font-mono);color:var(--gold-soft)">${(d.expert.win_rate||0).toFixed(1)}%</span>
                <span style="font:11px var(--font-mono);color:var(--secondary)">${(d.baseline.win_rate||0).toFixed(1)}%</span>
              </div>${bar(d.expert.win_rate||0, d.baseline.win_rate||0)}
            </div>
            <div style="margin-bottom:10px;">
              <div style="font:9px var(--font-mono);color:rgba(20, 26, 34, 0.45);text-transform:uppercase;margin-bottom:2px;">Sharpe Ratio</div>
              <div style="display:flex;justify-content:space-between;">
                <span style="font:600 11px var(--font-mono);color:var(--gold-soft)">${(d.expert.sharpe||0).toFixed(3)}</span>
                <span style="font:11px var(--font-mono);color:var(--secondary)">${(d.baseline.sharpe||0).toFixed(3)}</span>
              </div>${bar(Math.abs(d.expert.sharpe||0), Math.abs(d.baseline.sharpe||0))}
            </div>
            <div style="padding:8px 10px;background:rgba(18, 164, 107, 0.05);border:1px solid rgba(18, 164, 107, 0.12);border-radius:var(--radius-md);">
              <span style="font:9px var(--font-mono);color:rgba(20, 26, 34, 0.45);">Expert vs Buy &amp; Hold</span>
              <span style="float:right;font:700 11px var(--font-mono);color:${(d.expert.total_return-d.buy_hold_return)>=0?'var(--synced-green)':'var(--strong-sell)'}">${def(d.expert.total_return-d.buy_hold_return,'%')}</span>
            </div>`;
        }

        // dates
        const dateDiv = document.getElementById('bt-date-labels');
        if (dateDiv && d.dates && d.dates.length) {
            const n = d.dates.length;
            const lbls = [d.dates[0], d.dates[Math.floor(n / 2)], d.dates[n - 1]];
            dateDiv.innerHTML = lbls.map(dt => `<span style="font-family:var(--font-mono);font-size:8px;color:rgba(20, 26, 34, 0.45);">${dt ? dt.slice(0,7) : ''}</span>`).join('');
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

        // zero line
        const zeroY = vy(0).toFixed(1);
        const zeroLine = `<line x1="0" y1="${zeroY}" x2="${W}" y2="${zeroY}" stroke="rgba(20, 26, 34, 0.12)" stroke-width="0.8" stroke-dasharray="4,3"/>`;

        // grid
        const grid = `<pattern id="btGrid" width="70" height="45" patternUnits="userSpaceOnUse">
            <line x1="70" y1="0" x2="70" y2="${H}" stroke="rgba(20, 26, 34, 0.03)" stroke-width="1"/>
            <line x1="0" y1="45" x2="${W}" y2="45" stroke="rgba(20, 26, 34, 0.03)" stroke-width="1"/>
        </pattern>`;

        // train / test split
        const splitX = (d.split_index / n * W).toFixed(1);
        const splitLine = `<line x1="${splitX}" y1="0" x2="${splitX}" y2="${H}" stroke="rgba(20, 26, 34, 0.12)" stroke-width="0.8" stroke-dasharray="3,3"/>
            <text x="${parseFloat(splitX) + 4}" y="14" fill="rgba(20, 26, 34, 0.45)" style="font:7px Roboto">OOS 30%</text>`;

        container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:100%" preserveAspectRatio="none">
  <defs>${grid}</defs>
  <rect width="${W}" height="${H}" fill="url(#btGrid)"/>
  ${zeroLine}
  ${splitLine}
  ${toPath(bah,      'rgba(20, 26, 34, 0.45)', '4,3', 1.2)}
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
                         warn: 'var(--optimal-processing)', info: 'rgba(20, 26, 34, 0.45)' };
        btStatusEl.textContent = msg;
        btStatusEl.style.color = colors[type] || colors.info;
    }


    // Shared bits: menu, clock, price refresh

    // menu button on phones (below 768px the menu slides in over the page)
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

    // clock
    const clockEl = document.querySelector('[data-clock]');
    if (clockEl) {
        const tick = () => {
            clockEl.textContent = new Date().toUTCString().slice(17, 25) + ' UTC';
        };
        tick();
        setInterval(tick, 1000);
    }

    // refresh the BTC price every 10 s until the user analyses a ticker
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

    // small press effect on buttons
    document.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('mousedown', () => { btn.style.transform = 'scale(0.97)'; });
        btn.addEventListener('mouseup',   () => { btn.style.transform = ''; });
        btn.addEventListener('mouseleave',() => { btn.style.transform = ''; });
    });

    // ticker in capitals
    if (tickerInput) {
        tickerInput.addEventListener('input', () => {
            const pos = tickerInput.selectionStart;
            tickerInput.value = tickerInput.value.toUpperCase();
            tickerInput.setSelectionRange(pos, pos);
        });
    }

    // show slider values
    document.querySelectorAll('input[type="range"]').forEach(input => {
        input.addEventListener('input', function () {
            const label = this.parentElement.querySelector('[data-range-value]');
            if (label) label.textContent = (this.value / 100).toFixed(2);
        });
    });

    // Decision Logic tabs
    const tabBar = document.querySelector('.tab-bar');
    if (tabBar) {
        tabBar.querySelectorAll('.tab-bar__item').forEach(btn => {
            btn.addEventListener('click', function () {
                tabBar.querySelectorAll('.tab-bar__item').forEach(b => b.classList.remove('tab-bar__item--active'));
                this.classList.add('tab-bar__item--active');
            });
        });
    }


    // helpers

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


    // small pop-up messages

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
        const clr = {success:'rgba(18, 164, 107, 0.92)',warn:'rgba(217, 138, 0, 0.92)',error:'rgba(229, 68, 59, 0.92)',info:'rgba(18, 164, 107, 0.88)',demo:'rgba(20, 26, 34, 0.81)'};
        t.style.cssText = 'padding:10px 16px;background:'+( clr[type]||clr.info)+';color:#ffffff;border-radius:12px;font:600 11px "Roboto Mono",monospace;opacity:0;transition:opacity 0.25s;max-width:340px;cursor:pointer;pointer-events:auto;box-shadow:0 8px 24px rgba(16, 24, 40, 0.14);';
        t.textContent = msg;
        ctr.appendChild(t);
        setTimeout(() => { t.style.opacity = '1'; }, 10);
        const rm = () => { t.style.opacity='0'; setTimeout(()=>{ try{ctr.removeChild(t);}catch(e){} },260); };
        t.addEventListener('click', rm);
        setTimeout(rm, duration);
    }


    // Dashboard buttons

    // timeframe buttons
    document.querySelectorAll('.tf-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            document.querySelectorAll('.tf-btn').forEach(b => {
                b.classList.remove('tf-btn--active');
                b.style.background  = '';
                b.style.borderColor = '';
                b.style.color       = '';
            });
            this.classList.add('tf-btn--active');
            this.style.background  = 'rgba(20, 26, 34, 0.108)';
            this.style.borderColor = 'rgba(20, 26, 34, 0.18)';
            this.style.color       = 'var(--gold-soft)';
            showToast('Chart timeframe: ' + this.dataset.tf.toUpperCase(), 'info', 1800);
        });
    });

    // pause
    const btnPause = document.getElementById('btn-pause');
    if (btnPause) {
        let paused = false;
        btnPause.addEventListener('click', () => {
            paused = !paused;
            btnPause.textContent = paused ? 'Resume system' : 'Pause system';
            btnPause.style.borderColor  = paused ? 'rgba(217, 138, 0, 0.3)' : '';
            btnPause.style.color        = paused ? 'var(--optimal-processing)' : '';
            showToast(paused ? 'System paused' : 'System running again', paused ? 'warn' : 'success');
        });
    }

    // override
    const btnOverride = document.getElementById('btn-override');
    if (btnOverride) {
        btnOverride.addEventListener('click', () => {
            btnOverride.disabled = true;
            showToast('Override sent to the execution agent…', 'warn', 4000);
            setTimeout(() => {
                btnOverride.disabled = false;
                showToast('Override noted. Demo only, no real trade.', 'demo', 4000);
            }, 2200);
        });
    }

    // Execute Trade in the menu
    const btnExecTrade = document.getElementById('btn-execute-trade');
    if (btnExecTrade) {
        btnExecTrade.addEventListener('click', () => {
            btnExecTrade.disabled = true;
            showToast('Routing order to Execution Agent…', 'info', 1800);
            setTimeout(() => {
                btnExecTrade.disabled = false;
                showToast('Order sent. Demo only, no real trade.', 'demo', 4000);
            }, 1600);
        });
    }


    // Market Intelligence

    const miFeedContainer = document.getElementById('mi-feed-container');
    if (miFeedContainer) {

        const miSearch = document.getElementById('mi-search');
        let mnArticles     = [];   // articles from the last /api/market-news call
        let mnActiveFilter = 'ALL';

        function _escapeHtml(s) {
            return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
        }

        function _sentimentTagClass(s) {
            if (s === 'BULLISH') return 'tag--bullish';
            if (s === 'BEARISH') return 'tag--bearish';
            return 'tag--neutral';
        }

        // news feed, from /api/market-news (Marketaux or demo)
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

            // the list is rebuilt each time, so hook up the dismiss buttons again
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

        // live / demo status line
        function renderFeedStatus(d) {
            const dot   = document.getElementById('mi-feed-status-dot');
            const label = document.getElementById('mi-feed-status-label');
            const upd   = document.getElementById('mi-feed-updated');
            if (label) {
                label.textContent = d.is_live
                    ? `LIVE / CACHED FINANCE NEWS VIA ${(d.source || '').toUpperCase()}`
                    : 'Demo feed. Add MARKETAUX_API_KEY to .env for live news.';
                label.style.color = d.is_live ? 'var(--semantic-buy)' : 'var(--text-muted)';
            }
            if (dot) dot.style.background = d.is_live ? '' : 'rgba(20, 26, 34, 0.45)';
            if (upd) upd.textContent = 'Updated ' + (d.last_updated || '').replace('T', ' ');
        }

        // market brief, built from the same headlines as the feed
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
                ? `Headlines from ${d.source}, updated ${(d.last_updated || '').replace('T', ' ')}`
                : `Demo feed, updated ${(d.last_updated || '').replace('T', ' ')}`;

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

        // one request fills both the feed and the brief
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
        // poll again now and then; the server caches for NEWS_REFRESH_MINUTES
        setInterval(loadMarketNews, 60000);

        if (miSearch) {
            miSearch.addEventListener('input', renderMainFeed);
        }

        // All / Bullish / Bearish / Neutral
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

        // the advanced filters are not built
        const miFiltBtn = document.getElementById('mi-filters-btn');
        if (miFiltBtn) {
            miFiltBtn.addEventListener('click', () => {
                showToast('Advanced filters are not built yet.', 'demo', 2500);
            });
        }

        // sentiment gauge: a demo value that changes every few seconds.
        // It leaves #mi-feed-status-dot alone, that dot shows live or demo news.
        setInterval(() => {
            const valEl = document.getElementById('mi-sentiment-value');
            const lblEl = document.getElementById('mi-sentiment-label');
            if (valEl) {
                const v = 40 + Math.floor(Math.random() * 40);
                valEl.textContent = v;
                if (lblEl) lblEl.textContent = v > 65 ? 'Mostly greed' : v > 45 ? 'Neutral' : 'Mostly fear';
            }
        }, 15000);
    }


    // Decision Logic

    const decisionGrid = document.querySelector('.decision-grid');
    if (decisionGrid) {
        const tabItems = document.querySelectorAll('.tab-bar__item');

        // overlay panels for the Rule Evaluation and Defuzzification tabs
        function buildTabPanel(title, content) {
            const p = document.createElement('div');
            p.style.cssText = 'position:absolute;inset:0;background:rgba(255, 255, 255, 0.94);backdrop-filter:blur(8px);z-index:10;display:flex;align-items:center;justify-content:center;border-radius:var(--radius-lg);animation:fadeIn .25s ease;';
            p.innerHTML = '<div style="text-align:center;padding:24px;max-width:400px;">' +
                '<div style="font-family:var(--font-mono);font-size:10px;color:var(--gold-soft);text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;">' + title + '</div>' +
                '<div style="font-family:var(--font-mono);font-size:11px;color:rgba(20, 26, 34, 0.6);line-height:1.8;">' + content + '</div>' +
                '<button onclick="this.closest(\'[style*=\\\"position:absolute\\\"]\').remove();document.querySelectorAll(\'.tab-bar__item\').forEach(b=>b.classList.remove(\'tab-bar__item--active\'));document.querySelectorAll(\'.tab-bar__item\')[0].classList.add(\'tab-bar__item--active\')" style="margin-top:16px;font-family:var(--font-mono);font-size:10px;padding:8px 16px;background:rgba(20, 26, 34, 0.108);border:1px solid rgba(20, 26, 34, 0.225);border-radius:var(--radius-lg);color:var(--gold-soft);cursor:pointer;"> Back to Fuzzification</button>' +
                '</div>';
            p.dataset.tabPanel = '1';
            return p;
        }

        tabItems.forEach((btn, idx) => {
            btn.addEventListener('click', function () {
                tabItems.forEach(b => b.classList.remove('tab-bar__item--active'));
                this.classList.add('tab-bar__item--active');
                // remove the old overlay
                document.querySelectorAll('[data-tab-panel]').forEach(p => p.remove());

                if (idx === 1) {
                    // Rule Evaluation
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
                    // Defuzzification
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

        // Execute Trade button
        const dlExec = document.getElementById('dl-execute-trade');
        if (dlExec) {
            dlExec.addEventListener('click', () => {
                dlExec.disabled = true;
                showToast('Routing signal to Execution Agent…', 'info', 1600);
                setTimeout(() => {
                    dlExec.disabled = false;
                    const sig = document.getElementById('agg-signal')?.textContent || 'current signal';
                    showToast('Trade executed: ' + sig + '. Demo only, no real order.', 'demo', 4000);
                }, 1500);
            });
        }
    }


    // Backtesting quick pick

    // the quick pick sets the real backtest ticker, e.g. Bitcoin runs BTC
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
                    dot.innerHTML = '<svg viewBox="0 0 12 12" style="width:8px;height:8px;" fill="none"><path d="M2 6l3 3 5-5" stroke="#ffffff" stroke-width="1.5" stroke-linecap="round"/></svg>';
                }
                const ticker = this.dataset.ticker;
                const tickerEl = document.getElementById('bt-ticker');
                if (tickerEl && ticker) tickerEl.value = ticker;
                const name = this.querySelector('.asset-option__name')?.textContent || 'Asset';
                showToast('Backtest ticker set to ' + ticker + ' (' + name + ')', 'info', 1800);
            }
        });
    });

    // density slider label (only if the slider is on the page)
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


    // Settings (demo page)

    // show / hide key
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

    // connection test button
    const btnConnTest = document.getElementById('btn-connection-test');
    if (btnConnTest) {
        btnConnTest.addEventListener('click', () => {
            btnConnTest.disabled     = true;
            btnConnTest.textContent  = 'TESTING CONNECTIONS…';
            showToast('Checking connections…', 'info', 2000);
            setTimeout(() => {
                btnConnTest.disabled    = false;
                btnConnTest.textContent = 'RUN GLOBAL CONNECTION TEST';
                showToast('Demo only. Prices come from Yahoo Finance; news and the LLM need keys in .env.', 'warn', 5000);
            }, 2400);
        });
    }

    // agent toggles, found by their 38x20 size
    function isToggleTrack(el) {
        const s = el.style;
        return s && s.width === '38px' && s.height === '20px' &&
               s.borderRadius && s.position === 'relative';
    }

    document.querySelectorAll('.settings-grid div').forEach(el => {
        if (!isToggleTrack(el)) return;
        el.style.cursor = 'pointer';

        // start state from where the knob is
        const thumb    = el.firstElementChild;
        let isOn       = thumb && (thumb.style.left === '21px');
        const inAgentGrid = !!el.closest('.settings-agent-grid');
        const label    = inAgentGrid ? 'Agent' : 'Toggle';

        el.addEventListener('click', function () {
            isOn = !isOn;
            el.style.background = isOn ? 'var(--primary)' : 'rgba(20, 26, 34, 0.12)';
            el.style.transition  = 'background 0.3s';
            if (thumb) {
                thumb.style.left       = isOn ? '21px' : '3px';
                thumb.style.background = isOn ? 'var(--on-primary)' : 'rgba(20, 26, 34, 0.5)';
                thumb.style.transition = 'left 0.3s, background 0.3s';
            }
            showToast(label + (isOn ? ' enabled' : ' disabled') + ' (demo, nothing saved)', 'demo', 2800);
        });
    });

    // theme buttons
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
            showToast('Theme: ' + this.textContent.trim() + ' (demo, this browser only)', 'demo', 2500);
        });
    });

    // density buttons
    document.querySelectorAll('.settings-density-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            document.querySelectorAll('.settings-density-btn').forEach(b => {
                b.classList.remove('settings-density-btn--active');
                b.style.background  = '';
                b.style.borderColor = '';
                b.style.color       = '';
            });
            this.classList.add('settings-density-btn--active');
            this.style.background  = 'rgba(20, 26, 34, 0.09)';
            this.style.borderColor = 'rgba(20, 26, 34, 0.18)';
            this.style.color       = 'var(--gold-soft)';
            this.style.fontWeight  = '600';
            showToast('Layout: ' + this.textContent.trim() + ' density selected (demo)', 'demo', 2500);
        });
    });

    // calibration slider values
    document.querySelectorAll('.glass-panel input[type="range"]').forEach(slider => {
        if (slider.closest('.config-main')) return; // not the backtest page slider
        const parent = slider.parentElement;
        // the value span is already in the template
        slider.addEventListener('input', function () {
            const valSpan = this.parentElement.querySelector('.label-mono[style*="color: var(--primary)"], .label-mono[style*="color:var(--primary)"]');
            if (valSpan) {
                valSpan.textContent = (this.value / 100).toFixed(2);
            }
        });
    });

    // feedback when a setting changes
    let saveTimeout;
    document.querySelectorAll('.settings-grid .glass-input').forEach(inp => {
        if (inp.id === 'mi-search' || inp.id === 'bt-ticker' || inp.id === 'ticker-input') return;
        inp.addEventListener('change', () => {
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                showToast('Setting changed (demo, this browser only)', 'demo', 2200);
            }, 800);
        });
    });

    // agentic log on the Backtesting page
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
                    const col      = statusColor[status] || 'rgba(20, 26, 34, 0.45)';
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


    // Watchlist: click a row to analyse that ticker
    document.querySelectorAll('.watch-row[data-ticker]').forEach(row => {
        row.addEventListener('click', () => {
            if (!tickerInput || !analyzeBtn) return;
            document.querySelectorAll('.watch-row').forEach(r => r.classList.remove('is-active'));
            row.classList.add('is-active');
            tickerInput.value = row.dataset.ticker;
            analyzeBtn.click();
        });
    });

    // Dashboard bottom tabs
    document.querySelectorAll('[data-desk-tab]').forEach(tab => {
        tab.addEventListener('click', () => {
            const name = tab.dataset.deskTab;
            document.querySelectorAll('[data-desk-tab]').forEach(t => t.classList.toggle('is-on', t === tab));
            document.querySelectorAll('[data-desk-pane]').forEach(p => { p.hidden = p.dataset.deskPane !== name; });
        });
    });

    // Number keys 1-7 jump between pages (ignored while typing)
    const pages = ['/', '/market-intelligence', '/decision-logic', '/backtesting', '/agentic', '/walkthrough', '/settings'];
    document.addEventListener('keydown', e => {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        if (e.target.closest('input, textarea, select, [contenteditable]')) return;
        const i = parseInt(e.key, 10);
        if (i >= 1 && i <= pages.length && location.pathname !== pages[i - 1]) location.href = pages[i - 1];
    });

    // Show the hidden menu once on the first visit so people know it is there
    if (appSidebar && window.matchMedia('(min-width: 768px)').matches) {
        let seen = false;
        try { seen = localStorage.getItem('nt-menu-seen') === '1'; } catch (err) { seen = true; }
        if (!seen) {
            appSidebar.classList.add('sidebar--peek');
            setTimeout(() => appSidebar.classList.remove('sidebar--peek'), 1600);
            try { localStorage.setItem('nt-menu-seen', '1'); } catch (err) { /* private mode */ }
        }
    }

    // Top bar clock
    const topClock = document.getElementById('topbar-clock');
    if (topClock) {
        const tickTop = () => { topClock.textContent = new Date().toTimeString().slice(0, 8); };
        tickTop();
        setInterval(tickTop, 1000);
    }

});

