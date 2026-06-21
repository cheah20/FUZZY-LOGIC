import warnings, json, sys
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as flask_app
client = flask_app.app.test_client()
flask_app.app.config['TESTING'] = True

print('=== FUNCTIONAL INTERACTION RECOVERY VALIDATION ===\n')

# 1. Templates
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader(str(ROOT / 'templates')))
print('## 1. TEMPLATES')
for t in ['base.html','dashboard.html','backtesting.html','decision_logic.html']:
    try:
        env.get_template(t)
        print(f'  [PASS] {t}')
    except Exception as e:
        print(f'  [FAIL] {t}: {e}')

# 2. API endpoints
print('\n## 2. NEW API ENDPOINTS')
api_tests = [
    ('/api/analyze?ticker=AAPL',        200),
    ('/api/analyze?ticker=TSLA',        200),
    ('/api/analyze?ticker=BTC',         200),
    ('/api/analyze?ticker=MSFT',        200),
    ('/api/analyze?ticker=XXXXBAD',     404),
    ('/api/backtest?ticker=AAPL&period=1y', 200),
    ('/api/backtest?ticker=TSLA&period=1y', 200),
    ('/api/backtest?ticker=MSFT&period=2y', 200),
    ('/api/backtest?ticker=XXXXBAD&period=1y', 404),
]
for url, expected in api_tests:
    r  = client.get(url)
    d  = json.loads(r.data)
    ok = (r.status_code == expected)
    ticker = url.split('ticker=')[1].split('&')[0]
    if ok and r.status_code == 200:
        if 'analyze' in url:
            detail = 'price=${:,.2f} signal={} chart_bars={}'.format(
                d.get('price', 0),
                d.get('expert_label', '?'),
                len(d.get('chart_data', {}).get('close', [])))
        else:
            ecurve = d.get('expert', {}).get('equity_curve', [])
            detail = 'bars={} expert={:+.2f}% aligned={}'.format(
                d.get('total_bars'),
                d.get('expert', {}).get('total_return', 0),
                len(ecurve) == len(d.get('bah_curve', [])))
    elif ok:
        detail = 'error 404 returned correctly'
    else:
        detail = 'UNEXPECTED status {}'.format(r.status_code)
    print('  [{}] {:<22} {:<10} {}'.format(
        'PASS' if ok else 'FAIL',
        url.split('?')[0], ticker, detail))

# 3. Page content
print('\n## 3. PAGE CONTENT (id elements + interactive buttons)')
from flask import render_template
from interface import get_expert_reasoning_data
from backtester import get_backtesting_metrics, get_cached_comparison
from Trading_Data import get_simulated_market_snapshot
snap = get_simulated_market_snapshot()
ed   = get_expert_reasoning_data(snap['rsi'], snap['macd_hist'], snap['sentiment'], snap['atr'])
mkt  = {'btc_price': snap['price'], 'volatility': snap['volatility'],
        'active_agents': 4, 'synced_agents': 4, 'system_signal': ed['fuzzy_label'].upper()}
with flask_app.app.test_request_context('/'):
    dash = render_template('dashboard.html', market_data=mkt, agent_pipeline=[], expert_data=ed)
    bt   = render_template('backtesting.html',
                           backtesting_metrics=get_backtesting_metrics(),
                           comparison=get_cached_comparison())

checks = [
    ('Dashboard: ticker-input',      'id="ticker-input"'        in dash),
    ('Dashboard: analyze-btn',       'id="analyze-btn"'         in dash),
    ('Dashboard: live-price id',     'id="live-price"'          in dash),
    ('Dashboard: live-signal id',    'id="live-signal"'         in dash),
    ('Dashboard: chart container',   'id="live-chart-container"' in dash),
    ('Dashboard: agg-signal id',     'id="agg-signal"'          in dash),
    ('Dashboard: Expert Panel',      'EXPERT REASONING'         in dash),
    ('Backtesting: bt-ticker',       'id="bt-ticker"'           in bt),
    ('Backtesting: bt-run-btn',      'id="bt-run-btn"'          in bt),
    ('Backtesting: period buttons',  'bt-period-btn'            in bt),
    ('Backtesting: live results div','id="bt-live-results"'     in bt),
    ('Backtesting: equity chart div','id="bt-chart-container"'  in bt),
    ('Backtesting: status element',  'id="bt-status"'           in bt),
    ('Backtesting: expert-return id','id="bt-expert-return"'    in bt),
    ('Backtesting: winner badge id', 'id="bt-result-winner"'    in bt),
]
for label, cond in checks:
    print('  [{}] {}'.format('PASS' if cond else 'FAIL', label))

# 4. main.js
print('\n## 4. main.js INTERACTION LAYER')
js = open(ROOT / 'static/js/main.js', encoding='utf-8').read()
fns = [
    ('runAnalysis() async',       'async function runAnalysis'),
    ('updateDashboardMetrics()',   'function updateDashboardMetrics'),
    ('drawPriceChart()',           'function drawPriceChart'),
    ('runBacktest() async',        'async function runBacktest'),
    ('displayBacktestResults()',   'function displayBacktestResults'),
    ('drawEquityCurve()',          'function drawEquityCurve'),
    ('fetch /api/analyze',        '/api/analyze'),
    ('fetch /api/backtest',       '/api/backtest'),
    ('404 error handling',        'data.error'),
    ('loading/disabled state',    'setAnalyzeLoading'),
    ('Enter key handler',         'Enter'),
    ('uppercase ticker input',    'toUpperCase()'),
    ('SVG price chart',           'drawPriceChart'),
    ('SVG equity curve',          'drawEquityCurve'),
    ('period selector',           'activePeriod'),
]
for label, token in fns:
    print('  [{}] {}'.format('PASS' if token in js else 'FAIL', label))

# 5. Human Expert still active
print('\n## 5. HUMAN EXPERT LAYER STILL ACTIVE')
from fuzzy_engine import compute_full_pipeline
tc1_e = compute_full_pipeline(rsi=25, macd_hist=0.8, sentiment=-0.9, mode='expert')
tc1_b = compute_full_pipeline(rsi=25, macd_hist=0.8, sentiment=-0.9, mode='baseline')
print('  TC1 Expert  : {} ({:+.3f})'.format(tc1_e['label'], tc1_e['score']))
print('  TC1 Baseline: {} ({:+.3f})'.format(tc1_b['label'], tc1_b['score']))
print('  Delta: {:+.3f}'.format(tc1_e['score'] - tc1_b['score']))
print('  [{}] Human Expert changes decision'.format(
    'PASS' if tc1_e['label'] != tc1_b['label'] else 'FAIL'))

print('\n=== VALIDATION COMPLETE ===')
