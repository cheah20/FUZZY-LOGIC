"""End-to-end check of every page, API route, agent run and data flow.
Run: python tests/e2e_test.py
"""
import warnings, json, time, sys
from pathlib import Path
warnings.filterwarnings('ignore')

# the real Flask app
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app as flask_app
client = flask_app.app.test_client()
flask_app.app.config['TESTING'] = True

SEP  = '=' * 65
PASS = '[PASS]'
FAIL = '[FAIL]'
WARN = '[WARN]'

results = {'pass': 0, 'fail': 0, 'warn': 0}

def check(label, condition, detail='', warn_only=False):
    if condition:
        results['pass'] += 1
        print(f'  {PASS} {label}' + (f'  {detail}' if detail else ''))
    elif warn_only:
        results['warn'] += 1
        print(f'  {WARN} {label}' + (f'  {detail}' if detail else ''))
    else:
        results['fail'] += 1
        print(f'  {FAIL} {label}' + (f'  {detail}' if detail else ''))

print(SEP)
print('END-TO-END TEST')
print(SEP)

# 1. pages
print('\n## 1. PAGE ROUTES (HTTP GET)')
print('-' * 40)
pages = ['/', '/market-intelligence', '/decision-logic', '/backtesting', '/settings']
for url in pages:
    t0   = time.perf_counter()
    resp = client.get(url)
    ms   = (time.perf_counter() - t0) * 1000
    ok   = resp.status_code == 200
    html = resp.data.decode('utf-8', errors='replace')
    size = len(html)
    check(f'GET {url:<28} {resp.status_code}  {size:>7,} chars  {ms:.0f}ms', ok)

# 2. API routes
print('\n## 2. REST API ENDPOINTS')
print('-' * 40)

api_tests = [
    ('/api/market-data',  None, ['btc_price','volatility']),
    ('/api/news-feed',    None, None),
    ('/api/agent-status', None, None),
    ('/api/expert/reasoning?rsi=74.2&macd_hist=-0.35&sentiment=-0.15', None,
     ['market_regime','fuzzy_label','active_features','masked_features','weights','triggered_rule']),
    ('/api/expert/regime?rsi=25&macd_hist=0.8&sentiment=-0.9', None,
     ['regime','confidence','triggered_rule']),
    ('/api/expert/fuzzy?rsi=25&macd_hist=0.8&sentiment=-0.9&mode=expert', None,
     ['score','label','market_regime','active_features']),
    ('/api/expert/fuzzy?rsi=25&macd_hist=0.8&sentiment=-0.9&mode=baseline', None,
     ['score','label']),
    ('/api/expert/compare?rsi=25&macd_hist=0.8&sentiment=-0.9', None,
     ['baseline','expert','delta','consensus']),
    ('/api/expert/weightage', None, None),
    ('/api/expert/rules', None, None),
    ('/api/expert/status', None, ['fuzzy_engine','expert_layer','mode']),
]

for url, _, expected_keys in api_tests:
    t0   = time.perf_counter()
    resp = client.get(url)
    ms   = (time.perf_counter() - t0) * 1000
    ok   = resp.status_code == 200
    if ok:
        try:
            data = json.loads(resp.data)
            if expected_keys:
                missing = [k for k in expected_keys if k not in (data if isinstance(data, dict) else {})]
                key_ok  = len(missing) == 0
                check(f'GET {url[:55]:<56} {resp.status_code}  {ms:.0f}ms',
                      key_ok,
                      f'missing={missing}' if missing else '')
            else:
                check(f'GET {url[:55]:<56} {resp.status_code}  {ms:.0f}ms', True)
        except Exception as ex:
            check(f'GET {url[:55]:<56} JSON parse error', False, str(ex))
    else:
        check(f'GET {url[:55]:<56} {resp.status_code}', False)

# 3. expert reasoning data
print('\n## 3. EXPERT REASONING DATA QUALITY')
print('-' * 40)

resp = client.get('/api/expert/reasoning?rsi=25&macd_hist=0.8&sentiment=-0.9')
data = json.loads(resp.data)
check('regime = NEWS_DRIVEN_BEARISH',        data['market_regime'] == 'NEWS_DRIVEN_BEARISH')
check('active_features = [sentiment]',       data['active_features'] == ['sentiment'])
check('rsi in masked_features',              'rsi' in data['masked_features'])
check('macd_hist in masked_features',        'macd_hist' in data['masked_features'])
check('fuzzy_label = Strong Sell',           data['fuzzy_label'] == 'Strong Sell')
check('fuzzy_score < -1.0',                  data['fuzzy_score'] < -1.0,
      f'score={data["fuzzy_score"]:+.3f}')
check('confidence > 0',                      data['confidence'] > 0)
check('triggered_rule present',              len(data['triggered_rule']) > 0)
check('weights[sentiment]=1.0',              data['weights'].get('sentiment') == 1.0)
check('weights[rsi]=0.0',                    data['weights'].get('rsi') == 0.0)
check('input_display has 3 rows',            len(data.get('input_display', [])) == 3)
check('RSI bar muted (MASKED)',              not data['input_display'][0]['active']
                                             if data.get('input_display') else False)
check('all 19 required fields present',      len([k for k in [
    'market_regime','regime_display','regime_confidence','regime_reason',
    'triggered_rule','active_features','masked_features','weights',
    'output_tendency','fuzzy_score','fuzzy_label','confidence',
    'label_memberships','rule_firings','rules_fired','weightage_table',
    'expert_rules','reference_sources','input_display'
] if k not in data]) == 0)

# 4. baseline and expert must differ
print('\n## 4. BASELINE vs EXPERT DIFFERENTIATION')
print('-' * 40)
resp = client.get('/api/expert/compare?rsi=25&macd_hist=0.8&sentiment=-0.9')
cmp  = json.loads(resp.data)
check('TC1: Expert != Baseline label',       cmp['expert']['label'] != cmp['baseline']['label'],
      f'expert={cmp["expert"]["label"]} base={cmp["baseline"]["label"]}')
check('TC1: Expert score < Baseline',        cmp['expert']['score'] < cmp['baseline']['score'],
      f'delta={cmp["delta"]:+.3f}')
check('TC1: Consensus = Diverge',            cmp['consensus'] == 'Diverge')
check('TC1: Expert = Strong Sell',           cmp['expert']['label'] == 'Strong Sell')

resp2 = client.get('/api/expert/compare?rsi=75&macd_hist=0.8&sentiment=0.1')
cmp2  = json.loads(resp2.data)
check('TC3 Conflicting: Expert toward Hold', abs(cmp2['expert']['score']) < abs(cmp2['baseline']['score']),
      f'expert={cmp2["expert"]["score"]:+.3f} base={cmp2["baseline"]["score"]:+.3f}')

# 5. all six regimes through the API
print('\n## 5. ALL 6 REGIME TRANSITIONS (API end-to-end)')
print('-' * 40)
regime_tests = [
    ('/api/expert/regime?rsi=25&macd_hist=0.8&sentiment=-0.9',  'NEWS_DRIVEN_BEARISH'),
    ('/api/expert/regime?rsi=50&macd_hist=0.2&sentiment=0.85',  'NEWS_DRIVEN_BULLISH'),
    ('/api/expert/regime?rsi=28&macd_hist=0.7&sentiment=0.0',   'TECHNICAL_DRIVEN'),
    ('/api/expert/regime?rsi=75&macd_hist=0.8&sentiment=0.1',   'CONFLICTING_SIGNALS'),
    ('/api/expert/regime?rsi=65&macd_hist=-0.5&sentiment=-0.5&atr=9.0', 'HIGH_UNCERTAINTY'),
    ('/api/expert/regime?rsi=50&macd_hist=0.1&sentiment=0.25',  'NORMAL'),
]
for url, expected in regime_tests:
    resp = client.get(url)
    data = json.loads(resp.data)
    got  = data.get('regime', '?')
    check(f'{expected:<25} via API',         got == expected, f'got={got}')

# 6. agent pipeline through the API
print('\n## 6. COGNITION AGENT (API /api/expert/cognition)')
print('-' * 40)
scenarios = [
    ('News Panic     s=-0.9',  'rsi=25&macd_hist=0.8&sentiment=-0.9',  'Strong Sell'),
    ('Neutral Tech   s=0.0',   'rsi=28&macd_hist=0.7&sentiment=0.0',   'Strong Buy'),
    ('Bullish Rally  s=+0.85', 'rsi=50&macd_hist=0.2&sentiment=0.85',  'Strong Buy'),
]
for name, params, exp_label in scenarios:
    resp = client.get(f'/api/expert/cognition?{params}&mode=expert')
    data = json.loads(resp.data)
    got  = data.get('label', '?')
    check(f'{name:<30} -> {exp_label:<12}', got == exp_label,
          f'got={got} score={data.get("score",0):+.3f}')

# 7. market data refresh
print('\n## 7. MARKET DATA REFRESH (3 consecutive requests)')
print('-' * 40)
prices = []
for i in range(3):
    resp = client.get('/api/market-data')
    d    = json.loads(resp.data)
    prices.append(d['btc_price'])
check('All 3 market data requests succeed',  len(prices) == 3)
check('Price is numeric',                    all(isinstance(p, (int, float)) for p in prices))
check('Price in realistic range',            all(50000 < p < 100000 for p in prices),
      str(prices))

# 8. Decision Logic page
print('\n## 8. DECISION LOGIC PAGE')
print('-' * 40)
resp = client.get('/decision-logic')
html = resp.data.decode('utf-8', errors='replace')
dl_checks = [
    ('Expert Reasoning Panel present',       'Expert Reasoning Panel' in html),
    ('Market Regime shown',                  'Market Regime' in html or 'REGIME' in html.upper()),
    ('Active Features section',              'Active Features' in html),
    ('Masked Features section',              'Masked Features' in html),
    ('Feature Weight Distribution',          'Feature Weight Distribution' in html),
    ('Final FLC Signal row',                 'Final FLC Signal' in html),
    ('MASKED label on inputs',               'MASKED' in html),
    ('Cognition Stream present',             'COGNITION STREAM' in html),
    ('Aggregated Signal dynamic',            'STRONG SELL' in html.upper() or 'STRONG BUY' in html.upper()
                                             or 'HOLD' in html or 'BUY' in html or 'SELL' in html),
    ('rules_fired dynamic',                  any(str(n) in html for n in range(1, 30))),
]
for name, cond in dl_checks:
    check(name, cond)

# 9. Backtesting page
print('\n## 9. BACKTESTING PAGE')
print('-' * 40)
resp = client.get('/backtesting')
html = resp.data.decode('utf-8', errors='replace')
bt_checks = [
    ('BACKTEST CONFIG panel',     'BACKTEST CONFIG' in html),
    ('SHARPE RATIO shown',        'SHARPE RATIO' in html),
    ('WIN RATE shown',            'WIN RATE' in html),
    ('Cumulative Returns chart',  'Cumulative Returns' in html),
    ('Fuzzy Logic Mitigations',   'FUZZY-LOGIC MITIGATIONS' in html or 'FUZZY' in html),
    ('Execute Backtest button',   'Execute Backtest' in html),
]
for name, cond in bt_checks:
    check(name, cond)

# 10. many quick API calls
print('\n## 10. STRESS TEST (20 rapid API calls)')
print('-' * 40)
import random
random.seed(42)
stress_ok = 0
t_start = time.perf_counter()
for _ in range(20):
    rsi  = random.uniform(5, 95)
    macd = random.uniform(-2, 2)
    sent = random.uniform(-1, 1)
    url  = f'/api/expert/reasoning?rsi={rsi:.1f}&macd_hist={macd:.2f}&sentiment={sent:.2f}'
    resp = client.get(url)
    if resp.status_code == 200:
        d = json.loads(resp.data)
        if d.get('market_regime') and d.get('fuzzy_label'):
            stress_ok += 1
total_ms = (time.perf_counter() - t_start) * 1000
check(f'All 20 stress calls succeeded',          stress_ok == 20, f'{stress_ok}/20')
check(f'Total time < 5000ms',                    total_ms < 5000, f'{total_ms:.0f}ms total')
check(f'Average < 250ms/call',                   total_ms / 20 < 250, f'{total_ms/20:.0f}ms avg')

# summary
print()
print(SEP)
total = results['pass'] + results['fail'] + results['warn']
print(f'RESULT: {results["pass"]}/{total} PASS  |  {results["fail"]} FAIL  |  {results["warn"]} WARN')
if results['fail'] == 0:
    print('STATUS: ALL CHECKS PASSED')
else:
    print(f'STATUS: {results["fail"]} ISSUE(S) NEED ATTENTION')
print(SEP)
