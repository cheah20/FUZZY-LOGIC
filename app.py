from flask import Flask, render_template, jsonify, request
import json
from datetime import datetime

# Load .env file before anything else so DEEPSEEK_API_KEY / MARKETAUX_API_KEY
# / NEWS_API_PROVIDER / NEWS_REFRESH_MINUTES are available to the rest of the app.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — run: pip install python-dotenv

# ASTDSS Expert System Modules
from expert_layer import detect_market_regime, get_expert_layer_info
from feature_selector import select_features, get_weightage_table
from fuzzy_engine import compute_score, compute_full_pipeline
from tools import fuzzy_analysis_tool, compare_modes_tool
from crew import run_cognition_analysis, get_pipeline
from Trading_Data import get_simulated_market_snapshot, get_ticker_snapshot, get_ohlcv_chart_data
from backtester import get_backtesting_metrics, get_cached_comparison, run_backtest, _buy_and_hold_return
from interface import get_expert_reasoning_data, compare_baseline_vs_expert, get_system_status

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Simulated Market State
# This holds live-updating simulated data for the UI
_current_snapshot = get_simulated_market_snapshot()

market_data = {
    'btc_price': 64231.01,
    'volatility': 846.15,
    'active_agents': 4,
    'synced_agents': 4,
    'system_signal': 'OVERBOUGHT'
}

news_feed = [
    {
        'time': '14:20:00',
        'content': 'S&P 500 futures dip on hawkish Fed comments regarding sustained high interest rates through Q3.',
        'sentiment': 'BEARISH',
        'tags': ['BEARISH SENTIMENT', 'MACRO POLICY']
    },
    {
        'time': '14:15:22',
        'content': 'Tech sector rally gains momentum as major chipmaker announces breakthrough in 2nm production efficiency.',
        'sentiment': 'BULLISH',
        'tags': ['BULLISH SENTIMENT', 'SEMICONDUCTORS']
    },
    {
        'time': '14:02:49',
        'content': 'Crude oil volatility spikes following unscheduled OPEC technical meeting announcement.',
        'sentiment': 'HIGH VOLATILITY',
        'tags': ['HIGH VOLATILITY', 'ENERGY']
    },
    {
        'time': '13:55:12',
        'content': 'European Central Bank maintains deposit facility rate, signals "data-dependent" approach for upcoming sessions.',
        'sentiment': 'UNCERTAIN',
        'tags': ['UNCERTAIN SUPPLY', 'CENTRAL BANKS']
    }
]

agent_pipeline = [
    {'name': 'Daily Data', 'source': 'Binance WS', 'status': 'Synced', 'badge_color': 'green'},
    {'name': 'Perception', 'source': 'Model-VT Market', 'status': 'Idle', 'badge_color': 'gray'},
    {'name': 'Cognition', 'source': 'Fuzzy Logic Synthesis', 'status': 'Processing', 'badge_color': 'orange'},
    {'name': 'Execution', 'source': 'ArKey Active', 'status': 'Waiting', 'badge_color': 'gray'}
]

fuzzy_logic_data = {
    'rsi_intensity': 74.2,
    'volatility_spike': 8.12,
    'sentiment_score': 0.88,
    'rules_active': 12
}

backtesting_metrics = get_backtesting_metrics()

system_config = {
    'core_agent': 'Active Sync',
    'system_version': 'v5.0.0-EXPERT',
    'agents': [
        {'name': 'Daily Data Agent', 'description': 'ETL & Normalization', 'logic_weight': 85, 'status': 'SYNCED'},
        {'name': 'Perception Agent', 'description': 'Sentiment & Trend Extraction', 'logic_weight': 62, 'status': 'SCANNING'},
        {'name': 'Cognition Agent', 'description': 'Symbolic Reasoning + Expert Rules', 'logic_weight': 95, 'status': 'INFERENCING...'},
        {'name': 'Execution Agent', 'description': 'Automated Order Routing', 'logic_weight': 40, 'status': 'STANDBY'}
    ]
}


# Page Routes

@app.route('/')
def dashboard():
    # Refresh snapshot for live feel
    global _current_snapshot
    _current_snapshot = get_simulated_market_snapshot()
    market_data['btc_price'] = _current_snapshot['price']
    market_data['volatility'] = _current_snapshot['volatility']

    # Get expert analysis for the aggregate signal bar
    expert_data = get_expert_reasoning_data(
        rsi=_current_snapshot['rsi'],
        macd_hist=_current_snapshot['macd_hist'],
        sentiment=_current_snapshot['sentiment'],
        atr=_current_snapshot['atr'],
    )

    return render_template('dashboard.html',
                         market_data=market_data,
                         agent_pipeline=agent_pipeline,
                         expert_data=expert_data)


@app.route('/market-intelligence')
def market_intelligence():
    return render_template('market_intelligence.html',
                         news_feed=news_feed)


@app.route('/decision-logic')
def decision_logic():
    # Get real expert reasoning data for the Decision Logic page
    snapshot = get_simulated_market_snapshot()
    expert_data = get_expert_reasoning_data(
        rsi=snapshot['rsi'],
        macd_hist=snapshot['macd_hist'],
        sentiment=snapshot['sentiment'],
        atr=snapshot['atr'],
    )

    return render_template('decision_logic.html',
                         fuzzy_logic_data=fuzzy_logic_data,
                         expert_data=expert_data)


@app.route('/backtesting')
def backtesting():
    return render_template('backtesting.html',
                         backtesting_metrics=backtesting_metrics,
                         comparison=get_cached_comparison())


@app.route('/settings')
def settings():
    return render_template('settings.html',
                         system_config=system_config)


# REST API Routes — Expert System

@app.route('/api/market-data')
def get_market_data():
    global _current_snapshot
    _current_snapshot = get_simulated_market_snapshot()
    market_data['btc_price'] = _current_snapshot['price']
    market_data['volatility'] = _current_snapshot['volatility']
    return jsonify(market_data)


@app.route('/api/news-feed')
def get_news_feed():
    return jsonify(news_feed)


@app.route('/api/market-news')
def api_market_news():
    """
    Real (or demo-fallback) finance/trading news + AI Market Brief.
    GET /api/market-news?symbol=AAPL  (symbol optional — defaults to general market)
    """
    from news_service import get_market_news
    symbol = request.args.get('symbol', '').strip() or None
    try:
        result = get_market_news(symbol=symbol)
        return jsonify(result)
    except Exception as ex:
        # Never let the news layer take down the page — degrade to an explicit error
        # the frontend can show, rather than a 500.
        return jsonify({
            'source': 'demo', 'is_live': False, 'symbol': symbol or 'GENERAL',
            'articles': [], 'ai_brief': {'text': 'Market news unavailable.', 'mode': 'rule_based'},
            'error': str(ex),
        }), 200


@app.route('/api/agent-status')
def get_agent_status():
    pipeline = get_pipeline()
    return jsonify(pipeline.get_agent_status())


@app.route('/api/expert/reasoning')
def api_expert_reasoning():
    """Full expert reasoning data for the UI panel."""
    rsi = float(request.args.get('rsi', 74.2))
    macd_hist = float(request.args.get('macd_hist', -0.35))
    sentiment = float(request.args.get('sentiment', -0.15))
    atr_str = request.args.get('atr')
    atr = float(atr_str) if atr_str else None

    data = get_expert_reasoning_data(rsi, macd_hist, sentiment, atr)
    return jsonify(data)


@app.route('/api/expert/regime')
def api_market_regime():
    """Detect market regime only."""
    rsi = float(request.args.get('rsi', 50))
    macd_hist = float(request.args.get('macd_hist', 0))
    sentiment = float(request.args.get('sentiment', 0))
    atr_str = request.args.get('atr')
    atr = float(atr_str) if atr_str else None

    result = detect_market_regime(rsi, macd_hist, sentiment, atr)
    return jsonify(result)


@app.route('/api/expert/fuzzy')
def api_fuzzy_analysis():
    """Run fuzzy analysis with specified mode."""
    rsi = float(request.args.get('rsi', 50))
    macd_hist = float(request.args.get('macd_hist', 0))
    sentiment = float(request.args.get('sentiment', 0))
    atr_str = request.args.get('atr')
    atr = float(atr_str) if atr_str else None
    mode = request.args.get('mode', 'expert')

    if mode == 'baseline':
        result = compute_score(rsi, macd_hist, sentiment, atr)
    else:
        result = compute_full_pipeline(rsi, macd_hist, sentiment, atr, mode='expert')

    return jsonify(result)


@app.route('/api/expert/compare')
def api_compare_modes():
    """Compare baseline vs expert FLC side-by-side."""
    rsi = float(request.args.get('rsi', 74.2))
    macd_hist = float(request.args.get('macd_hist', -0.35))
    sentiment = float(request.args.get('sentiment', -0.15))
    atr_str = request.args.get('atr')
    atr = float(atr_str) if atr_str else None

    result = compare_modes_tool(rsi, macd_hist, sentiment, atr)
    return jsonify(result)


@app.route('/api/expert/cognition')
def api_cognition():
    """Run the full Cognition Agent analysis."""
    rsi = float(request.args.get('rsi', 74.2))
    macd_hist = float(request.args.get('macd_hist', -0.35))
    sentiment = float(request.args.get('sentiment', -0.15))
    atr_str = request.args.get('atr')
    atr = float(atr_str) if atr_str else None
    mode = request.args.get('mode', 'expert')

    result = run_cognition_analysis(rsi, macd_hist, sentiment, atr, mode=mode)
    return jsonify(result)


@app.route('/api/expert/weightage')
def api_weightage_table():
    """Get the feature weightage table for all regimes."""
    return jsonify(get_weightage_table())


@app.route('/api/expert/rules')
def api_expert_rules():
    """Get the list of expert rules with descriptions."""
    info = get_expert_layer_info()
    return jsonify(info['rules'])


@app.route('/api/expert/status')
def api_system_status():
    """Get overall expert system status."""
    return jsonify(get_system_status())


@app.route('/api/market-snapshot')
def api_market_snapshot():
    """Get simulated market snapshot with all indicators."""
    global _current_snapshot
    _current_snapshot = get_simulated_market_snapshot()
    return jsonify(_current_snapshot)


# Interactive Analysis & Backtest APIs (Functional Interaction Recovery)

@app.route('/api/analyze')
def api_analyze():
    """
    Full analysis pipeline for a user-specified ticker.
    GET /api/analyze?ticker=AAPL

    Flow: yfinance data -> expert_layer -> feature_selector
          -> fuzzy_engine -> explainability panel data
    """
    ticker = request.args.get('ticker', 'AAPL').strip()
    if not ticker:
        return jsonify({'error': 'ticker parameter is required'}), 400

    # Fetch real market data — reject if yfinance failed (source contains 'simulated')
    snap = get_ticker_snapshot(ticker, period='90d')
    if 'simulated' in snap.get('source', ''):
        return jsonify({'error': f'Ticker "{ticker.upper()}" not found or has no data. Please check the symbol (e.g. AAPL, TSLA, BTC, ETH).'}), 404

    # Run full expert pipeline
    try:
        reasoning = get_expert_reasoning_data(
            rsi=snap['rsi'],
            macd_hist=snap['macd_hist'],
            sentiment=snap['sentiment'],
            atr=snap.get('atr'),
        )
    except Exception as ex:
        return jsonify({'error': f'Analysis failed: {ex}'}), 500

    # Fetch OHLCV for chart rendering
    chart = get_ohlcv_chart_data(ticker, period='3mo')

    return jsonify({
        'ticker':          snap.get('ticker', ticker.upper()),
        'symbol':          snap.get('ticker_symbol', ticker.upper()),
        'price':           snap['price'],
        'rsi':             snap['rsi'],
        'macd_hist':       snap['macd_hist'],
        'atr':             snap.get('atr'),
        'sentiment':       snap['sentiment'],
        'volatility':      snap.get('volatility', 0),
        'source':          snap.get('source', 'unknown'),
        # Expert system output
        'market_regime':   reasoning['market_regime'],
        'regime_display':  reasoning['regime_display'],
        'regime_confidence': reasoning['regime_confidence'],
        'triggered_rule':  reasoning['triggered_rule'],
        'active_features': reasoning['active_features'],
        'masked_features': reasoning['masked_features'],
        'weights':         reasoning['weights'],
        'expert_label':    reasoning['fuzzy_label'],
        'expert_score':    reasoning['fuzzy_score'],
        'confidence':      reasoning['confidence'],
        'output_tendency': reasoning['output_tendency'],
        'regime_reason':   reasoning['regime_reason'],
        'input_display':   reasoning['input_display'],
        # Chart data
        'chart_data':      chart if 'error' not in chart else None,
    })


@app.route('/api/backtest')
def api_backtest():
    """
    Run backtesting for a user-specified ticker.
    GET /api/backtest?ticker=AAPL&period=1y

    Fetches real historical prices, runs baseline vs expert FLC,
    returns metrics + equity curves for chart rendering.
    """
    ticker  = request.args.get('ticker', 'AAPL').strip()
    period  = request.args.get('period', '2y')

    if not ticker:
        return jsonify({'error': 'ticker parameter is required'}), 400

    try:
        import yfinance as yf
        import warnings, random as _rnd
        warnings.filterwarnings('ignore')

        from Trading_Data import normalize_ticker
        symbol = normalize_ticker(ticker)
        hist   = yf.Ticker(symbol).history(period=period, interval='1d')

        if hist.empty or len(hist) < 60:
            return jsonify({'error': f'Not enough historical data for "{ticker}" (need 60+ bars). Try a different ticker or a longer period.'}), 404

        closes    = [float(x) for x in hist['Close']]
        dates     = [str(d.date()) for d in hist.index]

        # Sentiment proxy from 5-day price momentum
        sentiments = []
        _rnd.seed(42)
        for i in range(len(closes)):
            if i >= 5:
                pct = (closes[i] - closes[i-5]) / closes[i-5]
                s   = max(-1.0, min(1.0, pct * 20 + _rnd.uniform(-0.1, 0.1)))
            else:
                s = _rnd.uniform(-0.15, 0.15)
            sentiments.append(round(s, 4))

        # 70/30 split
        split     = int(len(closes) * 0.70)
        bah       = _buy_and_hold_return(closes)

        baseline  = run_backtest(closes, sentiments, mode='baseline')
        expert    = run_backtest(closes, sentiments, mode='expert')
        b_test    = run_backtest(closes[split:], sentiments[split:], mode='baseline')
        e_test    = run_backtest(closes[split:], sentiments[split:], mode='expert')

        # Normalise daily equity to % return, padded to match len(closes)
        def daily_to_pct(result_dict):
            curve = result_dict.get('daily_equity') or result_dict['equity_curve']
            base  = curve[0] if curve else 10000.0
            pct   = [round((v / base - 1) * 100, 2) for v in curve]
            # Pad front with 0.0 so length matches closes (lookback gap)
            while len(pct) < len(closes):
                pct.insert(0, 0.0)
            return pct

        # Buy-and-hold daily % curve aligned to price bars
        base_price = closes[0]
        bah_curve  = [round((c / base_price - 1) * 100, 2) for c in closes]

        return jsonify({
            'ticker':           ticker.upper(),
            'symbol':           symbol,
            'total_bars':       len(closes),
            'split_index':      split,
            'dates':            dates,
            'buy_hold_return':  round(bah, 2),
            # Full-period metrics
            'expert': {
                'total_return': expert['total_return_pct'],
                'win_rate':     expert['win_rate_pct'],
                'max_drawdown': expert['max_drawdown_pct'],
                'sharpe':       expert['sharpe_ratio'],
                'num_trades':   expert['num_trades'],
                'equity_curve': daily_to_pct(expert),
                'trades':       expert.get('trades', []),
            },
            'baseline': {
                'total_return': baseline['total_return_pct'],
                'win_rate':     baseline['win_rate_pct'],
                'max_drawdown': baseline['max_drawdown_pct'],
                'sharpe':       baseline['sharpe_ratio'],
                'num_trades':   baseline['num_trades'],
                'equity_curve': daily_to_pct(baseline),
                'trades':       baseline.get('trades', []),
            },
            # Out-of-sample (30%)
            'oos_expert':   {'total_return': e_test['total_return_pct'], 'sharpe': e_test['sharpe_ratio']},
            'oos_baseline': {'total_return': b_test['total_return_pct'], 'sharpe': b_test['sharpe_ratio']},
            'bah_curve':    bah_curve,
        })

    except Exception as ex:
        return jsonify({'error': f'Backtest failed: {ex}'}), 500


# Agentic Layer — PA3 Routes

@app.route('/agentic')
def agentic():
    """Agentic Analysis page — PA3 Agentic Fuzzy Layer."""
    from agent_profile import get_agent_profile
    from human_expert_kb import get_kb_summary, get_knowledge_base
    return render_template(
        'agentic.html',
        agent_profile=get_agent_profile(),
        kb_summary=get_kb_summary(),
        kb_rules=get_knowledge_base(),
    )


@app.route('/walkthrough')
def walkthrough():
    """Guided step-by-step walkthrough of the agentic pipeline (learning aid)."""
    return render_template('walkthrough.html')


@app.route('/api/agentic/analyze')
def api_agentic_analyze():
    """
    Run the full agentic decision pipeline.
    GET /api/agentic/analyze?symbol=AAPL&demo=true
    Optional overrides: rsi, macd_hist, atr, sentiment
    """
    from agentic_pipeline import run_agentic_analysis
    symbol   = request.args.get('symbol', 'DEMO')
    demo     = request.args.get('demo', 'true').lower() != 'false'
    rsi_arg      = request.args.get('rsi')
    macd_arg     = request.args.get('macd_hist')
    atr_arg      = request.args.get('atr')
    sent_arg     = request.args.get('sentiment')
    # Validate numeric params at the API boundary before calling pipeline
    try:
        rsi_val  = float(rsi_arg)  if rsi_arg  else None
        macd_val = float(macd_arg) if macd_arg else None
        atr_val  = float(atr_arg)  if atr_arg  else None
        sent_val = float(sent_arg) if sent_arg else None
    except ValueError as ve:
        return jsonify({'error': f'Invalid numeric parameter: {ve}'}), 400
    try:
        result = run_agentic_analysis(
            symbol=symbol,
            demo=demo,
            rsi=rsi_val,
            macd_hist=macd_val,
            atr=atr_val,
            sentiment=sent_val,
        )
        return jsonify(result)
    except Exception as ex:
        return jsonify({'error': f'Agentic analysis failed: {ex}'}), 500


@app.route('/api/agentic/experience-log')
def api_agentic_experience_log():
    """
    Return the most recent experience log entries.
    GET /api/agentic/experience-log?limit=10
    """
    from memory_log import get_recent_experience
    limit = min(int(request.args.get('limit', 10)), 50)
    return jsonify(get_recent_experience(limit))


@app.route('/api/agentic/failure-demo')
def api_agentic_failure_demo():
    """
    Return a structured failure scenario demonstration.
    GET /api/agentic/failure-demo?scenario=0
    scenario 0 = ATR misinterpretation (primary)
    scenario 1 = MACD single-indicator overconfidence
    """
    from failure_reflection import get_failure_demo, get_all_failure_demos
    all_arg = request.args.get('all', 'false').lower()
    if all_arg == 'true':
        return jsonify(get_all_failure_demos())
    scenario = int(request.args.get('scenario', 0))
    return jsonify(get_failure_demo(scenario))


@app.route('/api/agentic/human-expert-kb')
def api_agentic_kb():
    """
    Return the full Human Expert Knowledge Base.
    GET /api/agentic/human-expert-kb
    """
    from human_expert_kb import get_knowledge_base, get_kb_summary
    return jsonify({
        'rules':   get_knowledge_base(),
        'summary': get_kb_summary(),
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=True, reloader_type='stat')
