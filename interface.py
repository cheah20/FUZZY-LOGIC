#GUYS make sure read the README.md before running this code, 
# it has important instructions on how to set up and run the application.
# because i use ollama
# run main code in VS CODE is =  streamlit run interface.py
# then go your local CMD run = ollama serve

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from Trading_Data import get_stock_data
from fuzzy_engine import compute_score

st.set_page_config(
    page_title="Group 5 — Trading DSS",
    layout="wide",
    page_icon="📈"
)


# Ddashboard-wide CSS 
st.markdown("""
<style>
.stApp { background-color: #0f1117; color: #e0e0e0; }
.block-container { padding-top: 1rem; }
[data-testid="stMetricValue"] { color: #00d4aa; font-size: 1.8rem; }
[data-testid="stMetricLabel"] { color: #6b7280; }
div[data-testid="column"] > div { background: #1a1d27; border-radius: 8px; padding: 12px; border: 0.5px solid #2a2d3a; }
.stTextInput > div > div > input { background: #12141e; color: #e0e0e0; border: 0.5px solid #2a2d3a; }
.stButton > button { background: #00d4aa; color: #0a1a14; font-weight: 700; border: none; }
.stButton > button:hover { background: #00b894; color: #0a1a14; }
h1, h2, h3 { color: #00d4aa !important; }
.stDivider { border-color: #2a2d3a; }
</style>
""", unsafe_allow_html=True)

st.title("Fuzzy Logic Adaptive Stock Trading Decision Support System")
st.caption("Group 5  |  FLC + Multi-Agent LLM Pipeline  |  S&P 500 Stocks")
st.divider()

# Two tabs= live analysis and historical backtesting
tab_analysis, tab_backtest = st.tabs(["Live Analysis", "Backtesting"])

#here is initialize the variable to store the data,logs and .......
if "df"         not in st.session_state: st.session_state.df         = None
if "ticker"     not in st.session_state: st.session_state.ticker     = None
if "logs"       not in st.session_state: st.session_state.logs       = ["Waiting for input..."]
if "flc_result" not in st.session_state: st.session_state.flc_result = None
if "llm_result" not in st.session_state: st.session_state.llm_result = None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str):
    return get_stock_data(ticker)



# display the live data and analysis result
with tab_analysis:

    col_main, col_log = st.columns([7, 3])

    with col_main:
        st.subheader("Analysis Output")
        m1, m2, m3, m4 = st.columns(4)

        df         = st.session_state.df
        flc_result = st.session_state.flc_result

        if df is not None:
            last  = df.iloc[-1]
            prev  = df.iloc[-2]
            delta = last["Close"] - prev["Close"]

            macd_cols = [c for c in df.columns if c.startswith("MACD_")]
            macd_val  = last[macd_cols[0]] if macd_cols else None

            m1.metric("Price",    f"${last['Close']:.2f}", f"{delta:+.2f}")
            m2.metric("RSI (14)", f"{last['RSI']:.1f}" if not pd.isna(last["RSI"]) else "—")
            m3.metric("MACD",     f"{macd_val:.4f}" if macd_val is not None and not pd.isna(macd_val) else "—")

            if flc_result:
                m4.metric("FLC Score", f"{flc_result['score']:.1f} / 5", flc_result["label"])
            else:
                m4.metric("FLC Score", "— / 5")
        else:
            m1.metric("Price", "—")
            m2.metric("RSI (14)", "—")
            m3.metric("MACD", "—")
            m4.metric("FLC Score", "— / 5")

        # Candlestick chart show pass 12 months of price history
        if df is not None:
            df_chart = df.tail(252)
            fig = go.Figure(data=[go.Candlestick(
                x=df_chart.index,
                open=df_chart["Open"],
                high=df_chart["High"],
                low=df_chart["Low"],
                close=df_chart["Close"],
                increasing_line_color="#00d4aa",
                decreasing_line_color="#ff4b4b",
            )])
            fig.update_layout(
                paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
                font_color="#e0e0e0",
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=40, b=10),
                height=400,
                title=f"{st.session_state.ticker} — Last 12 Months",
                title_font_color="#00d4aa",
            )
            fig.update_xaxes(gridcolor="#2a2d3a")
            fig.update_yaxes(gridcolor="#2a2d3a")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chart will appear here after analysis.")

        # FLC recommendation display
        st.subheader("Recommendation")
        if df is not None and flc_result:
            label = flc_result["label"]
            score = flc_result["score"]

            color_map = {
                "Strong Buy": "success", "Buy": "success",
                "Hold": "warning",
                "Sell": "error", "Strong Sell": "error",
            }
            getattr(st, color_map.get(label, "info"))(
                f"**FLC Signal: {label}** (Score {score}/5)  |  "
                f"RSI: `{flc_result['rsi_label']}`  |  "
                f"MACD: `{flc_result['macd_label']}`"
            )

            llm_result = st.session_state.llm_result
            if llm_result:
                st.markdown("**LLM Analysis (CrewAI — 4 Agents)**")
                st.markdown(llm_result)
            else:
                st.info("FLC score ready. Click **RUN AGENTS** for the full LLM recommendation.")
        else:
            st.warning("Run analysis to see recommendation.")

    with col_log:
        st.subheader("System Log")
        st.code("\n".join(st.session_state.logs), language=None)

    st.divider()

    # Control panel for input and buttons
    bot_left, bot_right = st.columns([7, 3])

    with bot_left:
        ticker = st.text_input(
            "Enter stock ticker", value="AAPL",
            placeholder="e.g. AAPL, MSFT, TSLA"
        )
        col_btn1, col_btn2 = st.columns(2)

        # Button 1: FLC only = claculate the FLC score based on the latest RSI and MACD values.
        #  This is fast and does not require Ollama running
        with col_btn1:
            if st.button("ANALYZE (FLC only)", type="primary"):
                sym  = ticker.strip().upper()
                logs = [f"[DATA]  Fetching OHLCV for {sym}..."]

                with st.spinner(f"Downloading {sym}..."):
                    df = fetch_data(sym)

                if df is None:
                    logs.append(f"[ERROR] No data for {sym}")
                    st.session_state.logs = logs
                    st.error(f"No data found for '{sym}'.")
                else:
                    last = df.iloc[-1]
                    logs.append(f"[DATA]  {len(df):,} rows ({df.index[0].date()} to {df.index[-1].date()})")
                    logs.append("[TECH]  RSI(14) and MACD(12,26,9) computed.")

                    rsi      = float(last["RSI"]) if not pd.isna(last["RSI"]) else 50.0
                    macd_col = next((c for c in df.columns if c.startswith("MACDh_")), None)
                    macd_h   = float(last[macd_col]) if macd_col and not pd.isna(last[macd_col]) else 0.0

                    logs.append(f"[FLC]   RSI={rsi:.2f}, MACD-H={macd_h:.4f}")
                    flc = compute_score(rsi, macd_h, sentiment=0.0)
                    logs.append(f"[FLC]   Score={flc['score']}/5  Signal={flc['label']}")
                    logs.append(f"[READY] FLC analysis complete — {sym}")

                    st.session_state.update({
                        "df": df, "ticker": sym,
                        "flc_result": flc, "llm_result": None, "logs": logs,
                    })
                    st.rerun()

        # Button 2: Full 4-agent pipeline = usig ollama run (first try only)
        # i still testing and thinking 
        # final version i plan 4 agent 分出来，可以看到各自的working log, 还有最后的综合结果
        with col_btn2:
            if st.button("RUN AGENTS (LLM)", type="secondary"):
                if st.session_state.df is None:
                    st.warning("Run FLC analysis first.")
                else:
                    sym  = st.session_state.ticker
                    logs = list(st.session_state.logs)
                    logs.append(f"[LLM]   Launching 4-agent CrewAI pipeline for {sym}...")
                    st.session_state.logs = logs

                    try:
                        from agents.crew import run_trading_crew
                        with st.spinner("Running AI agents... (this may take 30–90 seconds)"):
                            crew_result = run_trading_crew(sym)

                        prev_flc = st.session_state.flc_result or {}
                        st.session_state.flc_result = {
                            "score"      : crew_result["score"],
                            "label"      : crew_result["label"],
                            "rsi_label"  : prev_flc.get("rsi_label", "—"),
                            "macd_label" : prev_flc.get("macd_label", "—"),
                        }
                        st.session_state.llm_result = crew_result["recommendation"]
                        logs += crew_result["logs"]
                        logs.append(f"[SENT]  Sentiment={crew_result.get('sentiment', 0):+.2f}")
                        logs.append("[EXEC]  Recommendation generated.")
                        st.session_state.logs = logs
                        st.rerun()

                    except Exception as e:
                        logs.append(f"[ERROR] {e}")
                        st.session_state.logs = logs
                        if "connection" in str(e).lower() or "refused" in str(e).lower():
                            st.error("Cannot connect to Ollama. Make sure it is running: ollama serve")
                        else:
                            st.error(f"Agent error: {e}")

    with bot_right:
        st.subheader("Log Legend")
        st.markdown("""\
[DATA]  Data download & processing
[TECH]  Technical indicator computation
[FLC]   Fuzzy Logic Controller output
[SENT]  News sentiment score
[LLM]   LLM agent activity
[EXEC]  Final recommendation
[ERROR] Pipeline error
""")


# TAB 2: BACKTESTING (还没想好design,先放这里 ，这边都是testing try only)
# Runs the FLC strategy on historical data and compares against Buy & Hold
# The 70/30 train/test split ensures we measure performance（maybe can be adjusted）
with tab_backtest:

    st.subheader("Historical Backtest — FLC Strategy vs Buy & Hold")
    st.caption(
        "Uses 5 years of historical data. First 70% = calibration period. "
        "Last 30% = test period where all performance metrics are measured."
    )

    bc1, bc2, bc3 = st.columns([3, 2, 2])

    with bc1:
        bt_ticker = st.text_input(
            "Backtest ticker", value="AAPL",
            placeholder="e.g. AAPL, SPY, MSFT",
            key="bt_ticker"
        )
    with bc2:
        buy_thresh = st.slider(
            "Buy threshold", min_value=3.0, max_value=4.5,
            value=3.5, step=0.1,
            help="FLC score must be at or above this value to trigger a BUY signal"
        )
    with bc3:
        sell_thresh = st.slider(
            "Sell threshold", min_value=1.5, max_value=3.0,
            value=2.5, step=0.1,
            help="FLC score must be at or below this value to trigger a SELL signal"
        )

    if st.button("RUN BACKTEST", type="primary", key="run_bt"):
        sym = bt_ticker.strip().upper()

        with st.spinner(f"Running backtest for {sym}..."):
            try:
                import importlib, backtester
                importlib.reload(backtester)

                # Apply the slider values to the backtester module
                backtester.BUY_THRESHOLD  = buy_thresh
                backtester.SELL_THRESHOLD = sell_thresh
                backtester.TICKER         = sym

                train_df, test_df  = backtester.load_and_split(sym)
                signals_df         = backtester.generate_signals(test_df)
                result_df, trades  = backtester.simulate_portfolio(signals_df)
                metrics            = backtester.calculate_metrics(result_df, trades)

                # Metrics display
                st.divider()
                r1, r2, r3, r4, r5 = st.columns(5)

                r1.metric(
                    "FLC Total Return",
                    f"{metrics['total_return']:+.1f}%",
                    f"vs B&H {metrics['bh_return']:+.1f}%"
                )
                r2.metric("Win Rate",     f"{metrics['win_rate']:.1f}%")
                r3.metric("Total Trades", str(metrics["total_trades"]))
                r4.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
                r5.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")

                #Portfolio value chart (this idea thinking need or not?)
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=result_df.index, y=result_df["portfolio_value"],
                    name="FLC Strategy",
                    line=dict(color="#00d4aa", width=2),
                ))

                start_price = float(result_df["Close"].iloc[0])
                bh_shares   = int(backtester.INITIAL_CASH // start_price)
                bh_leftover = backtester.INITIAL_CASH - bh_shares * start_price
                bh_curve    = result_df["Close"] * bh_shares + bh_leftover

                fig.add_trace(go.Scatter(
                    x=result_df.index, y=bh_curve,
                    name="Buy & Hold",
                    line=dict(color="orange", width=2, dash="dash"),
                ))

                buy_pts  = result_df[result_df["action"] == "BUY"]
                sell_pts = result_df[result_df["action"] == "SELL"]

                fig.add_trace(go.Scatter(
                    x=buy_pts.index, y=buy_pts["portfolio_value"],
                    mode="markers", name="BUY",
                    marker=dict(symbol="triangle-up", color="lime", size=10),
                ))
                fig.add_trace(go.Scatter(
                    x=sell_pts.index, y=sell_pts["portfolio_value"],
                    mode="markers", name="SELL",
                    marker=dict(symbol="triangle-down", color="red", size=10),
                ))

                fig.update_layout(
                    paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
                    font_color="#e0e0e0", height=420,
                    title=f"{sym} — FLC ({metrics['total_return']:+.1f}%) vs B&H ({metrics['bh_return']:+.1f}%)",
                    title_font_color="#00d4aa",
                    legend=dict(bgcolor="#1a1d27"),
                    xaxis=dict(gridcolor="#2a2d3a"),
                    yaxis=dict(gridcolor="#2a2d3a", title="Portfolio Value (USD)"),
                )
                st.plotly_chart(fig, use_container_width=True)

                #FLC score chart
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=result_df.index, y=result_df["flc_score"],
                    name="FLC Score",
                    line=dict(color="#c084fc", width=1.5),
                ))
                fig2.add_hline(y=buy_thresh,  line_color="lime",    line_dash="dash",
                               annotation_text=f"Buy >= {buy_thresh}")
                fig2.add_hline(y=sell_thresh, line_color="red",     line_dash="dash",
                               annotation_text=f"Sell <= {sell_thresh}")
                fig2.add_hline(y=3.0,         line_color="#6b7280", line_dash="dot")
                fig2.update_layout(
                    paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
                    font_color="#e0e0e0", height=220,
                    title="FLC Score Over Time",
                    title_font_color="#00d4aa",
                    yaxis=dict(range=[1, 5], gridcolor="#2a2d3a"),
                    xaxis=dict(gridcolor="#2a2d3a"),
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Verdict
                diff = metrics["total_return"] - metrics["bh_return"]
                if diff > 0:
                    st.success(f"FLC OUTPERFORMED Buy & Hold by {diff:.2f}%")
                else:
                    st.error(
                        f"FLC underperformed Buy & Hold by {abs(diff):.2f}%.  "
                        f"Try adjusting the thresholds above and re-running."
                    )

            except Exception as e:
                st.error(f"Backtest error: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info(
            "Set the ticker and thresholds above, then click **RUN BACKTEST**. "
            "Results will be compared against a simple Buy & Hold benchmark."
        )
