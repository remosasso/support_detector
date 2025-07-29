import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import threading
import yfinance as yf
import ta

# Import the analysis function from your swing script
from claude_swing import start_swing_analysis
st.set_page_config(page_title='Live Swing Trading Dashboard', layout='wide')

# --- CONFIGURATION ---
RESULTS_FILE = "swing_results_stable.csv"
LOCK_FILE = "swing_analysis.lock"


# --- HELPER FUNCTIONS ---

def try_read_csv(path, retries=5, delay=0.3):
    """Safely read CSV with retries to handle file-writing conflicts."""
    for _ in range(retries):
        try:
            return pd.read_csv(path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            time.sleep(delay)
        except Exception as e:
            st.warning(f"⚠ Error reading {path}: {e}")
            time.sleep(delay)
    return pd.DataFrame()

def get_swing_progress():
    if os.path.exists("swing_progress.txt"):
        try:
            with open("swing_progress.txt") as f:
                return float(f.read())
        except:
            return 0.0
    return 0.0

# Initialize session state for running analysis
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = os.path.exists(LOCK_FILE)


def set_swing_progress(progress: float):
    with open("swing_progress.txt", "w") as f:
        f.write(str(progress))

# --- SIDEBAR ---
with st.sidebar:
    st.title('📈 Ranked Swing Screener')
    st.markdown("""
    This dashboard ranks stocks based on a weighted score across four pillars:
    - **Quality:** Fundamental strength (margins, debt, ROE).
    - **Trend:** Long-term uptrend (Price vs. MAs).
    - **Pullback:** Oversold condition (RSI, nearness to support).
    - **Risk:** Volatility and liquidity (ATR, Beta, Volume).
    """)

    # --- Analysis Control ---
    st.session_state.analysis_running = os.path.exists(LOCK_FILE)

    if st.session_state.analysis_running:
        progress = get_swing_progress()
        st.info('🔄 Analysis in Progress...')
        st.progress(progress, text=f"{progress * 100:.1f}% Complete")
        st.caption("Dashboard will refresh automatically.")
    else:
        st.success("✅ Analysis complete or not running.")

    if st.button("▶ (Re-)Start Full Analysis", disabled=st.session_state.analysis_running, type="primary"):
        st.session_state.swing_progress = 0.0


        def run_analysis_thread():
            """Wrapper to run the analysis in a separate thread."""
            try:
                start_swing_analysis(set_swing_progress)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
            finally:
                if os.path.exists(LOCK_FILE):
                    os.remove(LOCK_FILE)
                st.session_state.analysis_running = False


        thread = threading.Thread(target=run_analysis_thread, daemon=True)
        thread.start()
        st.session_state.analysis_running = True
        st.rerun()

# --- MAIN DASHBOARD ---
st.title("🎯 Prime Swing Trading Candidates")

if os.path.exists(RESULTS_FILE):
    df = try_read_csv(RESULTS_FILE)

    if not df.empty:
        # Main Metrics
        col1, col2, col3, col4 = st.columns(4)
        prime_candidates = df[df['Tier'] == 'A (Prime)']

        col1.metric("Total Candidates Found", len(df))
        col2.metric("Prime Candidates (Tier A)", len(prime_candidates))
        col3.metric("Avg. R/R Ratio (Prime)",
                    f"{prime_candidates['rr_ratio'].mean():.2f} R" if not prime_candidates.empty else "N/A")
        col4.metric("Avg. ATR % (Prime)",
                    f"{prime_candidates['atr_pct'].mean():.2f}%" if not prime_candidates.empty else "N/A")

        st.markdown("---")

        # --- Display Results Table ---
        st.subheader("Ranked Opportunities")


        def color_tier(tier):
            if "A (Prime)" in tier: return 'background-color: #006400; color: white'
            if "B (Watch)" in tier: return 'background-color: #00008B; color: white'
            if "C (Speculative)" in tier: return 'background-color: #FF8C00; color: black'
            return ''


        display_df = df.copy()
        display_cols = ['Tier', 'Ticker', 'swing_score', 'Close', 'rr_ratio', 'atr_pct', 'rsi_4h', 'market_cap', 'beta',
                        'quality_rank', 'trend_rank', 'pullback_rank', 'risk_rank']
        display_cols = [col for col in display_cols if col in display_df.columns]
        display_df = display_df[display_cols]

        styled_df = display_df.style.applymap(color_tier, subset=['Tier']) \
            .format({'swing_score': '{:.1f}', 'Close': '${:.2f}', 'rr_ratio': '{:.2f} R', 'atr_pct': '{:.2f}%',
                     'rsi_4h': '{:.1f}', 'beta': '{:.2f}', 'quality_rank': '{:.1f}', 'trend_rank': '{:.1f}',
                     'pullback_rank': '{:.1f}', 'risk_rank': '{:.1f}'})

        st.dataframe(styled_df, use_container_width=True, height=min(500, 35 * len(df) + 38))
        st.markdown("---")

        # --- Detailed Chart Section ---
        st.subheader("📊 Detailed Ticker Analysis")

        ticker = st.selectbox("Select Ticker for Detailed View", df["Ticker"].unique())

        if ticker:
            # --- NEW: CHART TIMEFRAME SELECTION ---
            col1, col2 = st.columns(2)
            with col1:
                interval = st.selectbox(
                    "Select Chart Interval",
                    ['1d', '4h', '1h'],
                    index=0,
                    help="Select the time interval for each candlestick."
                )
            with col2:
                # Define valid periods for each interval to prevent errors
                if interval == '1d':
                    valid_periods = ['6mo', '1y', '2y', '5y', 'max']
                    default_index = 1  # '1y'
                elif interval == '4h':
                    # yfinance has limitations on 4h data, max is 2 years
                    valid_periods = ['1mo', '3mo', '6mo', '1y', '2y']
                    default_index = 2  # '6mo'
                else:  # 1h
                    # yfinance has limitations on 1h data, max is 2 years
                    valid_periods = ['5d', '1mo', '3mo', '6mo', '1y', '2y']
                    default_index = 1  # '1mo'

                period = st.selectbox(
                    "Select Chart Period",
                    valid_periods,
                    index=default_index,
                    help="Select the historical time range to display."
                )

            st.info(f"Fetching **{period}** of **{interval}** data for **{ticker}**...")
            try:
                tick_data = yf.Ticker(ticker)
                hist_df = tick_data.history(period=period, interval=interval, auto_adjust=True)

                if hist_df.empty:
                    st.warning(
                        f"Could not fetch historical data for {ticker} with interval {interval} and period {period}.")
                else:
                    # Calculate indicators on the fly
                    hist_df['MA50'] = hist_df['Close'].rolling(window=50).mean()
                    hist_df['MA200'] = hist_df['Close'].rolling(window=200).mean()
                    hist_df['RSI'] = ta.momentum.RSIIndicator(close=hist_df['Close'], window=14).rsi()
                    macd = ta.trend.MACD(close=hist_df['Close'])
                    hist_df['MACD'] = macd.macd()
                    hist_df['MACD_Signal'] = macd.macd_signal()

                    interval_name = {'1d': 'Daily', '4h': '4-Hourly', '1h': 'Hourly'}.get(interval, interval)

                    fig = make_subplots(
                        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=(
                        f'{ticker} {interval_name} Price Action', f'RSI ({interval})', f'MACD ({interval})'),
                        row_heights=[0.6, 0.2, 0.2]
                    )

                    fig.add_trace(
                        go.Candlestick(x=hist_df.index, open=hist_df['Open'], high=hist_df['High'], low=hist_df['Low'],
                                       close=hist_df['Close'], name='Price'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MA50'], name='50-Period MA',
                                             line=dict(color='orange', width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MA200'], name='200-Period MA',
                                             line=dict(color='blue', width=2)), row=1, col=1)

                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['RSI'], name='RSI', line=dict(color='purple')),
                                  row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)

                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MACD'], name='MACD', line=dict(color='cyan')),
                                  row=3, col=1)
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MACD_Signal'], name='Signal',
                                             line=dict(color='magenta')), row=3, col=1)
                    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)

                    fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                    fig.update_yaxes(title_text="RSI", row=2, col=1)
                    fig.update_yaxes(title_text="MACD", row=3, col=1)

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred while creating the chart for {ticker}: {e}")

    else:
        st.warning("Results file is empty. Either no opportunities were found or the analysis hasn't completed.")
else:
    st.info("Awaiting first analysis run. Click the button in the sidebar to begin.")

# --- AUTO-REFRESH LOGIC ---
if st.session_state.analysis_running:
    progress = get_swing_progress()
    time.sleep(1)
    st.rerun()
elif os.path.exists("swing_progress.txt") and get_swing_progress() >= 1.0:
    st.sidebar.success("✅ Swing analysis complete!")