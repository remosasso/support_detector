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
from gpt_swing import analyze_longterm_support_dips

st.set_page_config(page_title='Live Swing Trading Dashboard', layout='wide')

# --- CONFIGURATION ---
RESULTS_FILE = "swing_results_stable.csv"
LOCK_FILE = "swing_analysis.lock"

# --- HELPER FUNCTIONS ---
def try_read_csv(path, retries=5, delay=0.3):
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

if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = os.path.exists(LOCK_FILE)

if os.path.exists("swing_done.txt"):
    st.session_state.analysis_running = False
    os.remove("swing_done.txt")

if os.path.exists("swing_error.txt"):
    with open("swing_error.txt") as f:
        st.error("Swing analysis failed: " + f.read())
    os.remove("swing_error.txt")

def set_swing_progress(progress: float):
    with open("swing_progress.txt", "w") as f:
        f.write(str(progress))

# --- SIDEBAR ---
with st.sidebar:
    st.title('📈 Ranked Swing Screener')
    st.markdown("""
    This dashboard identifies sharp drops based on:
    - **RSI < 30**
    - **Close near 180-day support**
    - **5-day price drop > 5%**
    """)

    st.session_state.analysis_running = os.path.exists(LOCK_FILE)
    time.sleep(0.5)  # Ensure file copy is finished before rerun

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
            try:
                analyze_longterm_support_dips(set_swing_progress)  # this function should NOT use st.*
            except Exception as e:
                with open("swing_error.txt", "w") as f:
                    f.write(str(e))
            finally:
                if os.path.exists(LOCK_FILE):
                    os.remove(LOCK_FILE)
                # Do NOT touch st.session_state here
                with open("swing_done.txt", "w") as f:
                    f.write("done")


        thread = threading.Thread(target=run_analysis_thread, daemon=True)
        thread.start()
        st.session_state.analysis_running = True
        st.rerun()

# --- MAIN DASHBOARD ---
st.title("🎯 Sharp Drop Swing Candidates")

if os.path.exists(RESULTS_FILE):
    df = try_read_csv(RESULTS_FILE)

    if not df.empty:
        col1, col2 = st.columns(2)
        tier_counts = df['RSI_Tier'].value_counts()
        col1.metric("Total Candidates", len(df))
        col2.metric("Tier A (Extreme RSI < 20)", tier_counts.get("A (Extreme)", 0))

        st.markdown("---")
        st.subheader("Ranked Opportunities")

        display_df = df.copy()
        display_cols = ['RSI_Tier', 'Ticker', 'Close', 'RSI', '5D_Change_%', 'Dist_to_LongSupport_%', 'MarketCap', 'MarketCapTier']
        display_df = display_df[display_cols]

        def color_tier(tier):
            tier = str(tier)
            if "A" in tier: return 'background-color: #006400; color: white'
            if "B" in tier: return 'background-color: #00008B; color: white'
            if "C" in tier: return 'background-color: #FF8C00; color: black'
            return ''


        styled_df = display_df.style \
            .format({
            'RSI': '{:.2f}',
            '5D_Change_%': '{:.2f}%',
            'Close': '${:.2f}',
            'Dist_to_LongSupport_%': '{:.2f}',
            'MarketCap': '{}'  # Already formatted like '$30.3B'
        }) \
            .applymap(color_tier, subset=['RSI_Tier'])  # Keep your tier coloring

        st.dataframe(styled_df, use_container_width=True, height=min(500, 35 * len(df) + 38))
        st.markdown("---")

        st.subheader("📊 Detailed Ticker Analysis")
        ticker = st.selectbox("Select Ticker for Detailed View", df["Ticker"].unique())

        if ticker:
            col1, col2 = st.columns(2)
            with col1:
                interval = st.selectbox("Select Chart Interval", ['1d', '4h', '1h'], index=0)
            with col2:
                valid_periods = ['6mo', '1y', '2y', '5y'] if interval == '1d' else ['1mo', '3mo', '6mo']
                default_index = 1
                period = st.selectbox("Select Chart Period", valid_periods, index=default_index)

            st.info(f"Fetching **{period}** of **{interval}** data for **{ticker}**...")
            try:
                tick_data = yf.Ticker(ticker)
                hist_df = tick_data.history(period=period, interval=interval, auto_adjust=True)

                if hist_df.empty:
                    st.warning(f"Could not fetch historical data for {ticker}.")
                else:
                    hist_df['MA50'] = hist_df['Close'].rolling(window=50).mean()
                    hist_df['MA200'] = hist_df['Close'].rolling(window=200).mean()
                    hist_df['RSI'] = ta.momentum.RSIIndicator(close=hist_df['Close'], window=14).rsi()
                    macd = ta.trend.MACD(close=hist_df['Close'])
                    hist_df['MACD'] = macd.macd()
                    hist_df['MACD_Signal'] = macd.macd_signal()

                    interval_name = {'1d': 'Daily', '4h': '4-Hourly', '1h': 'Hourly'}.get(interval, interval)
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                        subplot_titles=(f'{ticker} {interval_name} Price Action',
                                                        f'RSI ({interval})', f'MACD ({interval})'),
                                        row_heights=[0.6, 0.2, 0.2])

                    fig.add_trace(go.Candlestick(x=hist_df.index, open=hist_df['Open'], high=hist_df['High'],
                                                 low=hist_df['Low'], close=hist_df['Close'], name='Price'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MA50'], name='50-Period MA',
                                             line=dict(color='orange')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MA200'], name='200-Period MA',
                                             line=dict(color='blue')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['RSI'], name='RSI',
                                             line=dict(color='purple')), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MACD'], name='MACD',
                                             line=dict(color='cyan')), row=3, col=1)
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

if st.session_state.analysis_running:
    progress = get_swing_progress()
    st.rerun()
elif os.path.exists("swing_progress.txt") and get_swing_progress() >= 1.0:
    st.sidebar.success("✅ Swing analysis complete!")

