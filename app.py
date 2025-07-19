import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import threading
import uuid
import yfinance as yf
from support import start_analysis

st.set_page_config(page_title='Live Stock Support Dashboard', layout='wide')
# Detect running state from lock file, not session_state
running = os.path.exists("analysis.lock")
def try_read_csv(path, retries=3, delay=0.2):
    for _ in range(retries):
        try:
            df = pd.read_csv(path)
            if not df.empty:
                return df
        except pd.errors.EmptyDataError:
            time.sleep(delay)
        except Exception as e:
            st.warning(f"⚠ Error reading {path}: {e}")
            time.sleep(delay)
    return pd.DataFrame()
# Initialize state
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False
if "progress" not in st.session_state:
    st.session_state.progress = 0.0
def get_progress_file():
    if os.path.exists("progress.txt"):
        with open("progress.txt") as f:
            try:
                return float(f.read())
            except:
                return 0.0
    return 0.0


def set_progress_file(progress: float):
    with open("progress.txt", "w") as f:
        f.write(str(progress))


with st.sidebar:

    st.title('📊 Live Stock Support Level Analysis')
    st.markdown("""
    This dashboard displays stocks near major support levels based on RSI and proximity to support.
    The analysis runs in the background and updates the dashboard as new data comes in.
    """)
    if running:
        progress = get_progress_file()
        st.session_state.analysis_started = True
        st.sidebar.markdown(f'🔄 Analysis in Progress {progress * 100:.1f}%')
        st.sidebar.progress(progress)
        st.sidebar.caption("Refreshing automatically...")
    else:
        st.sidebar.success("✅ No analysis running.")

    # Display live progress
    if st.button("▶ (Re-)Start Analysis in Background", disabled=running):
        if os.path.exists("analysis.lock"):
            st.warning("⚠ Analysis already running. Please wait or delete 'analysis.lock' to force restart.")
        else:
            st.session_state.analysis_started = False
            st.session_state.progress = 0.0
            set_progress_file(0.0)
            if os.path.exists("results.csv"):
                os.remove("results.csv")
            if os.path.exists("results_stable.txt"):
                os.remove("results_stable.csv")


            def run_analysis():
                try:
                    start_analysis(set_progress_file)
                except Exception as e:
                    st.session_state.analysis_started = False
                    st.error(f"Analysis failed: {e}")


            thread = threading.Thread(target=run_analysis, daemon=True)
            thread.start()
            st.session_state.analysis_started = True
            st.success("Started analysis thread. Dashboard will update as data comes in.")
            st.rerun()

    # if st.button("Hard reset"):
    #     os.remove("analysis.lock")


# Display updating dashboard
placeholder = st.empty()
REFRESH_INTERVAL = 10  # seconds

if os.path.exists("results_stable.csv"):
    df = try_read_csv("results_stable.csv")
    if not df.empty:

        unique_key = str(uuid.uuid4())
        # Input for individual ticker
        sorted_df = df.sort_values(by="Overall Score", ascending=False)
        st.subheader("📈 Stocks Near Major Support (Live)")
        st.dataframe(
            sorted_df[
                ["Ticker", "Current Price", "Support Level", "Proximity %", "RSI", "Market Cap", "Drop %", "Technical Score", "Fundamental Score", "Overall Score"]],
            use_container_width=True
        )

        ## dropdown for tickers
        ticker = st.selectbox("Select Ticker", sorted_df["Ticker"].unique(), index=0, key=f"live_tick_").upper()
        # Time range
        range_option = st.selectbox("Select Time Range", ["1Y", "6M", "3M", "1M"], key=f"live_range_")
        period_map_days = {
            "1Y": 252,
            "6M": 126,
            "3M": 63,
            "1M": 21,
        }
        period_days = period_map_days[range_option]
        interval_option = "1D"
        # Valid intervals based on range
        if range_option in ["1M", "3M"]:
            interval_option = st.selectbox("Select Interval", ["1D", "4H"], key=f"live_interval_")
        else:
            interval_option = "1D"  # Force 1D if too long
            st.info("📏 4H candles only available for short time ranges (≤3M)")
            chart_path = f"chart_data/{ticker}_1d.parquet"
        chart_path = f"chart_data/{ticker}_1d.parquet" if interval_option == "1D" else f"chart_data/{ticker}_4h.parquet"

        if os.path.exists(chart_path):
            try:


                chart_data = pd.read_parquet(chart_path)
                # Flatten MultiIndex if needed
                # Flatten MultiIndex if needed
                if isinstance(chart_data.columns, pd.MultiIndex):
                    chart_data.columns = chart_data.columns.get_level_values(0)

                # Handle datetime
                if "Date" in chart_data.columns:
                    chart_data["Date"] = pd.to_datetime(chart_data["Date"])
                elif "Datetime" in chart_data.columns:
                    chart_data["Date"] = pd.to_datetime(chart_data["Datetime"])
                elif pd.api.types.is_datetime64_any_dtype(chart_data.index):
                    # Localize only if tz-naive
                    if chart_data.index.tz is None:
                        chart_data["Date"] = pd.to_datetime(chart_data.index).tz_localize("UTC")
                    else:
                        chart_data["Date"] = pd.to_datetime(chart_data.index)
                else:
                    st.error("No valid datetime column or index found.")
                    st.stop()

                # Filter by range
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=period_days)
                chart_data = chart_data[chart_data["Date"] >= cutoff]

                # Step 5: Plot
                fig = go.Figure(data=[go.Candlestick(
                    x=chart_data['Date'],
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red',


                )])
                fig.update_layout(
                    title=f"{ticker} – {range_option} Candlestick Chart ({interval_option})",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    height=600
                )
                fig.update_xaxes(type="category")
                tick_indices = chart_data.index[::7]  # show every 10th bar (adjust if too few/many)
                fig.update_layout(
                    xaxis=dict(
                        tickmode="array",
                        tickvals=chart_data.loc[tick_indices, "Date"],
                        ticktext=chart_data.loc[tick_indices, "Date"].dt.strftime("%b %d"),
                        tickangle=0,
                    )
                )
                support_y = df[df['Ticker'] == ticker]['Support Level'].values[0]

                fig.add_trace(go.Scatter(
                    x=chart_data['Date'],  # categorical x-values still work here
                    y=[support_y] * len(chart_data),
                    mode="lines",
                    line=dict(color="blue", dash="dash", width=2),
                    name="Support Level",
                    hoverinfo="skip",
                    showlegend=False
                ))
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker}_")

            except Exception as e:
                st.error(f"Failed to render chart: {e}")
        else:
            st.warning(f"No OHLC data found at {chart_path}")

        fig2 = px.bar(
            sorted_df.head(20),
            x="Ticker", y="Overall Score",
            color="RSI",
            title="Top 20 Bounce Candidates",
            hover_data=["Support Level", "Proximity %"]
        )
        st.plotly_chart(fig2, use_container_width=True, key=f"live_top_")


    else:
        st.warning("📄 results.csv found, but no data yet. Waiting...")
else:
    st.info("🕒 Awaiting results.csv from background analysis...")

if st.session_state.analysis_started:
    progress = get_progress_file()

    time.sleep(1)
    st.rerun()

elif os.path.exists("progress.txt") and get_progress_file() >= 1.0:
    st.sidebar.success("✅ Analysis complete!")