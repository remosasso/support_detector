import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import threading
import uuid
from swing import start_swing_analysis

st.set_page_config(page_title='Live Swing Trading Dashboard', layout='wide')

# Detect running state from lock file
running = os.path.exists("swing_analysis.lock")


def try_read_csv(path, retries=3, delay=0.2):
    """Safely read CSV with retries"""
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
if "swing_analysis_started" not in st.session_state:
    st.session_state.swing_analysis_started = False
if "swing_progress" not in st.session_state:
    st.session_state.swing_progress = 0.0


def get_swing_progress():
    """Read progress from file"""
    if os.path.exists("swing_progress.txt"):
        try:
            with open("swing_progress.txt") as f:
                return float(f.read())
        except:
            return 0.0
    return 0.0


def set_swing_progress(progress: float):
    """Write progress to file"""
    with open("swing_progress.txt", "w") as f:
        f.write(str(progress))


# Sidebar
with st.sidebar:
    st.title('📈 Live Swing Trading Dashboard')
    st.markdown("""
    This dashboard identifies swing trading opportunities based on:
    - **RSI < 35** (4H) or **RSI < 30** (1D) (oversold)
    - **Price above MA200** (uptrend)
    - **Price below VWAP** (entry opportunity)
    - **MACD signals** (momentum)
    """)

    if running:
        progress = get_swing_progress()
        st.session_state.swing_analysis_started = True
        st.sidebar.markdown(f'🔄 Analysis in Progress {progress * 100:.1f}%')
        st.sidebar.progress(progress)
        st.sidebar.caption("Refreshing automatically...")
    else:
        st.sidebar.success("✅ No analysis running.")

    # Start Analysis Button
    if st.button("▶ (Re-)Start Swing Analysis", disabled=running):
        if os.path.exists("swing_analysis.lock"):
            st.warning("⚠ Analysis already running. Please wait or delete 'swing_analysis.lock' to force restart.")
        else:
            # Reset state
            st.session_state.swing_analysis_started = False
            st.session_state.swing_progress = 0.0
            set_swing_progress(0.0)

            # Clean old files
            for file in ["swing_results.csv", "swing_results_stable.csv"]:
                if os.path.exists(file):
                    os.remove(file)


            def run_swing_analysis():
                try:
                    start_swing_analysis(set_swing_progress)
                except Exception as e:
                    st.session_state.swing_analysis_started = False
                    st.error(f"Analysis failed: {e}")


            thread = threading.Thread(target=run_swing_analysis, daemon=True)
            thread.start()
            st.session_state.swing_analysis_started = True
            st.success("Started swing analysis thread. Dashboard will update as data comes in.")
            st.rerun()

# Main Dashboard
if os.path.exists("swing_results_stable.csv"):
    df = try_read_csv("swing_results_stable.csv")

    if not df.empty:
        # Sort by swing score
        sorted_df = df.sort_values(by="Swing_Score", ascending=False)

        # Determine if we have multi-timeframe data
        is_multi_timeframe = 'RSI_4H' in sorted_df.columns

        # Main title and metrics
        st.title("📈 Swing Trading Opportunities")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Opportunities", len(sorted_df))
        with col2:
            excellent_count = len(sorted_df[sorted_df['Entry_Quality'] == 'Excellent'])
            st.metric("Excellent Entries", excellent_count)
        with col3:
            if is_multi_timeframe:
                avg_rsi_4h = sorted_df['RSI_4H'].mean()
                st.metric("Avg RSI (4H)", f"{avg_rsi_4h:.1f}")
            else:
                avg_rsi = sorted_df['RSI'].mean()
                st.metric("Avg RSI", f"{avg_rsi:.1f}")
        with col4:
            if is_multi_timeframe:
                macd_crosses = len(sorted_df[sorted_df['MACD_Cross_4H'] == True])
                st.metric("MACD Crosses (4H)", macd_crosses)
            else:
                macd_crosses = len(sorted_df[sorted_df['MACD_Cross_Up'] == True])
                st.metric("MACD Crosses", macd_crosses)

        # Results table
        st.subheader("🎯 Swing Trading Candidates")

        # Format the dataframe for better display
        display_df = sorted_df.copy()

        # Define formatting functions for different data types
        if is_multi_timeframe:
            numeric_columns = {
                'Close': lambda x: f"${x:.2f}",
                'RSI_4H': lambda x: f"{x:.1f}",
                'RSI_1D': lambda x: f"{x:.1f}" if x > 0 else "N/A",
                'MACD_4H': lambda x: f"{x:.4f}",
                'MACD_Signal_4H': lambda x: f"{x:.4f}",
                'MACD_1D': lambda x: f"{x:.4f}" if x != 0 else "N/A",
                'MACD_Signal_1D': lambda x: f"{x:.4f}" if x != 0 else "N/A",
                'VWAP_4H': lambda x: f"${x:.2f}",
                'VWAP_1D': lambda x: f"${x:.2f}" if x > 0 else "N/A",
                'MA200': lambda x: f"${x:.2f}",
                'Support_20_4H': lambda x: f"${x:.2f}",
                'Support_20_1D': lambda x: f"${x:.2f}",
                'Volume_Ratio_4H': lambda x: f"{x:.1f}",
                'Volume_Ratio_1D': lambda x: f"{x:.1f}" if x > 0 else "N/A",
                'Swing_Score': lambda x: f"{x:.1f}"
            }
        else:
            numeric_columns = {
                'Close': lambda x: f"${x:.2f}",
                'RSI': lambda x: f"{x:.1f}",
                'MACD': lambda x: f"{x:.4f}",
                'MACD_Signal': lambda x: f"{x:.4f}",
                'VWAP': lambda x: f"${x:.2f}",
                'MA200': lambda x: f"${x:.2f}",
                'Support_20_4H': lambda x: f"${x:.2f}",
                'Support_20_1D': lambda x: f"${x:.2f}",
                'Volume_Ratio': lambda x: f"{x:.1f}",
                'Swing_Score': lambda x: f"{x:.1f}"
            }

        # Apply formatting only to columns that exist
        for col, formatter in numeric_columns.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(formatter)


        # Color coding for entry quality
        def color_entry_quality(val):
            # Since our font is white we should use dark color variant of the below

            if val == 'Excellent':
                return 'background-color: #006400'
            elif val == 'Good':
                return 'background-color: #FFD700'
            elif val == 'Fair':
                return 'background-color: #FF8C00'
            else:
                return 'background-color: #B22222'

        # select columns to display
        display_columns = [
            'Ticker', 'Close', 'MA200',
            'Support_20_1D', 'RSI_4H', 'RSI_1D',
            'MACD_Cross_4H', 'MACD_Cross_1D',
            'Swing_Score', 'Entry_Quality'
        ]
        display_columns = [col for col in display_columns if col in display_df.columns]
        display_df = display_df[display_columns]


        styled_df = display_df.style.applymap(color_entry_quality, subset=['Entry_Quality'])
        st.dataframe(styled_df, use_container_width=True)

        # Charts section
        st.subheader("📊 Analysis Charts")


        # Individual stock analysis
        ticker = st.selectbox("Select Ticker for Detailed View", sorted_df["Ticker"].unique(), index=0)

        # Time range and interval selection
        col1, col2 = st.columns(2)
        with col1:
            interval = st.selectbox("Interval", ["4H", "1D"], index=0)
        with col2:
            if interval == "4H":
                time_range = st.selectbox("Time Range", ["6M", "3M", "1M"], index=0)
            else:
                time_range = st.selectbox("Time Range", ["5Y", "2Y", "1Y", "6M", "3M", "1M"], index=0)

        # Load chart data
        chart_file = f"swing_chart_data/{ticker}_{interval.lower()}.parquet"

        if os.path.exists(chart_file):
            try:
                chart_data = pd.read_parquet(chart_file)

                # Handle MultiIndex columns
                if isinstance(chart_data.columns, pd.MultiIndex):
                    chart_data.columns = chart_data.columns.get_level_values(0)

                # Handle datetime index
                if pd.api.types.is_datetime64_any_dtype(chart_data.index):
                    if chart_data.index.tz is None:
                        chart_data["Date"] = pd.to_datetime(chart_data.index).tz_localize("UTC")
                    else:
                        chart_data["Date"] = pd.to_datetime(chart_data.index)
                else:
                    chart_data["Date"] = pd.to_datetime(chart_data.index).tz_localize("UTC")

                # Filter by time range
                if interval == "4H":
                    range_days = {"6M": 180, "3M": 90, "1M": 30}
                else:
                    range_days = {"5Y": 1825 ,"2Y": 730, "1Y": 365, "6M": 180, "3M": 90, "1M": 30}

                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=range_days[time_range])
                chart_data = chart_data[chart_data["Date"] >= cutoff]

                # Create subplots
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(
                    f'{ticker} Price & Indicators ({interval})', f'RSI ({interval})', f'MACD ({interval})'),
                    row_heights=[0.6, 0.2, 0.2]
                )

                # Candlestick chart - handle both single and multi-index columns
                price_columns = {}
                for price_type in ['Open', 'High', 'Low', 'Close']:
                    # Look for exact match first, then partial match
                    if price_type in chart_data.columns:
                        price_columns[price_type] = price_type
                    else:
                        matching_cols = [col for col in chart_data.columns if price_type in str(col)]
                        if matching_cols:
                            price_columns[price_type] = matching_cols[0]

                if len(price_columns) == 4:
                    fig.add_trace(
                        go.Candlestick(
                            x=chart_data['Date'],
                            open=chart_data[price_columns['Open']],
                            high=chart_data[price_columns['High']],
                            low=chart_data[price_columns['Low']],
                            close=chart_data[price_columns['Close']],
                            name="Price"
                        ), row=1, col=1
                    )

                # Add moving averages and indicators based on interval
                if interval == "4H":
                    # 4H specific indicators
                    if 'MA200' in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['MA200'],
                                name="MA200",
                                line=dict(color='blue', width=2)
                            ), row=1, col=1
                        )

                    if 'VWAP_4H' in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['VWAP_4H'],
                                name="VWAP (4H)",
                                line=dict(color='orange', width=2)
                            ), row=1, col=1
                        )
                    elif 'VWAP' in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['VWAP'],
                                name="VWAP",
                                line=dict(color='orange', width=2)
                            ), row=1, col=1
                        )

                    # RSI for 4H
                    rsi_col = 'RSI_4H' if 'RSI_4H' in chart_data.columns else 'RSI'
                    if rsi_col in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data[rsi_col],
                                name="RSI (4H)",
                                line=dict(color='purple')
                            ), row=2, col=1
                        )
                        # RSI levels for 4H
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=35, line_dash="dash", line_color="orange", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                    # MACD for 4H
                    macd_col = 'MACD_4H' if 'MACD_4H' in chart_data.columns else 'MACD'
                    signal_col = 'MACD_signal_4H' if 'MACD_signal_4H' in chart_data.columns else 'MACD_signal'

                    if macd_col in chart_data.columns and signal_col in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data[macd_col],
                                name="MACD (4H)",
                                line=dict(color='blue')
                            ), row=3, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data[signal_col],
                                name="Signal (4H)",
                                line=dict(color='red')
                            ), row=3, col=1
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

                    # Add support levels
                    if 'Support_20_4H' in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['Support_20_4H'],
                                name="Support",
                                line=dict(color='green', dash='dash', width=1)
                            ), row=1, col=1
                        )

                else:  # 1D interval
                    # 1D specific indicators
                    if 'MA200' in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['MA200'],
                                name="MA200",
                                line=dict(color='blue', width=2)
                            ), row=1, col=1
                        )

                    if 'VWAP_1D' in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['VWAP_1D'],
                                name="VWAP (1D)",
                                line=dict(color='orange', width=2)
                            ), row=1, col=1
                        )
                    elif 'VWAP' in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['VWAP'],
                                name="VWAP",
                                line=dict(color='orange', width=2)
                            ), row=1, col=1
                        )

                    # RSI for 1D
                    rsi_col = 'RSI_1D' if 'RSI_1D' in chart_data.columns else 'RSI'
                    if rsi_col in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data[rsi_col],
                                name="RSI (1D)",
                                line=dict(color='purple')
                            ), row=2, col=1
                        )
                        # RSI levels for 1D
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                    # MACD for 1D
                    macd_col = 'MACD_1D' if 'MACD_1D' in chart_data.columns else 'MACD'
                    signal_col = 'MACD_signal_1D' if 'MACD_signal_1D' in chart_data.columns else 'MACD_signal'

                    if macd_col in chart_data.columns and signal_col in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data[macd_col],
                                name="MACD (1D)",
                                line=dict(color='blue')
                            ), row=3, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data[signal_col],
                                name="Signal (1D)",
                                line=dict(color='red')
                            ), row=3, col=1
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

                # Add support levels
                if 'Support_20_1D' in chart_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data['Date'],
                            y=chart_data['Support_20_1D'],
                            name="Support",
                            line=dict(color='green', dash='dash', width=1)
                        ), row=1, col=1
                    )

                fig.update_layout(
                    height=800,
                    title=f"{ticker} Swing Trading Analysis ({interval} Timeframe)",
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display current metrics for selected stock
                stock_data = sorted_df[sorted_df['Ticker'] == ticker].iloc[0]

                if is_multi_timeframe:
                    # Multi-timeframe metrics display
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${stock_data['Close']}")
                        st.metric("RSI (4H)", f"{stock_data['RSI_4H']:.1f}")
                        if stock_data['RSI_1D'] > 0:
                            st.metric("RSI (1D)", f"{stock_data['RSI_1D']:.1f}")
                    with col2:
                        st.metric("VWAP (4H)", f"${stock_data['VWAP_4H']:.2f}")
                        if stock_data.get('VWAP_1D', 0) > 0:
                            st.metric("VWAP (1D)", f"${stock_data['VWAP_1D']:.2f}")
                        st.metric("MA200", f"${stock_data['MA200']:.2f}")
                    with col3:
                        st.metric("Support", f"${stock_data['Support_20_4H']:.2f}")
                        st.metric("Volume Ratio (4H)", f"{stock_data['Volume_Ratio_4H']:.1f}x")
                        if stock_data.get('Volume_Ratio_1D', 0) > 0:
                            st.metric("Volume Ratio (1D)", f"{stock_data['Volume_Ratio_1D']:.1f}x")
                    with col4:
                        st.metric("Swing Score", f"{stock_data['Swing_Score']:.1f}")
                        st.metric("Entry Quality", stock_data['Entry_Quality'])
                        if 'Timeframe_Alignment' in stock_data.index:
                            st.metric("TF Alignment", stock_data['Timeframe_Alignment'])
                else:
                    # Original single timeframe metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${stock_data['Close']}")
                        st.metric("RSI", f"{stock_data['RSI']:.1f}")
                    with col2:
                        st.metric("VWAP", f"${stock_data['VWAP']:.2f}")
                        st.metric("MA200", f"${stock_data['MA200']:.2f}")
                    with col3:
                        st.metric("Support", f"${stock_data['Support_20_4H']:.2f}")
                        st.metric("Volume Ratio", f"{stock_data['Volume_Ratio']:.1f}x")
                    with col4:
                        st.metric("Swing Score", f"{stock_data['Swing_Score']:.1f}")
                        st.metric("Entry Quality", stock_data['Entry_Quality'])

            except Exception as e:
                st.error(f"Failed to render chart: {e}")
                # Debug info
                if os.path.exists(chart_file):
                    try:
                        debug_data = pd.read_parquet(chart_file)
                        st.write(f"**Debug:** Chart file shape: {debug_data.shape}")
                        st.write(f"**Debug:** Available columns: {list(debug_data.columns)}")
                    except Exception as debug_e:
                        st.error(f"Debug error: {debug_e}")
        else:
            st.warning(f"No chart data found for {ticker} at {interval} interval")
            # Show available files
            if os.path.exists("swing_chart_data"):
                available_files = [f for f in os.listdir("swing_chart_data") if ticker in f]
                if available_files:
                    st.info(f"Available files for {ticker}: {available_files}")


    else:
        st.warning("📄 Results file found, but no swing opportunities detected yet.")

else:
    st.info("🕒 Awaiting swing analysis results...")

# Auto-refresh logic
if st.session_state.swing_analysis_started:
    progress = get_swing_progress()
    time.sleep(1)
    st.rerun()
elif os.path.exists("swing_progress.txt") and get_swing_progress() >= 1.0:
    st.sidebar.success("✅ Swing analysis complete!")