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
    - **RSI < 30** (oversold)
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

        # Main title and metrics
        st.title("📈 Swing Trading Opportunities")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Opportunities", len(sorted_df))
        with col2:
            excellent_count = len(sorted_df[sorted_df['Entry_Quality'] == 'Excellent'])
            st.metric("Excellent Entries", excellent_count)
        with col3:
            avg_rsi = sorted_df['RSI'].mean()
            st.metric("Avg RSI", f"{avg_rsi:.1f}")
        with col4:
            macd_crosses = len(sorted_df[sorted_df['MACD_Cross_Up'] == True])
            st.metric("MACD Crosses", macd_crosses)

        # Results table
        st.subheader("🎯 Swing Trading Candidates")


        # Color coding for entry quality
        def color_entry_quality(val):
            if val == 'Excellent':
                return 'background-color: #90EE90'
            elif val == 'Good':
                return 'background-color: #FFFFE0'
            elif val == 'Fair':
                return 'background-color: #FFE4B5'
            else:
                return 'background-color: #FFB6C1'


        styled_df = sorted_df.style.applymap(color_entry_quality, subset=['Entry_Quality'])
        st.dataframe(styled_df, use_container_width=True)

        # Charts section
        st.subheader("📊 Analysis Charts")

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Individual Stock", "Overview Charts", "Screening Metrics"])

        with tab1:
            # Individual stock analysis
            ticker = st.selectbox("Select Ticker for Detailed View", sorted_df["Ticker"].unique(), index=0)

            # Time range selection
            col1, col2 = st.columns(2)
            with col1:
                time_range = st.selectbox("Time Range", ["6M", "3M", "1M"], index=0)
            with col2:
                interval = st.selectbox("Interval", ["4H", "1D"], index=0)

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
                        # Localize only if tz-naive
                        if chart_data.index.tz is None:
                            chart_data["Date"] = pd.to_datetime(chart_data.index).tz_localize("UTC")
                        else:
                            chart_data["Date"] = pd.to_datetime(chart_data.index)
                    else:
                        chart_data["Date"] = pd.to_datetime(chart_data.index).tz_localize("UTC")

                    # Filter by time range
                    range_days = {"6M": 180, "3M": 90, "1M": 30}
                    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=range_days[time_range])
                    chart_data = chart_data[chart_data["Date"] >= cutoff]

                    # Create subplots
                    fig = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=(f'{ticker} Price & Indicators', 'RSI', 'MACD'),
                        row_heights=[0.6, 0.2, 0.2]
                    )

                    # Candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=chart_data['Date'],
                            open=chart_data['Open'],
                            high=chart_data['High'],
                            low=chart_data['Low'],
                            close=chart_data['Close'],
                            name="Price"
                        ), row=1, col=1
                    )

                    # Add MA200 and VWAP
                    if 'MA200' in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['MA200'],
                                name="MA200",
                                line=dict(color='blue', width=2)
                            ), row=1, col=1
                        )

                    if 'VWAP' in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['VWAP'],
                                name="VWAP",
                                line=dict(color='orange', width=2)
                            ), row=1, col=1
                        )

                    if 'Support_20' in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['Support_20'],
                                name="Support",
                                line=dict(color='green', dash='dash', width=1)
                            ), row=1, col=1
                        )

                    # RSI
                    if 'RSI' in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['RSI'],
                                name="RSI",
                                line=dict(color='purple')
                            ), row=2, col=1
                        )
                        # RSI levels
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                    # MACD
                    if 'MACD' in chart_data.columns and 'MACD_signal' in chart_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['MACD'],
                                name="MACD",
                                line=dict(color='blue')
                            ), row=3, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['MACD_signal'],
                                name="Signal",
                                line=dict(color='red')
                            ), row=3, col=1
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

                    fig.update_layout(
                        height=800,
                        title=f"{ticker} Swing Trading Analysis",
                        xaxis_rangeslider_visible=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Display current metrics for selected stock
                    stock_data = sorted_df[sorted_df['Ticker'] == ticker].iloc[0]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${stock_data['Close']}")
                        st.metric("RSI", f"{stock_data['RSI']:.1f}")
                    with col2:
                        st.metric("VWAP", f"${stock_data['VWAP']:.2f}")
                        st.metric("MA200", f"${stock_data['MA200']:.2f}")
                    with col3:
                        st.metric("Support", f"${stock_data['Support_20']:.2f}")
                        st.metric("Volume Ratio", f"{stock_data['Volume_Ratio']:.1f}x")
                    with col4:
                        st.metric("Swing Score", f"{stock_data['Swing_Score']:.1f}")
                        st.metric("Entry Quality", stock_data['Entry_Quality'])

                except Exception as e:
                    st.error(f"Failed to render chart: {e}")
            else:
                st.warning(f"No chart data found for {ticker}")

        with tab2:
            # Overview charts
            col1, col2 = st.columns(2)

            with col1:
                # Top opportunities by swing score
                fig_score = px.bar(
                    sorted_df.head(15),
                    x="Ticker", y="Swing_Score",
                    color="Entry_Quality",
                    title="Top 15 Swing Opportunities by Score",
                    color_discrete_map={
                        'Excellent': '#90EE90',
                        'Good': '#FFFFE0',
                        'Fair': '#FFE4B5',
                        'Poor': '#FFB6C1'
                    }
                )
                st.plotly_chart(fig_score, use_container_width=True)

            with col2:
                # RSI distribution
                fig_rsi = px.histogram(
                    sorted_df,
                    x="RSI",
                    title="RSI Distribution",
                    nbins=20
                )
                fig_rsi.add_vline(x=30, line_dash="dash", line_color="red",
                                  annotation_text="RSI 30")
                st.plotly_chart(fig_rsi, use_container_width=True)

            # MACD crosses vs non-crosses
            macd_summary = sorted_df.groupby('MACD_Cross_Up').agg({
                'Swing_Score': 'mean',
                'Ticker': 'count'
            }).round(2)
            macd_summary.columns = ['Avg_Score', 'Count']

            fig_macd = px.bar(
                macd_summary.reset_index(),
                x='MACD_Cross_Up', y='Count',
                title="MACD Cross Distribution",
                color='MACD_Cross_Up'
            )
            st.plotly_chart(fig_macd, use_container_width=True)

        with tab3:
            # Screening metrics
            st.subheader("📈 Screening Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Entry Quality Distribution:**")
                quality_counts = sorted_df['Entry_Quality'].value_counts()
                fig_quality = px.pie(
                    values=quality_counts.values,
                    names=quality_counts.index,
                    title="Entry Quality Distribution"
                )
                st.plotly_chart(fig_quality, use_container_width=True)

            with col2:
                st.write("**Score vs RSI Correlation:**")
                fig_scatter = px.scatter(
                    sorted_df,
                    x="RSI", y="Swing_Score",
                    color="Entry_Quality",
                    size="Volume_Ratio",
                    hover_data=["Ticker"],
                    title="Swing Score vs RSI"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Summary statistics
            st.write("**Summary Statistics:**")
            summary_stats = sorted_df[['RSI', 'Swing_Score', 'Volume_Ratio']].describe()
            st.dataframe(summary_stats)

    else:
        st.warning("📄 Results file found, but no swing opportunities detected yet.")

else:
    st.info("🕒 Awaiting swing analysis results...")

# Auto-refresh logic (restore original functionality)
if st.session_state.swing_analysis_started:
    progress = get_swing_progress()
    time.sleep(1)
    st.rerun()
elif os.path.exists("swing_progress.txt") and get_swing_progress() >= 1.0:
    st.sidebar.success("✅ Swing analysis complete!")