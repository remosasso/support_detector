import time

from tickers import us_ticker_dict, eu_ticker_dict

tickers = list(us_ticker_dict.keys()) + list(eu_ticker_dict.keys())

# Enhanced swing analysis with multiple timeframes

import yfinance as yf
import pandas as pd
import ta
import csv
import os
from tqdm import tqdm
# tickers = ['AMBP', "IBM", "BJRI"]
def compute_indicators_multi_timeframe(df_4h, df_1d, ticker):
    """
    Compute indicators for both timeframes and combine insights
    """
    indicators = {}

    # 4H timeframe indicators (for short-term signals)
    if not df_4h.empty:
        close_4h = df_4h['Close']
        high_4h = df_4h['High']
        low_4h = df_4h['Low']
        volume_4h = df_4h['Volume']

        indicators['RSI_4H'] = ta.momentum.RSIIndicator(close=close_4h, window=14).rsi()
        macd_4h = ta.trend.MACD(close=close_4h)
        indicators['MACD_4H'] = macd_4h.macd()
        indicators['MACD_signal_4H'] = macd_4h.macd_signal()
        indicators['VWAP_4H'] = ta.volume.VolumeWeightedAveragePrice(
            high=high_4h, low=low_4h, close=close_4h, volume=volume_4h
        ).vwap
        indicators['MA50_4H'] = close_4h.rolling(window=50).mean()
        indicators['Volume_MA_4H'] = volume_4h.rolling(window=20).mean()
        indicators['Volume_Ratio_4H'] = volume_4h / indicators['Volume_MA_4H']
        indicators['Support_20_4H'] = close_4h.rolling(window=20).min()
        indicators['Resistance_20_4H'] = close_4h.rolling(window=20).max()

        # Add to 4H dataframe
        for key, value in indicators.items():
            df_4h[key] = value

    # 1D timeframe indicators (for trend confirmation)
    if not df_1d.empty:
        close_1d = df_1d['Close']
        high_1d = df_1d['High']
        low_1d = df_1d['Low']
        volume_1d = df_1d['Volume']

        df_1d['RSI_1D'] = ta.momentum.RSIIndicator(close=close_1d, window=14).rsi()
        macd_1d = ta.trend.MACD(close=close_1d)
        df_1d['MACD_1D'] = macd_1d.macd()
        df_1d['MACD_signal_1D'] = macd_1d.macd_signal()
        df_1d['VWAP_1D'] = ta.volume.VolumeWeightedAveragePrice(
            high=high_1d, low=low_1d, close=close_1d, volume=volume_1d
        ).vwap
        df_1d['MA200'] = close_1d.rolling(window=200).mean()  # Long-term trend
        df_1d['MA50_1D'] = close_1d.rolling(window=50).mean()
        df_1d['Support_20_1D'] = close_1d.rolling(window=20).min()
        df_1d['Resistance_20_1D'] = close_1d.rolling(window=20).max()

        df_1d['Volume_MA_1D'] = volume_1d.rolling(window=20).mean()
        df_1d['Volume_Ratio_1D'] = volume_1d / df_1d['Volume_MA_1D']

    return df_4h, df_1d


def calculate_multi_timeframe_score(last_4h, last_1d, ticker):
    """
    Calculate swing score using both timeframes
    """
    score = 0

    # 4H RSI (primary signal - more sensitive)
    rsi_4h = last_4h['RSI_4H'].item()
    if rsi_4h < 20:
        score += 50
    elif rsi_4h < 25:
        score += 40
    elif rsi_4h < 30:
        score += 30
    elif rsi_4h < 35:  # Slightly higher threshold for 4H
        score += 10

    # 1D RSI (confirmation - trend context)
    if 'RSI_1D' in last_1d.index:
        rsi_1d = last_1d['RSI_1D'].item()
        if rsi_1d < 30:
            score += 30  # Strong oversold
        elif rsi_1d < 40:
            score += 15
        elif rsi_1d < 50:
            score += 5

    # MACD signals (combine both timeframes)
    # 4H MACD for entry timing
    if last_4h['MACD_4H'].item() > last_4h['MACD_signal_4H'].item():
        score += 20

    # 1D MACD for trend confirmation
    if 'MACD_1D' in last_1d.index and 'MACD_signal_1D' in last_1d.index:
        if last_1d['MACD_1D'].item() > last_1d['MACD_signal_1D'].item():
            score += 15

    # VWAP analysis (use 4H for entry precision)
    close_val = last_4h['Close']
    vwap_4h = last_4h['VWAP_4H'].item()
    vwap_distance = abs(close_val - vwap_4h) / vwap_4h
    if vwap_distance < 0.01:  # Very close to VWAP
        score += 15
    elif vwap_distance < 0.02:  # Within 2%
        score += 10
    elif vwap_distance < 0.05:  # Within 5%
        score += 5

    # Long-term trend (1D MA200)
    if 'MA200' in last_1d.index:
        ma200_val = last_1d['MA200'].item()
        ma200_distance = (close_val - ma200_val) / ma200_val
        if ma200_distance > 0.15:  # Strong uptrend
            score += 5
        elif ma200_distance > 0.1:  # Moderate uptrend
            score += 2.5
        elif ma200_distance > 0.05:  # Weak uptrend
            score += 1

    # Volume confirmation (prefer 1D for significance)
    if 'Volume_Ratio_1D' in last_1d.index:
        vol_ratio_1d = last_1d['Volume_Ratio_1D'].item()
        if vol_ratio_1d > 2.0:  # Very high volume
            score += 5
        elif vol_ratio_1d > 1.5:  # High volume
            score += 2.5

    # Support proximity (1D support levels)
    if 'Support_20_1D' in last_1d.index:
        support_val = last_1d['Support_20_1D'].item()
        support_distance = (close_val - support_val) / support_val
        if support_distance < 0.03:  # Very close to support
            score += 20
        elif support_distance < 0.08:  # Near support
            score += 15

    return score


def is_structurally_bullish(df_1d):
    """
    Checks if long-term structure is bullish:
    - MA200 trending up
    - Current close above both MA200 and MA50
    - 6-month price change positive
    """
    if df_1d.empty or len(df_1d) < 120:
        return False  # Not enough data

    close_series = df_1d['Close']
    ma200 = df_1d['MA200']
    ma50 = df_1d['MA50_1D']

    # Trend slope of MA200 over last 3 months
    recent_ma200 = ma200[-60:]
    slope = (recent_ma200.iloc[-1].item() - recent_ma200.iloc[0].item()) / recent_ma200.iloc[0].item()

    # Price structure
    recent_close = close_series.iloc[-1].item()
    prev_close = close_series.iloc[-120].item() # ~6 months ago

    # print(f"Recent Close: {recent_close}, MA200: {ma200.iloc[-1].item()}, MA50: {ma50.iloc[-1].item()}, Slope: {slope}")


    return (
            slope > 0.02 and
            recent_close > ma200.iloc[-1].item() and
            recent_close > prev_close
    )


def analyze_swing_opportunities_multi_tf(set_progress, update_info=False):
    """
    Enhanced analysis with multiple timeframes
    """
    # Cleanup and setup
    for f in ["swing_results.csv", "swing_results_stable.csv", "swing_progress.txt"]:
        if os.path.exists(f):
            os.remove(f)

    import shutil
    if os.path.exists("swing_chart_data"):
        shutil.rmtree("swing_chart_data")
    os.makedirs("swing_chart_data", exist_ok=True)

    # Create lock file
    if os.path.exists("swing_analysis.lock"):
        print("⚠ Analysis already running. Exiting.")
        return

    with open("swing_analysis.lock", "w") as f:
        f.write("running")
        print("Starting multi-timeframe swing analysis...")

    # Enhanced fieldnames for multi-timeframe analysis
    fieldnames = [
        "Ticker", "Close",
        "RSI_4H", "RSI_1D",
        "MACD_4H", "MACD_Signal_4H", "MACD_1D", "MACD_Signal_1D",
        "VWAP_4H", "VWAP_1D", "MA200", "Support_20_1D", "Support_20_4H",
        "MACD_Cross_4H", "MACD_Cross_1D",
        "Volume_Ratio_4H", "Volume_Ratio_1D",
        "Swing_Score", "Entry_Quality", "Timeframe_Alignment",
        "Beta", "Market_Cap", "Price", "ATR", "Estimated_Days"
    ]

    with open("swing_results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    results = []
    # tickers = ["GNE", "XP", "XYF", "AMBP", "TIMB", "IBM", "FCFS", "PM", "XP", "SARO", "JFIN", "LIGHT.AS", "FTK"]
    # swing_tickers2 = [
    #     "COIN", "INTC"
    #     "XYF", "IBM", "PM", "GNE", "DB1.DE",
    #     "BCH", "TIMB", "FCFS", "JFIN", "FTK",
    #     "PRDO", "GEL", "SPRO", "EHC", "BJRI",
    #     "ADEA", "DGICA", "ENIC", "UVE", "NATH",
    #     "TSLA", "AAPL", "GOOGL", "AMZN", "MSFT",
    #     "NVDA", "META", "NFLX", "AMD", "INTC",
    # ]
    # tickers = swing_tickers2
    if os.path.exists("liquid_big_tickers.txt"):
        with open("liquid_big_tickers.txt", "r") as f:
            liquid_big_tickers = [line.strip() for line in f.readlines()]
    else:
        liquid_big_tickers = []
    total = len(tickers)
    seen_last = False
    for i, ticker in tqdm(enumerate(tickers)):
        if ticker not in liquid_big_tickers:
            continue
        try:
            set_progress((i + 1) / total)
            move_on = False
            while True:
                try:
                    # Download both timeframes
                    tick = yf.Ticker(ticker)
                    df_4h = tick.history(period="6mo", interval="4h", auto_adjust=True)
                    df_1d = tick.history(period="5y", interval="1d", auto_adjust=True)
                    info = tick.info
                    market_cap = info.get('marketCap', 0)
                    price = info.get('currentPrice', 0)
                    beta = info.get('beta', 0)
                    close = info.get('previousClose', 0)
                    break
                except Exception as e:
                    if "401" in str(e):
                        print(f"⚠ Failed to fetch data for {ticker}. Retrying...")
                        time.sleep(60)
                    else:
                        move_on = True
                        break
            if move_on:
                print(f"⚠ Failed to fetch data for {ticker}. Skipping...")
                continue
            if df_4h.empty or df_1d.empty:
                continue

            # Compute indicators for both timeframes
            df_4h, df_1d = compute_indicators_multi_timeframe(df_4h, df_1d, ticker)

            # Get latest data points
            df_4h = df_4h.dropna()
            df_1d = df_1d.dropna()

            if df_4h.empty or df_1d.empty:
                continue

            last_4h = df_4h.iloc[-1]
            last_1d = df_1d.iloc[-1]

            # Multi-timeframe screening conditions
            close_val = last_4h['Close']
            close_val_1d = last_1d['Close']
            ma200_now = last_1d['MA200'].item()
            ma200_prev = df_1d.iloc[-5]['MA200'].item()

            drop_last_3_candles_pct = (last_4h['Close'].item() - last_4h['Open'].item()) / last_4h['Open'].item() * 100
            atr = ta.volatility.average_true_range(df_1d["High"], df_1d["Low"], df_1d["Close"],
                                                   window=14).iloc[-1] / close_val

            volume = last_4h['Volume'].item()
            upward_conditions = False
            if upward_conditions:
                # Primary conditions (must meet all)
                conditions = [
                    close_val_1d > last_1d['MA200'].item(),  # Above long-term trend
                    last_4h['RSI_4H'].item() < 35,  # 4H oversold (slightly higher threshold)
                    close_val < last_4h['VWAP_4H'].item(),  # Below VWAP for entry,
                    ma200_now > ma200_prev,  # MA200 is declining
                    is_structurally_bullish(df_1d)  # Long-term structure is bullish
                ]
                if len(df_1d) >= 250:

                    price_1y_ago = df_1d['Close'].iloc[-750].item()
                    if close_val < price_1y_ago:
                        print(f"⚠ {ticker} still below 1Y-ago price, skipping.")
                        continue
                if ma200_now < ma200_prev:
                    print("⚠ MA200 is declining, skipping ticker:", ticker)

            else:
                rsi_4h = last_4h['RSI_4H'].item()

                conditions = [
                        rsi_4h < 35,
                        market_cap > 1e10,
                        price > 1.0,
                        volume > 100000,  # Minimum volume threshold

                ]
            if market_cap > 1e10 and volume > 100000:
                liquid_big_tickers.append(ticker)
                with open("liquid_big_tickers.txt", "a") as f:
                    f.write(f"{ticker}\n")
                # Save to list of valid tickers


            # Secondary conditions (nice to have)
            secondary_score = 0
            if 'RSI_1D' in last_1d.index and last_1d['RSI_1D'].item() < 50:
                secondary_score += 1
            if last_4h['MACD_4H'].item() > last_4h['MACD_signal_4H'].item():
                secondary_score += 1

            # Must meet primary conditions + at least 1 secondary
            if not all(conditions) or secondary_score == 0:
                continue

            # Calculate multi-timeframe score
            swing_score = calculate_multi_timeframe_score(last_4h, last_1d, ticker)
            # swing_score += market_cap * 0.0000001  + volume * 0.00001  # Scale by market cap and volume
            swing_score *= beta  # Adjust score by beta for volatility
            # ATR-based score (volatility-aware scoring)
            if atr > 7:
                swing_score += 15
            elif atr > 5:
                swing_score += 10
            elif atr > 3:
                swing_score += 5
            # Determine entry quality and timeframe alignment
            if swing_score >= 80:
                entry_quality = "Excellent"
            elif swing_score >= 60:
                entry_quality = "Good"
            elif swing_score >= 40:
                entry_quality = "Fair"
            else:
                entry_quality = "Poor"

            # --- Estimated days formula ---
            if atr > 0 and beta > 0:
                est_days = round(0.1 / (atr * beta), 1)
            else:
                est_days = None  # Or float('inf') if you want to highlight invalid cases

            # Check timeframe alignment
            rsi_alignment = "Aligned" if (last_4h['RSI_4H'].item() < 35 and
                                          ('RSI_1D' not in last_1d.index or last_1d[
                                              'RSI_1D'].item() < 50)) else "Divergent"

            result = {
                'Ticker': ticker,
                'Close': round(close_val.item(), 2),
                'RSI_4H': round(last_4h['RSI_4H'].item(), 1),
                'RSI_1D': round(last_1d.get('RSI_1D', 0).item() if 'RSI_1D' in last_1d.index else 0, 1),
                'MACD_4H': round(last_4h['MACD_4H'].item(), 4),
                'MACD_Signal_4H': round(last_4h['MACD_signal_4H'].item(), 4),
                'MACD_1D': round(last_1d.get('MACD_1D', 0).item() if 'MACD_1D' in last_1d.index else 0, 4),
                'MACD_Signal_1D': round(
                    last_1d.get('MACD_signal_1D', 0).item() if 'MACD_signal_1D' in last_1d.index else 0, 4),
                'VWAP_4H': round(last_4h['VWAP_4H'].item(), 2),
                'VWAP_1D': round(last_1d.get('VWAP_1D', 0).item() if 'VWAP_1D' in last_1d.index else 0, 2),
                'MA200': round(last_1d['MA200'].item(), 2),
                'Support_20_4H': round(last_4h['Support_20_4H'].item(), 2),
                'Support_20_1D': round(last_1d['Support_20_1D'].item(), 2),
                'MACD_Cross_4H': last_4h['MACD_4H'].item() > last_4h['MACD_signal_4H'].item(),
                'MACD_Cross_1D': (last_1d.get('MACD_1D', 0).item() > last_1d.get('MACD_signal_1D',
                                                                                 0).item()) if 'MACD_1D' in last_1d.index else False,
                'Volume_Ratio_4H': round(last_4h['Volume_Ratio_4H'].item(), 1),
                'Volume_Ratio_1D': round(
                    last_1d.get('Volume_Ratio_1D', 0).item() if 'Volume_Ratio_1D' in last_1d.index else 0, 1),
                'Swing_Score': round(swing_score, 1),
                'Entry_Quality': entry_quality,
                'Timeframe_Alignment': rsi_alignment,
                'Beta': round(info.get('beta', 0), 2),
                'Market_Cap': info.get('marketCap', 0),
                'Price': info.get('currentPrice', 0),
                'ATR': round(atr, 4),
                'Estimated_Days': est_days,


            }

            results.append(result)

            # Write to CSV
            with open("swing_results.csv", "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result)

            # Save chart data (both timeframes)
            df_4h.to_parquet(f"swing_chart_data/{ticker}_4h.parquet")
            df_1d.to_parquet(f"swing_chart_data/{ticker}_1d.parquet")

            # Copy snapshot for dashboard
            copy_results_snapshot()

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    # Final cleanup
    copy_results_snapshot()
    if os.path.exists("swing_analysis.lock"):
        os.remove("swing_analysis.lock")

    print(f"✅ Multi-timeframe analysis complete. Found {len(results)} swing opportunities.")


def copy_results_snapshot():
    """Copy results to stable file for dashboard reading"""
    try:
        import shutil
        shutil.copyfile("swing_results.csv", "swing_results_stable.csv")
        df = pd.read_csv("swing_results_stable.csv")
        df = df.drop_duplicates(subset="Ticker", keep="last")
        df.to_csv("swing_results_stable.csv", index=False)
    except Exception as e:
        print(f"⚠ Failed to copy swing_results.csv: {e}")


def start_swing_analysis(set_progress, update_info=False):
    """Wrapper function to start the enhanced analysis"""
    analyze_swing_opportunities_multi_tf(set_progress, update_info)
# Debug mode
DEBUG = False
if DEBUG:
    if os.path.exists("swing_analysis.lock"):
        os.remove("swing_analysis.lock")
    start_swing_analysis(lambda x: print(f"Progress: {x * 100:.2f}%"))