import yfinance as yf
import pandas as pd
import ta
import numpy as np
import time
import csv
import os
import pickle as pkl
from tqdm import tqdm
from tickers import us_ticker_dict

tickers = list(us_ticker_dict.keys())


def compute_indicators(df, ticker):
    """
    Compute technical indicators for swing trading analysis
    """
    if df.empty:
        return df

    close = df['Close'][ticker]
    high = df['High'][ticker]
    low = df['Low'][ticker]
    volume = df['Volume'][ticker]

    # Technical indicators
    df['RSI'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd = ta.trend.MACD(close=close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
        high=high, low=low, close=close, volume=volume
    ).vwap

    df['MA200'] = close.rolling(window=200).mean()  # 200-period MA
    df['Support_20'] = close.rolling(window=20).min()  # 20-period support
    df['Resistance_20'] = close.rolling(window=20).max()  # 20-period resistance

    # Additional indicators for scoring
    df['MA50'] = close.rolling(window=50).mean()
    df['Volume_MA'] = volume.rolling(window=20).mean()
    df['Volume_Ratio'] = volume / df['Volume_MA']

    return df


def calculate_swing_score(row, ticker):
    """
    Calculate a comprehensive swing trading score
    """
    score = 0

    # RSI scoring (lower is better for oversold bounce)
    rsi_val = row['RSI'].item()
    if rsi_val < 20:
        score += 40
    elif rsi_val < 25:
        score += 30
    elif rsi_val < 30:
        score += 20

    # MACD cross scoring
    if row['MACD'].item() > row['MACD_signal'].item():
        score += 25

    # Distance from VWAP (closer = better entry)
    close_val = row['Close'][ticker]
    vwap_val = row['VWAP'].item()
    vwap_distance = abs(close_val - vwap_val) / vwap_val
    if vwap_distance < 0.02:  # within 2%
        score += 20
    elif vwap_distance < 0.05:  # within 5%
        score += 10

    # Trend strength (distance above MA200)
    ma200_val = row['MA200'].item()
    ma200_distance = (close_val - ma200_val) / ma200_val
    if ma200_distance > 0.1:  # 10% above MA200
        score += 15
    elif ma200_distance > 0.05:  # 5% above MA200
        score += 10

    # Volume confirmation
    vol_ratio = row['Volume_Ratio'].item()
    if vol_ratio > 1.5:  # Above average volume
        score += 10

    # Support proximity
    support_val = row['Support_20'].item()
    support_distance = (close_val - support_val) / support_val
    if support_distance < 0.05:  # within 5% of support
        score += 15

    return score


def copy_results_snapshot():
    """Copy results to stable file for dashboard reading"""
    try:
        import shutil
        shutil.copyfile("swing_results.csv", "swing_results_stable.csv")
        df = pd.read_csv("swing_results_stable.csv")
        df = df.drop_duplicates(subset="Ticker", keep="last")  # Keep latest entry per ticker
        df.to_csv("swing_results_stable.csv", index=False)
        print("📄 Copied snapshot to swing_results_stable.csv")
    except Exception as e:
        print(f"⚠ Failed to copy swing_results.csv: {e}")


def analyze_swing_opportunities(set_progress, update_info=False):
    """
    Main analysis function for swing trading opportunities
    """
    # Cleanup old files
    for f in ["swing_results.csv", "swing_results_stable.csv", "swing_progress.txt"]:
        if os.path.exists(f):
            os.remove(f)

    """
    Main analysis function for swing trading opportunities
    """
    # Cleanup old files
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
        print("Starting swing analysis...")

    # Initialize CSV with headers
    fieldnames = [
        "Ticker", "Close", "RSI", "MACD", "MACD_Signal", "VWAP",
        "MA200", "Support_20", "MACD_Cross_Up", "Volume_Ratio",
        "Swing_Score", "Entry_Quality"
    ]

    with open("swing_results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    results = []
    # tickers = ['AMBP']

    total = len(tickers)
    for i, ticker in tqdm(enumerate(tickers)):
        try:
            set_progress((i + 1) / total)

            # Download data
            df = yf.download(ticker, interval='4h', period='6mo', auto_adjust=True, progress=False)

            if df.empty:
                continue

            # Compute indicators
            df = compute_indicators(df, ticker).dropna()

            if df.empty:
                continue

            last = df.iloc[-1]

            # Core swing trading conditions
            if not (last['Close'][ticker] > last['MA200'].item() and
                    last['RSI'].item() < 30 and
                    last['Close'][ticker] < last['VWAP'].item()):
                continue

            # Calculate swing score
            swing_score = calculate_swing_score(last, ticker)

            # Determine entry quality
            if swing_score >= 80:
                entry_quality = "Excellent"
            elif swing_score >= 60:
                entry_quality = "Good"
            elif swing_score >= 40:
                entry_quality = "Fair"
            else:
                entry_quality = "Poor"

            result = {
                'Ticker': ticker,
                'Close': round(last['Close'][ticker].item(), 2),
                'RSI': round(last['RSI'].item(), 2),
                'MACD': round(last['MACD'].item(), 4),
                'MACD_Signal': round(last['MACD_signal'].item(), 4),
                'VWAP': round(last['VWAP'].item(), 2),
                'MA200': round(last['MA200'].item(), 2),
                'Support_20': round(last['Support_20'].item(), 2),
                'MACD_Cross_Up': last['MACD'].item() > last['MACD_signal'].item(),
                'Volume_Ratio': round(last['Volume_Ratio'].item(), 2),
                'Swing_Score': round(swing_score, 1),
                'Entry_Quality': entry_quality
            }

            results.append(result)

            # Write to CSV
            with open("swing_results.csv", "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result)

            # Save chart data
            df.to_parquet(f"swing_chart_data/{ticker}_4h.parquet")

            # Also save daily data for longer-term view
            df_daily = yf.download(ticker, period="1y", interval="1d", progress=False)
            if not df_daily.empty:
                df_daily = compute_indicators(df_daily, ticker)
                df_daily.to_parquet(f"swing_chart_data/{ticker}_1d.parquet")
            copy_results_snapshot()

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    # Create stable results
    try:
        import shutil
        shutil.copyfile("swing_results.csv", "swing_results_stable.csv")
        print("📄 Created stable results file")
    except Exception as e:
        print(f"⚠ Failed to create stable results: {e}")
    copy_results_snapshot()

    # Cleanup
    if os.path.exists("swing_analysis.lock"):
        os.remove("swing_analysis.lock")

    print(f"✅ Analysis complete. Found {len(results)} swing opportunities.")


def start_swing_analysis(set_progress, update_info=False):
    """Wrapper function to start the analysis"""
    analyze_swing_opportunities(set_progress, update_info)


# Debug mode
DEBUG = False
if DEBUG:
    if os.path.exists("swing_analysis.lock"):
        os.remove("swing_analysis.lock")
    start_swing_analysis(lambda x: print(f"Progress: {x * 100:.2f}%"))