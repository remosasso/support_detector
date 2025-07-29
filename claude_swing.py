import time
import yfinance as yf
import pandas as pd
import ta
import csv
import os
from tqdm import tqdm
import numpy as np
import shutil

# --- Configuration ---
# You can get your ticker list from your files or define it here

# For demonstration, using a predefined list of well-known tickers
# tickers = [
#     "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "JNJ", "V",
#     "PG", "UNH", "HD", "MA", "COST", "PEP", "AVGO", "ADBE", "CRM", "WMT",
#     "BAC", "MCD", "KO", "PFE", "DIS", "CSCO", "NFLX", "TMO", "ACN", "ABT",
#     "HON", "IBM", "SBUX", "CAT", "GS", "DE", "INTC", "AMD", "QCOM", "TXN"
# ]
# tickers = [
#     "IT", "NTRA", "CHTR", "CMG", "COST", "DUOL", "FI", "CHWY", "HCA", "SUI",
#     "STM", "EFX", "GDDY", "BIP", "LULU", "CNI", "INTC", "DOC", "TXN", "OTIS",
#     "ELS", "BRO", "SAP", "UMC", "TECK", "BUD", "NOK", "TXT", "CP", "EHC", "WAB",
#     "INVH", "MMC", "CCK", "RGLD", "LUV", "HON", "CNC", "SWKS", "PM", "AMH",
#     "CMCSA", "FMS", "RS", "IBM", "INFY", "GLPI", "AR", "ERIC", "BSBR", "COR",
#     "CVS", "SBAC", "EXE", "CB", "6857.T", "KO", "KR", "RELX", "EBR", "BBD",
#     "CL", "DG", "PNW", "HRL", "LMT", "TAK", "ACI", "SBS", "FMX",    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "JNJ", "V",
#     "PG", "UNH", "HD", "MA", "COST", "PEP", "AVGO", "ADBE", "CRM", "WMT",
#     "BAC", "MCD", "KO", "PFE", "DIS", "CSCO", "NFLX", "TMO", "ACN", "ABT",
#     "HON", "IBM", "SBUX", "CAT", "GS", "DE", "INTC", "AMD", "QCOM", "TXN"
# ]


if os.path.exists("liquid_big_tickers.txt"):
    with open("liquid_big_tickers.txt", "r") as f:
        tickers = [line.strip() for line in f.readlines()]
        # unique
        tickers = list(set(tickers))
else:
    from tickers import us_ticker_dict, eu_ticker_dict
    tickers = list(us_ticker_dict.keys()) + list(eu_ticker_dict.keys())


# --- Helper & Data Calculation Functions ---

def pct_rank(series, high_is_good=True):
    """Ranks a series into a percentile (0-1) scale. Higher is better."""
    pct = series.rank(pct=True, method="max")
    return pct if high_is_good else 1 - pct


def compute_indicators_multi_timeframe(df_4h, df_1d, ticker):
    """
    Compute technical indicators for both 4H and 1D timeframes.
    This function remains as a data provider.
    """
    # 4H timeframe indicators
    if not df_4h.empty:
        close_4h = df_4h['Close']
        df_4h['RSI_4H'] = ta.momentum.RSIIndicator(close=close_4h, window=14).rsi()

    # 1D timeframe indicators
    if not df_1d.empty:
        close_1d = df_1d['Close']
        high_1d = df_1d['High']
        low_1d = df_1d['Low']
        df_1d['MA200'] = close_1d.rolling(window=200).mean()
        df_1d['MA50_1D'] = close_1d.rolling(window=50).mean()
        df_1d['Support_20_1D'] = close_1d.rolling(window=20).min()
        df_1d['Resistance_20_1D'] = close_1d.rolling(window=20).max()
        df_1d['ATR_1D'] = ta.volatility.AverageTrueRange(high=high_1d, low=low_1d, close=close_1d,
                                                         window=14).average_true_range()

    return df_4h, df_1d


def get_pillar_metrics(tick, info, last_4h, last_1d, df_1d):
    """
    Gathers all raw metrics needed for the four pillars.
    *** Includes a more robust ATR-based Risk/Reward calculation. ***
    """
    try:
        close_price = last_4h['Close'].item()

        # --- Pillar 1: Quality Metrics ---
        # ... (no change needed here)
        ebitda_margins = info.get('ebitdaMargins', 0)
        debt_to_equity = info.get('debtToEquity')
        roe = info.get('returnOnEquity', 0)
        market_cap = info.get('marketCap', 0)

        # --- Pillar 2: Trend Metrics ---
        # ... (no change needed here)
        ma50 = last_1d['MA50_1D'].item()
        ma200 = last_1d['MA200'].item()
        price_gt_ma200 = 1 if close_price > ma200 else 0
        ma50_gt_ma200 = 1 if ma50 > ma200 else 0

        # --- Pillar 3: Mean-Reversion (Pullback) Metrics ---
        # ... (no change needed here)
        rsi_4h = last_4h['RSI_4H'].item()
        support_20d = last_1d['Support_20_1D'].item()
        dist_to_support_pct = (close_price - support_20d) / support_20d if support_20d > 0 else 0

        # --- Pillar 4: Risk/Reward & Liquidity Metrics ---
        atr_val = last_1d['ATR_1D'].item()
        atr_pct = (atr_val / close_price) * 100 if close_price > 0 else 0
        avg_dollar_volume = info.get('averageVolume', 0) * close_price

        # ### IMPROVEMENT HERE ###
        # Use an ATR-based stop for a more realistic R/R calculation.
        resistance = last_1d['Resistance_20_1D'].item()
        potential_reward = resistance - close_price

        # Define risk as 2x the daily ATR from the close.
        defined_risk = 2 * atr_val

        # Calculate the new, more robust R/R ratio.
        # Ensure risk is not zero to avoid division errors.
        rr_ratio = potential_reward / defined_risk if defined_risk > 0 else 0

        return {
            "Ticker": info.get('symbol'),
            "Close": close_price,
            # ... (rest of the keys are the same)
            "rr_ratio": rr_ratio,  # This value will now be much more realistic
            # ... (all other keys)
            "ebitda_margins": ebitda_margins,
            "debt_to_equity": debt_to_equity,
            "roe": roe,
            "market_cap": market_cap,
            "price_gt_ma200": price_gt_ma200,
            "ma50_gt_ma200": ma50_gt_ma200,
            "rsi_4h": rsi_4h,
            "dist_to_support_pct": dist_to_support_pct,
            "atr_pct": atr_pct,
            "beta": info.get('beta', 1.0),
            "avg_dollar_volume": avg_dollar_volume
        }

    except (KeyError, IndexError, TypeError) as e:
        print(f"    - Could not gather pillar metrics for {info.get('symbol', 'N/A')}: {e}")
        return None
# --- Main Analysis Function ---

def analyze_swing_opportunities(set_progress, update_info=False):
    """
    Refactored analysis using a pillar-based, percentile-ranked scoring system.
    """
    # --- Setup and Cleanup ---
    output_filename = "swing_results_ranked.csv"
    lock_filename = "swing_analysis.lock"

    if os.path.exists(lock_filename):
        print("⚠ Analysis already running. Exiting.")
        return
    with open(lock_filename, "w") as f:
        f.write("running")

    if os.path.exists(output_filename): os.remove(output_filename)

    print("Starting Advanced Swing Analysis...")

    # --- STEP 1: GATHER RAW DATA FOR ALL TICKERS ---
    all_results_data = []
    print("\n--- Step 1: Gathering raw data for all tickers ---")
    for i, ticker in tqdm(enumerate(tickers), total=len(tickers), desc="Fetching Data"):
        if callable(set_progress): set_progress((i + 1) / len(tickers))

        move_on = False
        while True:
            try:
                # Download both timeframes
                tick = yf.Ticker(ticker)
                info = tick.info
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

        # Pre-computation filters to save time
        if info.get('marketCap', 0) < 1e10 or info.get('averageVolume', 0) < 200000:
            print(f"Skipping {ticker} due to low market cap or volume.")
            continue

        df_4h = tick.history(period="3mo", interval="1h", auto_adjust=True)
        df_1d = tick.history(period="2y", interval="1d", auto_adjust=True)

        if df_4h.empty or df_1d.empty or len(df_1d) < 201:
            print(f"Skipping {ticker} due to insufficient data.")
            continue

        df_4h, df_1d = compute_indicators_multi_timeframe(df_4h, df_1d, ticker)
        df_4h, df_1d = df_4h.dropna(), df_1d.dropna()

        if df_4h.empty or df_1d.empty:
            print(f"Skipping {ticker} due to NaN values in indicators.")
            continue

        metrics = get_pillar_metrics(tick, info, df_4h.iloc[-1], df_1d.iloc[-1], df_1d)
        if metrics: all_results_data.append(metrics)

    # except Exception as e:
    #     Hide common, non-critical yfinance messages to reduce noise
        # if "No data found" not in str(e) and "404" not in str(e):
        #     print(f"Error processing {ticker}: {e}")
        # continue

    if not all_results_data:
        print("\nNo valid ticker data gathered. Exiting analysis.")
        if os.path.exists(lock_filename): os.remove(lock_filename)
        return

    # --- STEP 2: CREATE DATAFRAME & APPLY HARD FILTERS ---
    print("\n--- Step 2: Applying hard filters to the universe ---")
    df = pd.DataFrame(all_results_data)
    df = df.dropna(subset=['debt_to_equity', 'ebitda_margins'])

    print(f"Starting with {len(df)} potential tickers.")
    original_tickers = df.copy()
    df = df.loc[df["rr_ratio"] >= 2.0].copy()
    print(f"{len(df)} tickers remain after R/R filter (>= 2.0).")
    print("Tickers that were filtered out ", original_tickers.loc[original_tickers["rr_ratio"] < 2.0, "Ticker"].tolist())

    df = df.loc[df["ebitda_margins"] > 0.10].copy()
    print(f"{len(df)} tickers remain after EBIT Margin filter (> 10%).")
    print("Tickers that were filtered out ", original_tickers.loc[original_tickers["ebitda_margins"] <= 0.10, "Ticker"].tolist())

    df = df.loc[df["debt_to_equity"] < 200].copy()  # yfinance D/E is in %, so < 200 means D/E < 2
    print(f"{len(df)} tickers remain after Debt/Equity filter (< 2).")
    print("Tickers that were filtered out ", original_tickers.loc[original_tickers["debt_to_equity"] >= 200, "Ticker"].tolist())

    df = df.loc[df["price_gt_ma200"] == 1].copy()
    print(f"{len(df)} tickers remain after Trend filter (Price > 200DMA).")
    print("Tickers that were filtered out ", original_tickers.loc[original_tickers["price_gt_ma200"] == 0, "Ticker"].tolist())

    if df.empty:
        print("\nNo tickers passed the hard filters. No prime setups found today.")
        if os.path.exists(lock_filename): os.remove(lock_filename)
        return

    # --- STEP 3: NORMALIZE METRICS WITH PERCENTILE RANKING ---
    print("\n--- Step 3: Normalizing metrics with percentile ranks ---")

    # Pillar 1: Quality Rank
    df["quality_rank"] = (
                                 pct_rank(df["market_cap"], high_is_good=True) +
                                 pct_rank(df["roe"], high_is_good=True) +
                                 pct_rank(df["debt_to_equity"], high_is_good=False)  # Lower debt is better
                         ) / 3

    # Pillar 2: Trend Rank
    df["trend_rank"] = (df["price_gt_ma200"] + df["ma50_gt_ma200"]) / 2

    # Pillar 3: Pullback Rank
    df["pullback_rank"] = (
                                  pct_rank(df["rsi_4h"], high_is_good=False) +  # Lower RSI is better
                                  pct_rank(df["dist_to_support_pct"], high_is_good=False)  # Closer to support is better
                          ) / 2

    # Pillar 4: Risk Rank (with flipped Beta & ATR logic)
    beta_bonus = 1 + (1.2 - df["beta"]).clip(0, 0.4)
    beta_adj = np.where(df["beta"] > 1.2, 0.8, beta_bonus)

    df["risk_rank"] = pct_rank(df["atr_pct"], high_is_good=False) * beta_adj  # Lower ATR is better, adjusted by beta

    # --- STEP 4: CALCULATE FINAL WEIGHTED SCORE ---
    print("\n--- Step 4: Calculating final weighted swing score ---")
    df["swing_score"] = (
                                0.2 * df["quality_rank"]
                                + 0.2 * df["trend_rank"]
                                + 0.5 * df["pullback_rank"]
                                + 0.1 * df["risk_rank"]
                        ) * 100

    # --- STEP 5: TIER OUTPUT AND SAVE ---
    print("\n--- Step 5: Tiering and saving final results ---")
    conditions = [
        df.swing_score >= 80,
        df.swing_score >= 60,
    ]
    choices = ["A (Prime)", "B (Watch)"]
    df["Tier"] = np.select(conditions, choices, default="C (Speculative)")

    final_results = df.sort_values(by="swing_score", ascending=False)

    output_columns = [
        "Tier", "Ticker", "swing_score", "Close", "rr_ratio",
        "market_cap", "beta", "atr_pct", "rsi_4h",
        "quality_rank", "trend_rank", "pullback_rank", "risk_rank"
    ]
    final_results = final_results.reindex(columns=output_columns).copy()

    # Formatting for better readability
    final_results['swing_score'] = final_results['swing_score'].round(1)
    final_results['rr_ratio'] = final_results['rr_ratio'].round(2)
    final_results['market_cap'] = final_results['market_cap'].apply(lambda x: f"${x / 1e9:.1f}B")
    for col in ["quality_rank", "trend_rank", "pullback_rank", "risk_rank"]:
        final_results[col] = (final_results[col] * 100).round(1)

    final_results.to_csv(output_filename, index=False)

    # --- Final Cleanup and Summary ---
    if os.path.exists(lock_filename): os.remove(lock_filename)

    print(f"\n✅ Advanced analysis complete. Found {len(final_results)} potential setups.")
    print(f"Results saved to '{output_filename}'")

    # Optional: Print top 5 results to console
    print("\n--- Top 5 Prime Candidates ---")
    print(final_results.head(50).to_string(index=False))


# --- Wrapper and Execution Block ---

def start_swing_analysis(set_progress, update_info=False):
    """Wrapper function to start the enhanced analysis"""
    analyze_swing_opportunities(set_progress, update_info)

DEBUG = False
if DEBUG:
    if os.path.exists("swing_analysis.lock"):
        os.remove("swing_analysis.lock")
    start_swing_analysis(lambda x: print(f"Progress: {x * 100:.2f}%"))