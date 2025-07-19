import time
import csv
import os
import pickle as pkl
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
from tickers import us_ticker_dict, eu_ticker_dict
tickers = list(us_ticker_dict.keys()) + list(eu_ticker_dict.keys())
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def compute_rsi(close_series, period=14):
    """
    Compute RSI from a pandas Series that may have multi-index (Date, Ticker).
    Returns a flat Series indexed by Date.
    """
    # If Series has a multi-index, reduce to just Date index
    if isinstance(close_series.index, pd.MultiIndex):
        if 'Date' in close_series.index.names:
            close_series = close_series.droplevel([i for i in close_series.index.names if i != 'Date'])
        else:
            close_series = close_series.droplevel(-1)

    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_support_levels(df, window=12, min_touches=3, tolerance=0.01):
    levels = []
    closes = df['Close'].values
    dates = []
    for i in range(window, len(closes) - window):
        local_window = closes[i - window:i + window + 1]
        if closes[i] == min(local_window):
            levels.append(closes[i])
            dates.append(df.index[i])

    grouped_levels = []
    for level in levels:
        if not any(abs(level - x) / x < tolerance for x in grouped_levels):
            grouped_levels.append(level)

    support_levels = []
    for lvl in grouped_levels:
        touches = np.sum(np.abs(closes - lvl) / lvl < tolerance)
        if touches >= min_touches:
            support_levels.append(float(lvl.item()))

    support_series = pd.Series(index=df.index, data=np.nan)
    for date, level in zip(dates, support_levels):
        support_series.loc[date] = level

    return support_series.ffill()

def is_sharp_drop(df, drop_threshold=0.1, full_lookback=30, slope_window=5, slope_threshold=-0.01):
    """
    Checks for a sharp drop based on:
    1. Percentage drop over a longer lookback period (e.g. 30 days)
    2. Slope over recent 'slope_days' (e.g. 3–5 days) being steep enough
    """
    close = df['Close'].iloc[-full_lookback:]
    if len(close) < slope_window + 1:
        return False

    # Compute drop % over the full window
    drop_pct = (close.iloc[0] - close.iloc[-1]) / close.iloc[0]
    best_slope = 0  # Most negative (sharpest down)

    # Scan for any sharp 5-day downward slope in the window
    for i in range(len(close) - slope_window):
        window = close.iloc[i:i + slope_window]
        slope = np.polyfit(range(slope_window), window.values, 1)[0] / window.iloc[0]
        best_slope = min(best_slope, slope.item())

        if drop_pct.item() >= drop_threshold and slope.item() <= slope_threshold:
            print(f"✓ Sharp drop detected: {drop_pct.index[0]} drop_pct={drop_pct.item():.2%}, price near support, slope={slope.item()}")
            return True, best_slope

    return False, None

def find_sharp_decline_to_support(df, support_levels, drop_threshold=0.1, days_lookback=30, proximity_threshold=0.03):
    current_price = df['Close'].iloc[-1]
    recent_high = df['Close'].rolling(window=days_lookback).max().iloc[-1]

    # Force scalars
    if isinstance(current_price, pd.Series):
        current_price = current_price.item()
    if isinstance(recent_high, pd.Series):
        recent_high = recent_high.item()

    drop_pct = (recent_high - current_price) / recent_high
    near_support = any(
        isinstance(level, (float, int)) and abs(current_price - level) / level < proximity_threshold
        for level in support_levels
    )

    return drop_pct >= drop_threshold and near_support

def copy_results_snapshot():
    try:
        import shutil
        shutil.copyfile("results.csv", "results_stable.csv")
        df = pd.read_csv("results_stable.csv")
        df = df.drop_duplicates(subset="Ticker", keep="last")  # or keep="best" logic
        df.to_csv("results_stable.csv", index=False)
        print("📄 Copied snapshot to results_stable.csv")
    except Exception as e:
        print(f"⚠ Failed to copy results.csv: {e}")

import yfinance as yf
import numpy as np
import pandas as pd

# ---------- tiny helper ---------------------------------------------------
def first_value(df: pd.DataFrame, *candidates):
    """
    Return the first non-NaN value among the candidate column names.
    Works on a (dates × columns) DataFrame where the newest date is row 0.
    """
    for c in candidates:
        if c in df.columns:
            v = df[c].iloc[0]
            if pd.notna(v):
                return v
    return np.nan

# ---------- main snapshot -------------------------------------------------
def fundamental_snapshot(ticker: str, *, freq="yearly", update_info=False) -> dict:
    ticker_path = f"ticker_data/{ticker}.pkl"

    # Load from cache if exists
    if os.path.exists(ticker_path):
        with open(ticker_path, "rb") as f:
            cached_data = pkl.load(f)
        inf = cached_data["info"]
        inc = cached_data["income_stmt"]
        cfs = cached_data["cashflow"]
        bal = cached_data["balance_sheet"]
    else:
        t = yf.Ticker(ticker)
        inf = t.get_info()
        inc = t.get_income_stmt(freq=freq).T.sort_index(ascending=False)
        cfs = t.get_cashflow(freq=freq).T.sort_index(ascending=False)
        bal = t.get_balance_sheet(freq=freq).T.sort_index(ascending=False)

        # Only cache what’s serializable
        cached_data = {
            "info": inf,
            "income_stmt": inc,
            "cashflow": cfs,
            "balance_sheet": bal
        }
        with open(ticker_path, "wb") as f:
            pkl.dump(cached_data, f)

    
    market_cap = inf.get("marketCap", 0)

    # # transpose so dates → rows, items → columns
    # inc = t.get_income_stmt(freq=freq).T.sort_index(ascending=False)
    # cfs = t.get_cashflow(freq=freq).T.sort_index(ascending=False)
    # bal = t.get_balance_sheet(freq=freq).T.sort_index(ascending=False)

    revenue = first_value(inc, "TotalRevenue", "Total Revenue")
    op_margin = inf.get("operatingMargins", np.nan)

    # --- Free-cash-flow margin -------------------------------------------
    fcf = first_value(cfs, "FreeCashFlow", "Free Cash Flow")
    if np.isnan(fcf):
        ocf   = first_value(cfs, "OperatingCashFlow",
                                 "Operating Cash Flow",
                                 "CashFlowFromContinuingOperatingActivities")
        capex = abs(first_value(cfs, "CapitalExpenditure", "Capital Expenditure"))
        fcf   = ocf - capex if pd.notna(ocf) and pd.notna(capex) else np.nan
    fcf_margin = fcf / revenue if revenue else np.nan

    # --- Net-debt / EBITDA ----------------------------------------------
    debt = first_value(bal, "TotalDebt", "Total Debt")
    cash = first_value(bal, "CashAndCashEquivalents", "Cash")
    net_debt = debt - cash if pd.notna(debt) and pd.notna(cash) else np.nan
    nde_ratio = net_debt / inf.get("ebitda", np.nan)

    # --- 5-year revenue CAGR --------------------------------------------
    if len(inc) >= 5 and pd.notna(revenue):
        rev_old = inc["TotalRevenue"].iloc[4]
        rev_cagr_5y = (revenue / rev_old) ** (1/5) - 1
    else:
        rev_cagr_5y = np.nan

    fund_score = sum([
        op_margin > 0.08,
        fcf_margin > 0.05,
        nde_ratio < 3,
        rev_cagr_5y > 0.03,
    ])
    return fund_score, market_cap


def start_analysis(set_progress, update_info=False):
    for f in ["results.csv", "results_stable.csv", "progress.txt"]:
        if os.path.exists(f):
            os.remove(f)
    import shutil
    
    if os.path.exists("chart_data"):
        shutil.rmtree("chart_data")
    os.makedirs("chart_data", exist_ok=True)
    os.makedirs("ticker_data", exist_ok=True)
    if os.path.exists("analysis.lock"):
        print("⚠ Analysis already running. Exiting start_analysis.")
        return

    with open("analysis.lock", "w") as f:
        f.write("running")
        print("Starting analysis...")
    # Initialize CSV with header

    with open("results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Ticker", "Current Price", "Support Level", "Proximity %", "RSI", "Market Cap", "Drop %", "Technical Score", "Fundamental Score", "Overall Score"])
        writer.writeheader()

    total = len(tickers)
    for i, ticker in tqdm(enumerate(tickers)):
        try:
            set_progress((i + 1) / total)  # update progress externally
            file_path = f"chart_data/{ticker}.parquet"
            if os.path.exists(file_path):
                last_modified = os.path.getmtime(file_path)
                age_hours = (time.time() - last_modified) / 3600
                if age_hours < 24:
                    df = pd.read_parquet(file_path)
                else:
                    df = yf.download(ticker, period="1y", interval="1d", progress=False)
            else:
                df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df.empty:
                continue
            df['RSI'] = compute_rsi(df['Close'])[ticker].values
            current_rsi = df['RSI'].iloc[-1]
            current_price = float(df['Close'].iloc[-1])
            support_levels = find_support_levels(df)
            fund_score, market_cap = fundamental_snapshot(ticker, update_info=update_info)
            for level in support_levels:
                proximity = (current_price - level) / level
                if 0 < proximity < 0.03 and current_rsi < 40 and current_price > 9:
                    if find_sharp_decline_to_support(df, support_levels) and is_sharp_drop(df):
                        _, slope = is_sharp_drop(df)
                        slope_score = abs(slope) * 1000 if slope is not None else 0  # Scale for interpretability
                        large_cap_bonus = np.log10(market_cap) if market_cap > 0 else 0

                        # Append each result row
                        with open("results.csv", "a", newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=["Ticker", "Current Price", "Support Level", "Proximity %", "RSI", "Market Cap", "Drop %", "Technical Score", "Fundamental Score", "Overall Score"])

                            recent_high = df['Close'].rolling(window=30).max().iloc[-1].item()
                            drop_pct = (recent_high - current_price) / recent_high * 100
                            # Weighting components more intuitively
                            rsi_score = max(0, 40 - current_rsi)  # Higher when RSI is lower, capped at 40
                            drop_score = drop_pct  # Prefer higher drops
                            proximity_score = max(0, 1 - (proximity / 0.03))  # Linear decay to 0 at 3%

                            # Optional: Penalize low-priced tickers
                            price_score = np.log10(current_price) if current_price > 0 else 0

                            # Combine with tuned weights
                            technical_score = (rsi_score * 0.6) + (drop_score * 0.6) + (proximity_score * 0.3) + (
                                        price_score * 0.5) + (slope_score * 1.2) + (large_cap_bonus * 0.2)
                            overall_score = technical_score + fund_score * 25
                            writer.writerow({
                                "Ticker": ticker,
                                "Current Price": round(current_price, 2),
                                "Support Level": round(level, 2),
                                "Proximity %": round(proximity * 100, 2),
                                "RSI": round(current_rsi, 2),
                                "Market Cap": round(market_cap/1_000_000_000, 2),
                                "Drop %": round(drop_pct, 2),
                                "Technical Score": round(technical_score, 2),
                                "Fundamental Score": round(fund_score, 2) * 25,
                                "Overall Score": round(overall_score, 2)
                            })
                            os.makedirs("chart_data", exist_ok=True)

                            # 4-hour data (max 60d)
                            df_4h = yf.download(ticker, period="60d", interval="4h", progress=False)
                            if not df_4h.empty:
                                df_4h.to_parquet(f"chart_data/{ticker}_4h.parquet")
                            df.to_parquet(f"chart_data/{ticker}_1d.parquet")
                            copy_results_snapshot()
        except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

    copy_results_snapshot()
    if os.path.exists("analysis.lock"):
        os.remove("analysis.lock")


DEBUG = False
if DEBUG:
    if os.path.exists("analysis.lock"):
        os.remove("analysis.lock")
    start_analysis(lambda x: print(f"Progress: {x * 100:.2f}%"))