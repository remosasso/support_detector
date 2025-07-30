import time
import yfinance as yf
import pandas as pd
import ta
import os
from tqdm import tqdm

def copy_results_snapshot():
    """Copy results to stable file for dashboard reading using atomic writes"""
    try:
        import shutil
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(suffix=".csv", prefix="tmp_swing_results_")
        os.close(temp_fd)

        shutil.copyfile("swing_results_ranked.csv", temp_path)

        df = pd.read_csv(temp_path)
        df = df.drop_duplicates(subset="Ticker", keep="last")
        df.to_csv(temp_path, index=False)

        os.replace(temp_path, "swing_results_stable.csv")
        print("✅ Atomically saved swing_results_stable.csv successfully.")
    except Exception as e:
        print(f"⚠ Failed to copy swing_results.csv: {e}")


# --- Ticker Loading ---
if os.path.exists("liquid_big_tickers.txt"):
    with open("liquid_big_tickers.txt", "r") as f:
        tickers = list(set([line.strip() for line in f.readlines()]))
else:
    from tickers import us_ticker_dict, eu_ticker_dict
    tickers = list(us_ticker_dict.keys()) + list(eu_ticker_dict.keys())

tickers = list(set(tickers))  # Remove duplicates
tickers = [
    "IT", "NTRA", "CHTR", "CMG", "COST", "DUOL", "FI", "CHWY", "HCA", "SUI",
    "STM", "EFX", "GDDY", "BIP", "LULU", "CNI", "INTC", "DOC", "TXN", "OTIS",
    "ELS", "BRO", "SAP", "UMC", "TECK", "BUD", "NOK", "TXT", "CP", "EHC", "WAB",
    "INVH", "MMC", "CCK", "RGLD", "LUV", "HON", "CNC", "SWKS", "PM", "AMH",
    "CMCSA", "FMS", "RS", "IBM", "INFY", "GLPI", "AR", "ERIC", "BSBR", "COR",
    "CVS", "SBAC", "EXE", "CB", "6857.T", "KO", "KR", "RELX", "EBR", "BBD",
    "CL", "DG", "PNW", "HRL", "LMT", "TAK", "ACI", "SBS", "FMX",    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "JNJ", "V",
    "PG", "UNH", "HD", "MA", "COST", "PEP", "AVGO", "ADBE", "CRM", "WMT",
    "BAC", "MCD", "KO", "PFE", "DIS", "CSCO", "NFLX", "TMO", "ACN", "ABT",
    "HON", "IBM", "SBUX", "CAT", "GS", "DE", "INTC", "AMD", "QCOM", "TXN"
]
# --- Parameters ---
RSI_THRESHOLD = 35
SUPPORT_DISTANCE_THRESHOLD = 0.03
DROP_5D_THRESHOLD = -0.05
VOLUME_THRESHOLD = 200_000
MARKET_CAP_THRESHOLD = 10_000_000_000  # $10B

# --- Main Analysis Function ---

# Main
def analyze_longterm_support_dips(set_progress):
    results = []
    total = len(tickers)
    lock_filename = "swing_analysis.lock"
    if os.path.exists(lock_filename):
        print("⚠ Analysis already running. Exiting.")
        return
    with open(lock_filename, "w") as f:
        f.write("running")

    for i, ticker in enumerate(tickers):
        print(f"Analyzing {ticker} ({i + 1}/{total})")
        set_progress((i + 1) / total)

        try:
            while True:
                try:
                    yf_ticker = yf.Ticker(ticker)
                    info = yf_ticker.info
                    break
                except Exception as e:
                    if "401" in str(e):
                        print("Retrying due to 401 error...")
                        time.sleep(60)
                        continue
                    else:
                        print(f"⚠ Error fetching info for {ticker}: {e}")
                        raise e
            market_cap = info.get("marketCap", 0)
            avg_vol = info.get("averageVolume", 0)
            if market_cap < MARKET_CAP_THRESHOLD or avg_vol < VOLUME_THRESHOLD:
                continue

            df = yf_ticker.history(period="1y", interval="1d", auto_adjust=True)
            if df.empty or len(df) < 180:
                continue

            df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
            df["Support_180"] = df["Close"].rolling(window=180).min()
            df.dropna(inplace=True)

            latest = df.iloc[-1]
            rsi = latest["RSI"]
            close = latest["Close"]
            support = latest["Support_180"]
            dist_to_support = (close - support) / support if support > 0 else 1.0
            change_5d = (close - df["Close"].iloc[-6]) / df["Close"].iloc[-6]
            if (
                rsi < RSI_THRESHOLD and
                dist_to_support < SUPPORT_DISTANCE_THRESHOLD and
                change_5d < DROP_5D_THRESHOLD
            ):
                results.append({
                    "Ticker": ticker,
                    "Close": round(close, 2),
                    "RSI": round(rsi, 2),
                    "Dist_to_LongSupport_%": round(dist_to_support * 100, 2),
                    "5D_Change_%": round(change_5d * 100, 2),
                    "MarketCapRaw": market_cap,
                    "MarketCap": f"${market_cap / 1e9:.1f}B",
                    "Volume": info.get("averageVolume", 0)
                })
                print("Found setup:", ticker, "RSI:", rsi, "Dist to Support:", round(dist_to_support * 100, 2), "%")

                df_out = pd.DataFrame(results)

                # Tier by RSI
                df_out["RSI_Tier"] = pd.cut(
                    df_out["RSI"],
                    bins=[-1, 20, 25, 35],
                    labels=["A (Extreme)", "B (Oversold)", "C (Weak Dip)"]
                )

                # Tier by Market Cap
                def cap_class(cap):
                    if cap >= 100e9:
                        return "MegaCap"
                    elif cap >= 10e9:
                        return "LargeCap"
                    elif cap >= 2e9:
                        return "MidCap"
                    else:
                        return "SmallCap"

                df_out["MarketCapTier"] = df_out["MarketCapRaw"].apply(cap_class)

                df_out = df_out.sort_values(by="RSI").drop(columns=["MarketCapRaw"])
                # Save results to CSV
                df_out.to_csv("swing_results_ranked.csv", index=False)
                copy_results_snapshot()

        except Exception as e:
            print(f"⚠ Error analyzing {ticker}: {e}")
            continue

    df_out["MarketCapTier"] = df_out["MarketCapRaw"].apply(cap_class)

    df_out = df_out.sort_values(by="RSI").drop(columns=["MarketCapRaw"])
    print(f"✅ Analysis complete. {len(df_out)} setups saved.")
    print(df_out.head(10))
    # Save results to CSV
    df_out.to_csv("swing_results_ranked.csv", index=False)
    print("trying save")
    copy_results_snapshot()

if __name__ == "__main__":
    def progress(x): print(f"Progress: {x * 100:.2f}%")
    analyze_longterm_support_dips(progress)