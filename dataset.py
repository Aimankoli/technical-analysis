"""
Dataset generation for technical analysis ML research.
Downloads stock data from Yahoo Finance and computes technical indicators.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from pathlib import Path
from datetime import datetime


def download_stock_data(
    symbols: list[str],
    start_date: str = "2010-01-01",
    end_date: str | None = None,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """
    Download historical stock data from Yahoo Finance.

    Args:
        symbols: List of stock ticker symbols
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (defaults to today)
        interval: Data interval (1d, 1h, etc.)

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    data = {}
    for symbol in symbols:
        print(f"Downloading {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        if not df.empty:
            df = df.reset_index()
            data[symbol] = df
            print(f"  Downloaded {len(df)} rows for {symbol}")
        else:
            print(f"  No data found for {symbol}")

    return data


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()

    # Ensure column names are lowercase for ta library
    df.columns = [c.lower() for c in df.columns]

    # Simple Moving Averages
    for period in [5, 10, 20, 50, 200]:
        sma = SMAIndicator(close=df["close"], window=period)
        df[f"sma_{period}"] = sma.sma_indicator()

    # Exponential Moving Averages
    for period in [12, 26, 50]:
        ema = EMAIndicator(close=df["close"], window=period)
        df[f"ema_{period}"] = ema.ema_indicator()

    # MACD
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    # RSI
    for period in [14, 21]:
        rsi = RSIIndicator(close=df["close"], window=period)
        df[f"rsi_{period}"] = rsi.rsi()

    # Stochastic Oscillator
    stoch = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Bollinger Bands
    bb = BollingerBands(close=df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pband"] = bb.bollinger_pband()

    # Average True Range
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"])
    df["atr"] = atr.average_true_range()

    # On-Balance Volume
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
    df["obv"] = obv.on_balance_volume()

    # Price-based features
    df["price_change"] = df["close"].pct_change()
    df["price_change_5d"] = df["close"].pct_change(periods=5)
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_to_high"] = (df["high"] - df["close"]) / df["close"]
    df["close_to_low"] = (df["close"] - df["low"]) / df["close"]

    # Volume features
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    return df


def add_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Add prediction labels based on future price movement.

    Args:
        df: DataFrame with price data
        horizon: Number of periods ahead to look for price movement
        threshold: Minimum percentage change to classify as up/down (0.0 = any movement)

    Returns:
        DataFrame with added label columns
    """
    df = df.copy()

    # Future return
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1

    # Binary label: 1 = price goes up, 0 = price goes down
    if threshold > 0:
        df["label"] = np.where(
            df["future_return"] > threshold, 1,
            np.where(df["future_return"] < -threshold, 0, np.nan)
        )
    else:
        df["label"] = (df["future_return"] > 0).astype(int)

    # Multi-class label for different return ranges
    df["label_multiclass"] = pd.cut(
        df["future_return"],
        bins=[-np.inf, -0.05, -0.02, 0.02, 0.05, np.inf],
        labels=["strong_down", "down", "neutral", "up", "strong_up"]
    )

    return df


def create_dataset(
    symbols: list[str],
    start_date: str = "2010-01-01",
    end_date: str | None = None,
    horizon: int = 5,
    threshold: float = 0.0,
    output_dir: str = "data",
) -> pd.DataFrame:
    """
    Create a complete dataset for ML training.

    Args:
        symbols: List of stock ticker symbols
        start_date: Start date for data
        end_date: End date for data
        horizon: Prediction horizon in days
        threshold: Minimum price change threshold for labeling
        output_dir: Directory to save output files

    Returns:
        Combined DataFrame with all stocks
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Download data
    raw_data = download_stock_data(symbols, start_date, end_date)

    all_data = []
    for symbol, df in raw_data.items():
        print(f"Processing {symbol}...")

        # Add indicators
        df = add_technical_indicators(df)

        # Add labels
        df = add_labels(df, horizon=horizon, threshold=threshold)

        # Add symbol column
        df["symbol"] = symbol

        # Save individual stock data
        df.to_csv(output_path / f"{symbol}.csv", index=False)
        print(f"  Saved {symbol}.csv")

        all_data.append(df)

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Drop rows with NaN labels (last `horizon` rows won't have labels)
    combined_df = combined_df.dropna(subset=["label"])

    # Save combined dataset
    combined_df.to_csv(output_path / "combined_dataset.csv", index=False)
    print(f"\nSaved combined dataset with {len(combined_df)} rows")

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"  Label distribution:")
    print(combined_df["label"].value_counts(normalize=True))

    return combined_df


# Default symbols for S&P 500 representative sample
DEFAULT_SYMBOLS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Finance
    "JPM", "BAC", "GS", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV",
    # Consumer
    "WMT", "KO", "PG", "MCD",
    # Industrial
    "CAT", "BA", "GE",
    # Energy
    "XOM", "CVX",
    # ETFs for broader market
    "SPY", "QQQ",
]


if __name__ == "__main__":
    # Create dataset with default parameters
    df = create_dataset(
        symbols=DEFAULT_SYMBOLS,
        start_date="2015-01-01",
        horizon=5,  # Predict 5 days ahead
        threshold=0.0,  # Any movement counts
    )
