"""
Compute technical indicators for OHLCV data.

All indicators are computed using ONLY past values (no lookahead).

Usage:
    python -m src.features --config configs/config.yaml
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Compute Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, window: int) -> pd.Series:
    """Compute Exponential Moving Average."""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def compute_bollinger_bands(
    series: pd.Series, window: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands.

    Returns:
        Tuple of (middle_band/SMA, upper_band, lower_band)
    """
    sma = compute_sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = sma + k * std
    lower = sma - k * std
    return sma, upper, lower


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    RS = average gain / average loss over window
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Use exponential moving average for smoothing (Wilder's method)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_returns(series: pd.Series, periods: int) -> pd.Series:
    """Compute percentage returns over given periods."""
    return series.pct_change(periods=periods)


def compute_trend_slope(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Compute trend slope using linear regression over rolling window.

    Returns the slope coefficient normalized by price level.
    """
    def _slope(x):
        if len(x) < window or np.isnan(x).any():
            return np.nan
        y = np.array(x)
        x_vals = np.arange(len(y))
        # Linear regression: y = mx + b
        # m = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        n = len(y)
        sum_x = x_vals.sum()
        sum_y = y.sum()
        sum_xy = (x_vals * y).sum()
        sum_x2 = (x_vals ** 2).sum()

        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        # Normalize by mean price
        return slope / y.mean() if y.mean() != 0 else 0.0

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def compute_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute all technical indicators for the OHLCV data.

    Args:
        df: DataFrame with Date, Open, High, Low, Close, Volume columns
        config: Configuration dictionary

    Returns:
        DataFrame with original columns plus indicator columns
    """
    df = df.copy()

    bb_window = config.get("bb_window", 20)
    bb_k = config.get("bb_k", 2)
    rsi_window = config.get("rsi_window", 14)

    close = df["Close"]

    # SMA20
    df["sma20"] = compute_sma(close, window=20)

    # Bollinger Bands
    _, df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(
        close, window=bb_window, k=bb_k
    )

    # RSI
    df["rsi14"] = compute_rsi(close, window=rsi_window)

    # Returns
    df["return_1d"] = compute_returns(close, periods=1)
    df["return_5d"] = compute_returns(close, periods=5)
    df["return_20d"] = compute_returns(close, periods=20)

    # Volatility (20-day rolling std of returns)
    df["volatility_20d"] = df["return_1d"].rolling(window=20, min_periods=20).std()

    # Bollinger %B: position within the bands
    # %B = (Close - Lower) / (Upper - Lower)
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_percent_b"] = (close - df["bb_lower"]) / bb_range.replace(0, np.nan)

    # Distance from SMA20 (normalized)
    df["distance_to_sma20"] = (close - df["sma20"]) / df["sma20"]

    # Trend slope (10-day linear regression slope)
    df["trend_slope_10d"] = compute_trend_slope(close, window=10)

    # Volume z-score (20-day)
    vol_mean = df["Volume"].rolling(window=20, min_periods=20).mean()
    vol_std = df["Volume"].rolling(window=20, min_periods=20).std()
    df["volume_zscore"] = (df["Volume"] - vol_mean) / vol_std.replace(0, np.nan)

    return df


def save_indicators(df: pd.DataFrame, output_path: str) -> None:
    """Save indicators DataFrame to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved indicators to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute technical indicators")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    ticker = config["ticker"]

    # Load raw OHLCV data
    input_path = Path("data/raw") / f"{ticker}.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {input_path}. Run data_fetch.py first."
        )

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=["Date"])

    # Compute indicators
    print("Computing technical indicators...")
    df_indicators = compute_all_indicators(df, config)

    # Save
    output_path = Path("data/raw") / f"{ticker}_indicators.csv"
    save_indicators(df_indicators, str(output_path))

    # Print summary
    print("\nIndicator columns added:")
    indicator_cols = [
        "sma20", "bb_upper", "bb_lower", "rsi14",
        "return_1d", "return_5d", "return_20d", "volatility_20d",
        "bb_percent_b", "distance_to_sma20", "trend_slope_10d", "volume_zscore"
    ]
    for col in indicator_cols:
        valid_count = df_indicators[col].notna().sum()
        print(f"  {col}: {valid_count} valid values")

    print(f"\nTotal rows: {len(df_indicators)}")
    print(f"Rows with all indicators valid: {df_indicators[indicator_cols].dropna().shape[0]}")


if __name__ == "__main__":
    main()
