"""
Fetch OHLCV data from Yahoo Finance.

Usage:
    python -m src.data_fetch --config configs/config.yaml
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., "SPY")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    print(f"Fetching {ticker} data from {start_date} to {end_date}...")

    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Reset index to make Date a column
    df = df.reset_index()

    # Handle multi-level columns if present (yfinance sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

    # Select and rename columns
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Convert Date to datetime and sort
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Forward-fill any missing values (shouldn't be many for daily data)
    df = df.ffill()

    # Drop any remaining NaN rows
    df = df.dropna()

    print(f"Downloaded {len(df)} trading days")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df


def save_ohlcv(df: pd.DataFrame, output_path: str) -> None:
    """Save OHLCV data to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV data from Yahoo Finance")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    ticker = config["ticker"]
    start_date = config["start_date"]
    end_date = config["end_date"]

    # Fetch data
    df = fetch_ohlcv(ticker, start_date, end_date)

    # Save to data/raw/{ticker}.csv
    output_path = Path("data/raw") / f"{ticker}.csv"
    save_ohlcv(df, str(output_path))

    # Print summary statistics
    print("\nSummary Statistics:")
    print(df[["Open", "High", "Low", "Close", "Volume"]].describe())


if __name__ == "__main__":
    main()
