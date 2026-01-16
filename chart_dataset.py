"""
Chart-based dataset generation for technical analysis ML research.
Generates candlestick chart images with technical indicators for vision models.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from datetime import datetime
import json
from io import BytesIO
from PIL import Image


# Configuration
WINDOW_SIZE = 60  # Input window: 60 trading days
HORIZON = 5  # Prediction horizon: 5 trading days
IMAGE_SIZE = 224  # Standard CNN input size


def download_spy_data(
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download SPY historical data from Yahoo Finance.

    Args:
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (defaults to today)

    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Downloading SPY data from {start_date} to {end_date}...")
    ticker = yf.Ticker("SPY")
    df = ticker.history(start=start_date, end=end_date, interval="1d")

    if df.empty:
        raise ValueError("No data downloaded for SPY")

    # Reset index to make Date a column, then set it back as index
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.set_index("Date")

    # Keep only OHLCV columns
    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    print(f"Downloaded {len(df)} rows")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for the dataframe.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # SMA 20 (for overlay on chart)
    df["SMA20"] = df["Close"].rolling(window=20).mean()

    # SMA 50 (additional context)
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    # Bollinger Bands (20-period, 2 std)
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * rolling_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * rolling_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_PctB"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # RSI (14-period)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Returns
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(periods=5)
    df["Return_20d"] = df["Close"].pct_change(periods=20)

    # Volatility (20-day rolling std of returns)
    df["Volatility_20d"] = df["Return_1d"].rolling(window=20).std()

    # Volume features
    df["Volume_SMA20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"]

    # Distance from moving averages (normalized)
    df["Dist_SMA20"] = (df["Close"] - df["SMA20"]) / df["SMA20"]
    df["Dist_SMA50"] = (df["Close"] - df["SMA50"]) / df["SMA50"]

    return df


def create_chart_image(
    window_df: pd.DataFrame,
    image_size: int = IMAGE_SIZE,
) -> Image.Image:
    """
    Create a candlestick chart image with technical indicators.

    Args:
        window_df: DataFrame with OHLCV data for the window (must have DatetimeIndex)
        image_size: Output image size (square)

    Returns:
        PIL Image of the chart
    """
    # Prepare data for mplfinance
    plot_df = window_df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Create additional plots for indicators
    add_plots = []

    # SMA 20 overlay on main chart
    if "SMA20" in window_df.columns and not window_df["SMA20"].isna().all():
        add_plots.append(
            mpf.make_addplot(window_df["SMA20"], color="blue", width=1.0)
        )

    # Bollinger Bands
    if "BB_Upper" in window_df.columns and not window_df["BB_Upper"].isna().all():
        add_plots.append(
            mpf.make_addplot(window_df["BB_Upper"], color="gray", width=0.7, linestyle="--")
        )
        add_plots.append(
            mpf.make_addplot(window_df["BB_Lower"], color="gray", width=0.7, linestyle="--")
        )

    # RSI in separate panel
    if "RSI" in window_df.columns and not window_df["RSI"].isna().all():
        add_plots.append(
            mpf.make_addplot(window_df["RSI"], panel=2, color="purple", width=1.0, ylabel="RSI")
        )
        # Add RSI reference lines (30 and 70)
        rsi_30 = pd.Series([30] * len(window_df), index=window_df.index)
        rsi_70 = pd.Series([70] * len(window_df), index=window_df.index)
        add_plots.append(
            mpf.make_addplot(rsi_30, panel=2, color="green", width=0.5, linestyle="--")
        )
        add_plots.append(
            mpf.make_addplot(rsi_70, panel=2, color="red", width=0.5, linestyle="--")
        )

    # Create custom style for clean charts
    mc = mpf.make_marketcolors(
        up="green",
        down="red",
        edge="inherit",
        wick="inherit",
        volume="in",
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle="",
        y_on_right=False,
    )

    # Calculate figure size for desired output resolution
    dpi = 100
    fig_size = (image_size / dpi, image_size / dpi)

    # Create the chart
    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style=style,
        addplot=add_plots if add_plots else None,
        volume=True,
        panel_ratios=(3, 1, 1) if "RSI" in window_df.columns else (3, 1),
        figsize=fig_size,
        returnfig=True,
        tight_layout=True,
        xrotation=0,
        datetime_format="%m-%d",
        show_nontrading=False,
    )

    # Remove axis labels and ticks for cleaner image
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="both", which="both", labelsize=4)

    # Save to buffer
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    buf.seek(0)
    plt.close(fig)

    # Load and resize to exact dimensions
    img = Image.open(buf)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

    return img


def compute_label(df: pd.DataFrame, end_idx: int, horizon: int = HORIZON) -> dict:
    """
    Compute the prediction label for a window.

    Args:
        df: Full DataFrame with price data
        end_idx: Index of the last day in the window
        horizon: Number of days ahead to predict

    Returns:
        Dictionary with label information
    """
    if end_idx + horizon >= len(df):
        return None

    current_price = df.iloc[end_idx]["Close"]
    future_price = df.iloc[end_idx + horizon]["Close"]

    future_return = (future_price / current_price) - 1
    label = 1 if future_return > 0 else 0

    return {
        "label": label,
        "future_return": float(future_return),
        "current_price": float(current_price),
        "future_price": float(future_price),
        "current_date": str(df.index[end_idx].date()),
        "future_date": str(df.index[end_idx + horizon].date()),
    }


def extract_features(window_df: pd.DataFrame) -> dict:
    """
    Extract numerical features for the last day of the window.

    Args:
        window_df: DataFrame for the window

    Returns:
        Dictionary of feature values
    """
    last_row = window_df.iloc[-1]

    features = {
        "close": float(last_row["Close"]),
        "volume": float(last_row["Volume"]),
        "sma20": float(last_row["SMA20"]) if pd.notna(last_row["SMA20"]) else None,
        "sma50": float(last_row["SMA50"]) if pd.notna(last_row["SMA50"]) else None,
        "bb_upper": float(last_row["BB_Upper"]) if pd.notna(last_row["BB_Upper"]) else None,
        "bb_lower": float(last_row["BB_Lower"]) if pd.notna(last_row["BB_Lower"]) else None,
        "bb_pctb": float(last_row["BB_PctB"]) if pd.notna(last_row["BB_PctB"]) else None,
        "rsi": float(last_row["RSI"]) if pd.notna(last_row["RSI"]) else None,
        "macd": float(last_row["MACD"]) if pd.notna(last_row["MACD"]) else None,
        "macd_signal": float(last_row["MACD_Signal"]) if pd.notna(last_row["MACD_Signal"]) else None,
        "return_1d": float(last_row["Return_1d"]) if pd.notna(last_row["Return_1d"]) else None,
        "return_5d": float(last_row["Return_5d"]) if pd.notna(last_row["Return_5d"]) else None,
        "return_20d": float(last_row["Return_20d"]) if pd.notna(last_row["Return_20d"]) else None,
        "volatility_20d": float(last_row["Volatility_20d"]) if pd.notna(last_row["Volatility_20d"]) else None,
        "volume_ratio": float(last_row["Volume_Ratio"]) if pd.notna(last_row["Volume_Ratio"]) else None,
        "dist_sma20": float(last_row["Dist_SMA20"]) if pd.notna(last_row["Dist_SMA20"]) else None,
        "dist_sma50": float(last_row["Dist_SMA50"]) if pd.notna(last_row["Dist_SMA50"]) else None,
    }

    return features


def get_split(date: pd.Timestamp) -> str:
    """
    Determine train/val/test split based on date.

    Train: 2015-2021
    Val: 2022-2023
    Test: 2024+
    """
    year = date.year
    if year <= 2021:
        return "train"
    elif year <= 2023:
        return "val"
    else:
        return "test"


def generate_chart_dataset(
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    output_dir: str = "data/charts",
    window_size: int = WINDOW_SIZE,
    horizon: int = HORIZON,
    image_size: int = IMAGE_SIZE,
    stride: int = 1,
) -> pd.DataFrame:
    """
    Generate the complete chart-based dataset.

    Args:
        start_date: Start date for data
        end_date: End date for data
        output_dir: Directory to save output
        window_size: Number of days in each window
        horizon: Prediction horizon in days
        image_size: Output image size
        stride: Step size between windows (1 = every day)

    Returns:
        DataFrame with metadata for all samples
    """
    output_path = Path(output_dir)

    # Create split directories
    for split in ["train", "val", "test"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)

    # Download and process data
    df = download_spy_data(start_date, end_date)
    df = compute_indicators(df)

    # Drop rows where indicators aren't computed yet (need warmup period)
    # SMA50 needs 50 days, so we start after that
    warmup = 50
    df = df.iloc[warmup:].copy()

    print(f"Data after warmup: {len(df)} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Generate samples
    samples = []
    total_windows = (len(df) - window_size - horizon) // stride + 1

    print(f"\nGenerating {total_windows} chart images...")

    for i in range(0, len(df) - window_size - horizon, stride):
        window_df = df.iloc[i : i + window_size].copy()
        end_idx = i + window_size - 1

        # Get the date of the last day in the window
        window_end_date = df.index[end_idx]
        split = get_split(window_end_date)

        # Compute label
        label_info = compute_label(df, end_idx, horizon)
        if label_info is None:
            continue

        # Extract features
        features = extract_features(window_df)

        # Generate unique sample ID
        sample_id = f"SPY_{window_end_date.strftime('%Y%m%d')}"

        # Create chart image
        img = create_chart_image(window_df, image_size)

        # Save image
        img_path = output_path / split / "images" / f"{sample_id}.png"
        img.save(img_path)

        # Create sample record
        sample = {
            "sample_id": sample_id,
            "split": split,
            "image_path": str(img_path),
            "window_start": str(df.index[i].date()),
            "window_end": str(window_end_date.date()),
            **label_info,
            **features,
        }
        samples.append(sample)

        # Progress update
        if (len(samples)) % 100 == 0:
            print(f"  Generated {len(samples)}/{total_windows} samples...")

    # Create metadata DataFrame
    metadata_df = pd.DataFrame(samples)

    # Save metadata
    metadata_df.to_csv(output_path / "metadata.csv", index=False)

    # Save split-specific metadata
    for split in ["train", "val", "test"]:
        split_df = metadata_df[metadata_df["split"] == split]
        split_df.to_csv(output_path / split / "metadata.csv", index=False)

    # Print statistics
    print("\n" + "=" * 50)
    print("Dataset Generation Complete")
    print("=" * 50)
    print(f"Total samples: {len(metadata_df)}")
    print(f"\nSplit distribution:")
    print(metadata_df["split"].value_counts())
    print(f"\nLabel distribution:")
    print(metadata_df["label"].value_counts())
    print(f"\nLabel distribution by split:")
    print(metadata_df.groupby("split")["label"].value_counts())
    print(f"\nOutput directory: {output_path.absolute()}")

    return metadata_df


if __name__ == "__main__":
    df = generate_chart_dataset()
