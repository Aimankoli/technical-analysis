"""
Build sample windows and render candlestick chart images.

IMPORTANT:
- No titles, no tick labels, no date annotations on charts.
- y-axis scaling uses ONLY min/max within the window (padded by 2%).
- Images are resized to (image_size, image_size).

Usage:
    python -m src.render_charts --config configs/config.yaml
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def assign_split(date: pd.Timestamp, config: dict) -> Optional[str]:
    """Assign a sample to train/val/test split based on its end date."""
    date = pd.Timestamp(date)

    train_start = pd.Timestamp(config["train_start"])
    train_end = pd.Timestamp(config["train_end"])
    val_start = pd.Timestamp(config["val_start"])
    val_end = pd.Timestamp(config["val_end"])
    test_start = pd.Timestamp(config["test_start"])
    test_end = pd.Timestamp(config["test_end"])

    if train_start <= date <= train_end:
        return "train"
    elif val_start <= date <= val_end:
        return "val"
    elif test_start <= date <= test_end:
        return "test"
    else:
        return None


def build_samples(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Build sample windows with labels.

    For each valid index t:
    - window = [t-lookback_days+1 ... t]
    - future = [t+1 ... t+horizon_days]
    - label y = 1 if (Close[t+horizon]/Close[t] - 1) > 0 else 0

    Returns DataFrame with columns:
        sample_id, start_idx, end_idx, start_date, end_date, label, forward_return, split
    """
    lookback = config["lookback_days"]
    horizon = config["horizon_days"]
    stride = config.get("stride", 1)

    samples = []
    sample_id = 0

    # Need at least lookback days before and horizon days after
    for t in range(lookback - 1, len(df) - horizon):
        if (t - lookback + 1) % stride != 0:
            continue

        end_date = df.iloc[t]["Date"]
        start_date = df.iloc[t - lookback + 1]["Date"]

        # Compute label and forward return
        close_t = df.iloc[t]["Close"]
        close_future = df.iloc[t + horizon]["Close"]
        forward_return = (close_future / close_t) - 1
        label = 1 if forward_return > 0 else 0

        # Assign split
        split = assign_split(end_date, config)
        if split is None:
            continue  # Skip samples outside defined splits

        samples.append({
            "sample_id": sample_id,
            "start_idx": t - lookback + 1,
            "end_idx": t,
            "start_date": start_date,
            "end_date": end_date,
            "label": label,
            "forward_return": forward_return,
            "split": split
        })
        sample_id += 1

    return pd.DataFrame(samples)


def render_candlestick_chart(
    window_df: pd.DataFrame,
    config: dict,
    output_path: str
) -> None:
    """
    Render a candlestick chart with overlays.

    IMPORTANT: No titles, no axis labels, no date text.
    y-axis scaling uses only data within the window.
    """
    image_size = config["image_size"]
    include_volume = config.get("include_volume", True)
    include_rsi = config.get("include_rsi_panel", True)
    overlay_sma = config.get("overlay_sma20", True)
    overlay_bb = config.get("overlay_bollinger", True)

    # Determine subplot layout
    n_panels = 1
    height_ratios = [3]
    if include_volume:
        n_panels += 1
        height_ratios.append(1)
    if include_rsi:
        n_panels += 1
        height_ratios.append(1)

    # Create figure with tight layout
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(image_size / 20, image_size / 20),  # Reasonable base size
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.02},
        squeeze=False
    )
    axes = axes.flatten()

    # Colors
    up_color = "#26a69a"    # Green
    down_color = "#ef5350"  # Red
    sma_color = "#ff9800"   # Orange
    bb_color = "#2196f3"    # Blue

    # Price panel (candlesticks)
    ax_price = axes[0]

    # Get price data
    opens = window_df["Open"].values
    highs = window_df["High"].values
    lows = window_df["Low"].values
    closes = window_df["Close"].values

    # Y-axis scaling with 2% padding
    price_min = min(lows.min(), window_df["bb_lower"].min() if "bb_lower" in window_df else lows.min())
    price_max = max(highs.max(), window_df["bb_upper"].max() if "bb_upper" in window_df else highs.max())
    price_range = price_max - price_min
    y_min = price_min - 0.02 * price_range
    y_max = price_max + 0.02 * price_range

    # Draw candlesticks
    width = 0.6
    for i in range(len(window_df)):
        color = up_color if closes[i] >= opens[i] else down_color

        # Wick (high-low line)
        ax_price.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.8)

        # Body (open-close rectangle)
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i])
        rect = mpatches.Rectangle(
            (i - width / 2, body_bottom),
            width, body_height,
            facecolor=color, edgecolor=color
        )
        ax_price.add_patch(rect)

    # Overlay SMA20
    if overlay_sma and "sma20" in window_df:
        sma_values = window_df["sma20"].values
        valid_mask = ~np.isnan(sma_values)
        if valid_mask.any():
            x_vals = np.arange(len(window_df))[valid_mask]
            ax_price.plot(x_vals, sma_values[valid_mask], color=sma_color, linewidth=1, alpha=0.8)

    # Overlay Bollinger Bands
    if overlay_bb and "bb_upper" in window_df and "bb_lower" in window_df:
        bb_upper = window_df["bb_upper"].values
        bb_lower = window_df["bb_lower"].values
        valid_mask = ~(np.isnan(bb_upper) | np.isnan(bb_lower))
        if valid_mask.any():
            x_vals = np.arange(len(window_df))[valid_mask]
            ax_price.plot(x_vals, bb_upper[valid_mask], color=bb_color, linewidth=0.8, alpha=0.6)
            ax_price.plot(x_vals, bb_lower[valid_mask], color=bb_color, linewidth=0.8, alpha=0.6)
            ax_price.fill_between(
                x_vals, bb_lower[valid_mask], bb_upper[valid_mask],
                color=bb_color, alpha=0.1
            )

    ax_price.set_xlim(-0.5, len(window_df) - 0.5)
    ax_price.set_ylim(y_min, y_max)
    ax_price.axis("off")

    current_axis_idx = 1

    # Volume panel
    if include_volume:
        ax_vol = axes[current_axis_idx]
        volumes = window_df["Volume"].values
        colors = [up_color if closes[i] >= opens[i] else down_color for i in range(len(window_df))]

        ax_vol.bar(range(len(window_df)), volumes, color=colors, width=0.7)
        ax_vol.set_xlim(-0.5, len(window_df) - 0.5)
        ax_vol.axis("off")
        current_axis_idx += 1

    # RSI panel
    if include_rsi and "rsi14" in window_df:
        ax_rsi = axes[current_axis_idx]
        rsi_values = window_df["rsi14"].values
        valid_mask = ~np.isnan(rsi_values)

        if valid_mask.any():
            x_vals = np.arange(len(window_df))[valid_mask]
            ax_rsi.plot(x_vals, rsi_values[valid_mask], color="#9c27b0", linewidth=1)

            # Overbought/oversold lines (no labels)
            ax_rsi.axhline(y=70, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
            ax_rsi.axhline(y=30, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
            ax_rsi.axhline(y=50, color="gray", linewidth=0.3, linestyle="-", alpha=0.3)

        ax_rsi.set_xlim(-0.5, len(window_df) - 0.5)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.axis("off")

    # Remove all margins
    plt.tight_layout(pad=0)

    # Save and resize
    temp_path = output_path + ".tmp.png"
    fig.savefig(
        temp_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
        edgecolor="none"
    )
    plt.close(fig)

    # Resize to exact image_size x image_size
    img = Image.open(temp_path)
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    img.save(output_path)
    os.remove(temp_path)


def main():
    parser = argparse.ArgumentParser(description="Build samples and render chart images")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip rendering images that already exist")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    ticker = config["ticker"]

    # Load data with indicators
    input_path = Path("data/raw") / f"{ticker}_indicators.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Indicators file not found at {input_path}. Run features.py first."
        )

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=["Date"])

    # Build samples
    print("Building sample windows...")
    samples_df = build_samples(df, config)
    print(f"Created {len(samples_df)} samples")

    # Print split distribution
    split_counts = samples_df["split"].value_counts()
    print("\nSplit distribution:")
    for split, count in split_counts.items():
        label_dist = samples_df[samples_df["split"] == split]["label"].value_counts()
        up_pct = label_dist.get(1, 0) / count * 100
        print(f"  {split}: {count} samples ({up_pct:.1f}% up)")

    # Save samples metadata
    samples_path = Path("data/samples/samples.parquet")
    os.makedirs(samples_path.parent, exist_ok=True)
    samples_df.to_parquet(samples_path, index=False)
    print(f"\nSaved samples to {samples_path}")

    # Also save as CSV for easy inspection
    samples_csv_path = Path("data/samples/samples.csv")
    samples_df.to_csv(samples_csv_path, index=False)

    # Render chart images
    images_dir = Path("data/images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"\nRendering {len(samples_df)} chart images...")
    for _, sample in tqdm(samples_df.iterrows(), total=len(samples_df)):
        sample_id = sample["sample_id"]
        output_path = images_dir / f"{sample_id}.png"

        if args.skip_existing and output_path.exists():
            continue

        # Extract window data
        start_idx = sample["start_idx"]
        end_idx = sample["end_idx"]
        window_df = df.iloc[start_idx:end_idx + 1].copy()

        render_candlestick_chart(window_df, config, str(output_path))

    print(f"Done! Images saved to {images_dir}")


if __name__ == "__main__":
    main()
