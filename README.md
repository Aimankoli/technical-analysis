# LLM-Augmented Technical Analysis

A research project to evaluate whether technical analysis (chart patterns + indicators) can predict stock price movements using machine learning.

## Overview

This project implements a multi-modal fusion model that combines:
1. **Candlestick chart images** with SMA20, Bollinger Bands, and RSI overlays
2. **LLM-generated technical analysis** embedded via FinBERT
3. **Numeric indicators** (returns, volatility, RSI, etc.)

The model predicts whether price will go up or down over the next 5 trading days.

## Project Structure

```
.
├── configs/
│   └── config.yaml          # All hyperparameters and settings
├── data/
│   ├── raw/                  # OHLCV data and indicators
│   ├── samples/              # Sample metadata (parquet)
│   ├── images/               # Rendered chart images
│   ├── llm/                  # Cached LLM outputs
│   ├── embeddings/           # FinBERT embeddings
│   └── models/               # Trained models and results
├── src/
│   ├── data_fetch.py         # Download OHLCV from Yahoo Finance
│   ├── features.py           # Compute technical indicators
│   ├── render_charts.py      # Build samples and render charts
│   ├── llm_analyze.py        # Generate LLM technical analysis
│   ├── finbert_embed.py      # Create FinBERT embeddings
│   ├── dataset.py            # PyTorch dataset loader
│   ├── model.py              # Fusion model architecture
│   ├── train.py              # Training script
│   ├── eval.py               # Evaluation metrics
│   └── backtest.py           # Trading strategy backtest
├── notebooks/
│   └── colab_demo.ipynb      # Interactive demo
├── run_pipeline.py           # Run the complete pipeline
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Run everything (skip LLM if Ollama not available)
python run_pipeline.py --config configs/config.yaml --skip-llm

# Or run steps individually:
python -m src.data_fetch --config configs/config.yaml
python -m src.features --config configs/config.yaml
python -m src.render_charts --config configs/config.yaml
python -m src.finbert_embed --config configs/config.yaml
python -m src.train --config configs/config.yaml
python -m src.eval --config configs/config.yaml
python -m src.backtest --config configs/config.yaml
```

## Key Design Decisions

### No Data Leakage
- All indicators computed using only past data
- Strict time-based train/val/test splits
- Chart images contain no dates, tickers, or future information
- y-axis scaling uses only data within the lookback window

### Model Architecture
- **Image branch**: ResNet18 pretrained on ImageNet, outputs 512-dim features
- **Text branch**: MLP (768 -> 256 -> 128) for FinBERT embeddings
- **Numeric branch**: MLP (8 -> 64 -> 32) for indicators
- **Fusion**: Concatenate + MLP head for binary classification

### Backtest Strategy
- Entry: if P(up) > 0.55, go long
- Exit: after 5 trading days (holding period)
- No shorting (MVP simplification)
- Transaction cost: 10 bps per round-trip

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
ticker: "SPY"
lookback_days: 30
horizon_days: 5
image_size: 112

# Date splits
train_start: "2015-01-01"
train_end: "2017-12-31"
val_start: "2018-01-01"
val_end: "2018-12-31"
test_start: "2019-01-01"
test_end: "2019-12-31"

# Model
use_text_embeddings: true
use_numeric_features: true
cnn_backbone: "resnet18"
```

## LLM Analysis (Optional)

To enable LLM-generated technical analysis:

1. Install and run Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.1:8b-instruct`
3. Run: `python -m src.llm_analyze --config configs/config.yaml`

The LLM generates structured JSON analysis including:
- Trend direction (bullish/bearish/sideways)
- Momentum strength
- Volatility assessment
- Support/resistance levels
- Summary text

## Ablation Studies

To test the contribution of each modality, modify `config.yaml`:

```yaml
# CNN only
use_text_embeddings: false
use_numeric_features: false

# Text only
use_text_embeddings: true
use_numeric_features: false
# (requires custom model that skips image branch)

# Full fusion
use_text_embeddings: true
use_numeric_features: true
```

## Outputs

After running the pipeline:

- `data/models/best.pt` - Trained model checkpoint
- `data/models/metrics.json` - Training metrics
- `data/models/eval_results.json` - Test set evaluation
- `data/models/backtest_results.json` - Backtest performance

## References

This project is inspired by research on using visual representations of financial data for prediction, including:
- CNN-based approaches to candlestick chart analysis
- Multi-modal fusion of visual and textual financial data
- LLM applications in financial analysis
