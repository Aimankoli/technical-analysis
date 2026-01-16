"""
Main pipeline script that runs the complete workflow.

Usage:
    python run_pipeline.py --config configs/config.yaml

This will:
1. Fetch OHLCV data (if not cached)
2. Compute technical indicators
3. Build sample windows and render chart images
4. Generate LLM analysis (optional, requires Ollama)
5. Generate FinBERT embeddings
6. Train the fusion model
7. Evaluate on test set
8. Run backtest

Note: LLM analysis step requires Ollama running locally or can be skipped.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_step(name: str, command: list[str]) -> bool:
    """Run a pipeline step and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}\n")

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"\nERROR: {name} failed with code {result.returncode}")
        return False

    print(f"\n{name} completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the complete pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM analysis step (uses dummy embeddings)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip steps if output already exists")
    parser.add_argument("--gemini-key", type=str, default=None,
                        help="Google API key for Gemini (or set GOOGLE_API_KEY env var)")
    args = parser.parse_args()

    config = load_config(args.config)
    ticker = config["ticker"]

    # Set Gemini API key if provided
    if args.gemini_key:
        os.environ["GOOGLE_API_KEY"] = args.gemini_key

    print("\n" + "=" * 60)
    print("LLM-Augmented Technical Analysis Pipeline")
    print("=" * 60)
    print(f"\nConfig: {args.config}")
    print(f"Ticker: {ticker}")
    print(f"Date range: {config['start_date']} to {config['end_date']}")

    # Step 1: Fetch OHLCV data
    raw_path = Path(f"data/raw/{ticker}.csv")
    if args.skip_existing and raw_path.exists():
        print(f"\nSkipping data fetch - {raw_path} exists")
    else:
        if not run_step("Fetch OHLCV Data", f"{sys.executable} -m src.data_fetch --config {args.config}"):
            return 1

    # Step 2: Compute indicators
    indicators_path = Path(f"data/raw/{ticker}_indicators.csv")
    if args.skip_existing and indicators_path.exists():
        print(f"\nSkipping indicators - {indicators_path} exists")
    else:
        if not run_step("Compute Indicators", f"{sys.executable} -m src.features --config {args.config}"):
            return 1

    # Step 3: Build samples and render charts
    samples_path = Path("data/samples/samples.parquet")
    images_exist = len(list(Path("data/images").glob("*.png"))) > 0 if Path("data/images").exists() else False

    if args.skip_existing and samples_path.exists() and images_exist:
        print(f"\nSkipping chart rendering - samples and images exist")
    else:
        skip_flag = "--skip-existing" if args.skip_existing else ""
        if not run_step("Build Samples & Render Charts",
                       f"{sys.executable} -m src.render_charts --config {args.config} {skip_flag}"):
            return 1

    # Step 4: LLM Analysis (optional)
    if args.skip_llm:
        print("\nSkipping LLM analysis (--skip-llm flag)")
    else:
        cache_path = Path(config.get("cache_path", "data/llm/analysis.jsonl"))
        if args.skip_existing and cache_path.exists():
            print(f"\nSkipping LLM analysis - {cache_path} exists")
        else:
            print("\nNote: LLM analysis requires Ollama running locally.")
            print("If Ollama is not available, use --skip-llm flag.")
            if not run_step("LLM Analysis", f"{sys.executable} -m src.llm_analyze --config {args.config}"):
                print("\nLLM analysis failed. You can retry with --skip-llm to skip this step.")
                # Don't fail the whole pipeline for this optional step
                pass

    # Step 5: FinBERT Embeddings
    embeddings_path = Path("data/embeddings/finbert.npy")
    if args.skip_existing and embeddings_path.exists():
        print(f"\nSkipping FinBERT embeddings - {embeddings_path} exists")
    else:
        if not run_step("FinBERT Embeddings", f"{sys.executable} -m src.finbert_embed --config {args.config}"):
            # Create dummy embeddings if FinBERT fails (e.g., no GPU)
            print("\nCreating dummy embeddings...")
            import numpy as np
            import pandas as pd
            import json

            samples_df = pd.read_parquet("data/samples/samples.parquet")
            n_samples = len(samples_df)
            embeddings = np.zeros((n_samples, 768), dtype=np.float32)

            os.makedirs("data/embeddings", exist_ok=True)
            np.save("data/embeddings/finbert.npy", embeddings)

            mapping = {sid: idx for idx, sid in enumerate(samples_df["sample_id"])}
            with open("data/embeddings/sample_id_to_idx.json", "w") as f:
                json.dump(mapping, f)

            print("Created dummy embeddings (zeros)")

    # Step 6: Train model
    model_path = Path("data/models/best.pt")
    if args.skip_existing and model_path.exists():
        print(f"\nSkipping training - {model_path} exists")
    else:
        if not run_step("Train Model", f"{sys.executable} -m src.train --config {args.config}"):
            return 1

    # Step 7: Evaluate
    if not run_step("Evaluate Model", f"{sys.executable} -m src.eval --config {args.config}"):
        return 1

    # Step 8: Backtest
    if not run_step("Run Backtest", f"{sys.executable} -m src.backtest --config {args.config}"):
        return 1

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    # Load and print key results
    import json
    results_path = Path("data/models/backtest_results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        print("\n📊 Key Results:")
        strat = results["strategy"]
        print(f"  Test AUC:            (see data/models/metrics.json)")
        print(f"  Sharpe Ratio:        {strat['sharpe']:.2f}")
        print(f"  CAGR:                {strat['cagr']*100:.2f}%")
        print(f"  Max Drawdown:        {strat['max_drawdown']*100:.2f}%")
        print(f"  Hit Rate:            {strat['hit_rate']*100:.2f}%")
        print(f"  Trades:              {strat['num_trades']}")

    print("\n📁 Output Files:")
    print("  - data/models/best.pt         (trained model)")
    print("  - data/models/metrics.json    (training metrics)")
    print("  - data/models/eval_results.json (evaluation metrics)")
    print("  - data/models/backtest_results.json (backtest results)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
