"""
Backtesting script for the trading strategy.

Strategy:
- If prob_up > threshold: go long for horizon_days then exit
- Else: stay flat (no short for MVP)
- Apply transaction costs (10 bps per trade round-trip)

Metrics:
- Equity curve
- CAGR
- Sharpe ratio
- Max drawdown
- Hit rate

Usage:
    python -m src.backtest --config configs/config.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_predictions(predictions_path: str) -> pd.DataFrame:
    """Load predictions from JSON file."""
    with open(predictions_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame({
        "sample_id": data["sample_ids"],
        "end_date": pd.to_datetime(data["end_dates"]),
        "prob_up": data["probs"],
        "label": data["labels"],
        "forward_return": data["forward_returns"],
    })

    return df.sort_values("end_date").reset_index(drop=True)


def compute_cagr(equity_curve: np.ndarray, trading_days_per_year: int = 252) -> float:
    """Compute Compound Annual Growth Rate."""
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0

    total_return = equity_curve[-1] / equity_curve[0]
    n_days = len(equity_curve)
    n_years = n_days / trading_days_per_year

    if n_years <= 0 or total_return <= 0:
        return 0.0

    cagr = (total_return ** (1 / n_years)) - 1
    return cagr


def compute_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0, trading_days_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / trading_days_per_year
    sharpe = np.sqrt(trading_days_per_year) * excess_returns.mean() / excess_returns.std()
    return sharpe


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute maximum drawdown."""
    if len(equity_curve) < 2:
        return 0.0

    cummax = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - cummax) / cummax
    return abs(drawdowns.min())


def run_backtest(
    predictions_df: pd.DataFrame,
    entry_threshold: float = 0.55,
    horizon_days: int = 5,
    transaction_cost_bps: float = 10.0,
    initial_capital: float = 10000.0,
) -> dict:
    """
    Run a simple walk-forward backtest.

    Strategy:
    - On each day, if prob_up > threshold and not already in position, enter long
    - Hold for horizon_days, then exit
    - Track equity curve and compute performance metrics
    """
    transaction_cost = transaction_cost_bps / 10000.0  # Convert bps to decimal

    # Sort by date
    df = predictions_df.sort_values("end_date").reset_index(drop=True)

    # Track positions and equity
    equity = initial_capital
    equity_curve = [equity]
    daily_returns = []

    position = None  # (entry_date, entry_price_equiv, days_held)
    trades = []

    # We'll use forward_return directly as the return for holding horizon_days
    # This is already computed in the sample generation

    for idx, row in df.iterrows():
        current_date = row["end_date"]
        prob_up = row["prob_up"]
        forward_return = row["forward_return"]
        label = row["label"]

        # Check if we should exit existing position
        if position is not None:
            position_days_held = position["days_held"] + 1
            if position_days_held >= horizon_days:
                # Exit position
                # Use the forward return from entry date
                trade_return = position["forward_return"]
                # Apply exit transaction cost
                trade_return -= transaction_cost

                equity *= (1 + trade_return)
                daily_returns.append(trade_return)

                trades.append({
                    "entry_date": str(position["entry_date"]),
                    "exit_date": str(current_date),
                    "forward_return": position["forward_return"],
                    "net_return": trade_return,
                    "signal": position["signal"],
                    "actual_label": position["label"],
                    "hit": 1 if (trade_return > 0) else 0,
                })

                position = None
            else:
                position["days_held"] = position_days_held

        # Check if we should enter new position
        if position is None and prob_up > entry_threshold:
            # Enter long position
            # Apply entry transaction cost
            equity *= (1 - transaction_cost)
            position = {
                "entry_date": current_date,
                "days_held": 0,
                "forward_return": forward_return,
                "signal": prob_up,
                "label": label,
            }
            daily_returns.append(-transaction_cost)  # Entry cost

        equity_curve.append(equity)

    # Close any remaining position at the end
    if position is not None:
        trade_return = position["forward_return"] - transaction_cost
        equity *= (1 + trade_return)
        daily_returns.append(trade_return)
        trades.append({
            "entry_date": str(position["entry_date"]),
            "exit_date": "end",
            "forward_return": position["forward_return"],
            "net_return": trade_return,
            "signal": position["signal"],
            "actual_label": position["label"],
            "hit": 1 if (trade_return > 0) else 0,
        })
        equity_curve.append(equity)

    equity_curve = np.array(equity_curve)
    daily_returns = np.array(daily_returns)

    # Compute metrics
    cagr = compute_cagr(equity_curve)
    sharpe = compute_sharpe(daily_returns)
    max_dd = compute_max_drawdown(equity_curve)

    # Hit rate (% of trades that were profitable)
    if trades:
        hit_rate = sum(t["hit"] for t in trades) / len(trades)
        avg_return = np.mean([t["net_return"] for t in trades])
    else:
        hit_rate = 0.0
        avg_return = 0.0

    # Buy and hold comparison
    total_market_return = df["forward_return"].sum()
    bh_equity = initial_capital * (1 + total_market_return)

    results = {
        "strategy": {
            "final_equity": equity,
            "total_return": (equity / initial_capital) - 1,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "num_trades": len(trades),
            "hit_rate": hit_rate,
            "avg_trade_return": avg_return,
        },
        "buy_and_hold": {
            "final_equity": bh_equity,
            "total_return": (bh_equity / initial_capital) - 1,
        },
        "trades": trades,
        "equity_curve": equity_curve.tolist(),
        "parameters": {
            "entry_threshold": entry_threshold,
            "horizon_days": horizon_days,
            "transaction_cost_bps": transaction_cost_bps,
            "initial_capital": initial_capital,
        },
    }

    return results


def print_backtest_results(results: dict):
    """Print backtest results in a formatted way."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    params = results["parameters"]
    print(f"\nParameters:")
    print(f"  Entry threshold:     {params['entry_threshold']:.2f}")
    print(f"  Holding period:      {params['horizon_days']} days")
    print(f"  Transaction cost:    {params['transaction_cost_bps']} bps")
    print(f"  Initial capital:     ${params['initial_capital']:,.2f}")

    strat = results["strategy"]
    print(f"\nStrategy Performance:")
    print(f"  Final equity:        ${strat['final_equity']:,.2f}")
    print(f"  Total return:        {strat['total_return']*100:.2f}%")
    print(f"  CAGR:                {strat['cagr']*100:.2f}%")
    print(f"  Sharpe ratio:        {strat['sharpe']:.2f}")
    print(f"  Max drawdown:        {strat['max_drawdown']*100:.2f}%")
    print(f"  Number of trades:    {strat['num_trades']}")
    print(f"  Hit rate:            {strat['hit_rate']*100:.2f}%")
    print(f"  Avg trade return:    {strat['avg_trade_return']*100:.2f}%")

    bh = results["buy_and_hold"]
    print(f"\nBuy & Hold Benchmark:")
    print(f"  Final equity:        ${bh['final_equity']:,.2f}")
    print(f"  Total return:        {bh['total_return']*100:.2f}%")

    excess = strat["total_return"] - bh["total_return"]
    print(f"\nExcess Return vs Buy & Hold: {excess*100:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Entry probability threshold")
    parser.add_argument("--cost", type=float, default=10.0,
                        help="Transaction cost in bps (round-trip)")
    args = parser.parse_args()

    config = load_config(args.config)
    horizon_days = config.get("horizon_days", 5)

    # Load predictions
    predictions_path = Path("data/models/test_predictions.json")
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}. Run eval.py first."
        )

    print("Loading predictions...")
    predictions_df = load_predictions(str(predictions_path))
    print(f"Loaded {len(predictions_df)} predictions")

    # Run backtest
    print("\nRunning backtest...")
    results = run_backtest(
        predictions_df,
        entry_threshold=args.threshold,
        horizon_days=horizon_days,
        transaction_cost_bps=args.cost,
    )

    # Print results
    print_backtest_results(results)

    # Save results
    results_path = Path("data/models/backtest_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {results_path}")


if __name__ == "__main__":
    main()
