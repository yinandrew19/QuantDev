"""
Walk-forward backtest harness for the DoubleML mortgage-spread model.

ADDITIVE ONLY: this file does not modify mbsSpread.py, data.py, or kFold.py.
It imports `run_double_ml` and `baselineForests` from mbsSpread.py exactly as
they're written and re-runs them across rolling/expanding windows, so the
existing single-split flow in `mbsSpread.main()` keeps working unchanged.

What this answers that a single train/test split can't:
  - Is theta (the estimated causal impact of T10 on the spread) stable
    across time, or does it swing fold to fold? (theta_mean / theta_std)
  - Does the DoubleML estimate actually beat naive ML baselines
    out-of-sample, on average, not just in one lucky window?
  - If you traded the direction of the predicted spread move, what would
    the Sharpe ratio, max drawdown, hit rate, and information coefficient
    have been?

Cost note: `run_double_ml` already runs its own internal 3-fold
GridSearchCV over a 32-point grid for two LightGBM models per call. Each
backtest fold pays that full cost again, so a long backtest (many folds)
is expensive — use --max-folds while iterating locally, and run full
backtests on the existing MLStack Fargate task rather than a laptop.

Usage:
    python backtest.py --train-weeks 104 --test-weeks 8 --step-weeks 8
    python backtest.py --local-parquet ./cached_dataset.parquet --max-folds 5
"""

import argparse
import io
import json
import os
import re
from contextlib import redirect_stdout
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import data
from mbsSpread import baselineForests, run_double_ml

THETA_RE = re.compile(r"Estimated Theta \(Causal Impact of \w+\):\s*([\-0-9.eE+]+)")
SE_RE = re.compile(r"Standard Error:\s*([\-0-9.eE+]+)")


# ---------------------------------------------------------------------------
# Window generator
# ---------------------------------------------------------------------------


def walk_forward_splits(
    df: pd.DataFrame, train_size: int, test_size: int, step: int, expanding: bool = True
):
    """
    Yield (train_df, test_df) windows walking forward through df.

    expanding=True: train window grows from the start
    expanding=False: train window is a fixed-size rolling window (
    check whether older regimes are actively impacting the fit).
    """
    n = len(df)
    start = train_size
    while start + test_size <= n:
        train_start = 0 if expanding else max(0, start - train_size)
        train_df = df.iloc[train_start:start]
        test_df = df.iloc[start : start + test_size]
        yield train_df, test_df
        start += step


# ---------------------------------------------------------------------------
# Theta capture without touching run_double_ml's return signature
# ---------------------------------------------------------------------------


def _run_double_ml_capture(dfTrain: pd.DataFrame, dfTest: pd.DataFrame) -> dict:
    """
    Calls run_double_ml unmodified, but additionally captures the printed
    theta/std_err lines via stdout redirection + regex, since run_double_ml
    only prints those values rather than returning them (see SUGGESTION
    comment in mbsSpread.py). Returns the original dict plus "theta" and
    "std_err" (None if parsing fails, so a parsing miss never breaks a run).
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = run_double_ml(dfTrain, dfTest)
    text = buf.getvalue()

    theta_match = THETA_RE.search(text)
    se_match = SE_RE.search(text)
    result["theta"] = float(theta_match.group(1)) if theta_match else None
    result["std_err"] = float(se_match.group(1)) if se_match else None
    return result


# ---------------------------------------------------------------------------
# Per-fold + aggregate metrics
# ---------------------------------------------------------------------------


def max_drawdown(pnl: pd.Series) -> float:
    """Max drawdown on cumulative additive PnL (spread units, not returns)."""
    if pnl.empty:
        return float("nan")
    cum = pnl.cumsum()
    peak = cum.cummax()
    # loss in negative
    return float((cum - peak).min())


def run_backtest(
    df: pd.DataFrame,
    train_size: int = 104,
    test_size: int = 8,
    step: int = 8,
    expanding: bool = True,
    max_folds: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Walk df forward, refit run_double_ml + baselineForests on each window.

    Returns:
        fold_df: one row per fold with theta, std_err, MSEs, hit rate, IC.
        pooled_pnl: concatenated directional strategy PnL across all folds'
            test windows, for the aggregate Sharpe/drawdown calc.
    """
    rows = []
    pnl_chunks = []

    for fold, (train_df, test_df) in enumerate(
        walk_forward_splits(df, train_size, test_size, step, expanding)
    ):
        if max_folds is not None and fold >= max_folds:
            break

        dml_result = _run_double_ml_capture(train_df, test_df)
        baseline = baselineForests(train_df, test_df)

        y_true = dml_result["True"]
        y_pred = pd.Series(dml_result["Prediction"], index=y_true.index)

        realized_change = y_true.diff().dropna()
        predicted_change = y_pred.diff().dropna()
        common_idx = realized_change.index.intersection(predicted_change.index)
        realized_change = realized_change.loc[common_idx]
        predicted_change = predicted_change.loc[common_idx]

        hit = np.sign(realized_change) == np.sign(predicted_change)
        # information coefficient to find prediction accuracy
        ic = (
            float(np.corrcoef(realized_change, predicted_change)[0, 1])
            if len(realized_change) > 1
            else float("nan")
        )

        # Check if betting on the sign of the predicted spread
        # move, earn the realized move with that sign applied.
        fold_pnl = realized_change * np.sign(predicted_change)
        pnl_chunks.append(fold_pnl)

        rows.append(
            {
                "fold": fold,
                "train_start": train_df.index[0],
                "train_end": train_df.index[-1],
                "test_start": test_df.index[0],
                "test_end": test_df.index[-1],
                "n_train": len(train_df),
                "n_test": len(test_df),
                "theta": dml_result["theta"],
                "std_err": dml_result["std_err"],
                "dml_mse": float(dml_result["MSE"]),
                "lgb_mse": float(baseline["lgb"]),
                "rf_mse": float(baseline["rf"]),
                "hit_rate": float(hit.mean()) if len(hit) else float("nan"),
                "information_coef": ic,
            }
        )
        print(
            f"[fold {fold}] {test_df.index[0].date()} -> {test_df.index[-1].date()} "
            f"theta={dml_result['theta']} dml_mse={dml_result['MSE']:.6g} "
            f"lgb_mse={baseline['lgb']:.6g} rf_mse={baseline['rf']:.6g} "
            f"hit_rate={rows[-1]['hit_rate']}"
        )

    fold_df = pd.DataFrame(rows)
    pooled_pnl = pd.concat(pnl_chunks) if pnl_chunks else pd.Series(dtype=float)
    return fold_df, pooled_pnl


def summarize(
    fold_df: pd.DataFrame, pooled_pnl: pd.Series, periods_per_year: int = 52
) -> dict:
    """Aggregate diagnostics across all folds — the numbers worth reporting."""
    theta = fold_df["theta"].dropna()
    pnl_mean = pooled_pnl.mean() if len(pooled_pnl) else float("nan")
    pnl_std = pooled_pnl.std(ddof=1) if len(pooled_pnl) > 1 else float("nan")
    sharpe = (
        float(pnl_mean / pnl_std * np.sqrt(periods_per_year))
        if pnl_std and not np.isnan(pnl_std) and pnl_std != 0
        else float("nan")
    )

    return {
        "n_folds": int(len(fold_df)),
        "theta_mean": float(theta.mean()) if len(theta) else None,
        "theta_std": float(theta.std(ddof=1)) if len(theta) > 1 else None,
        "theta_min": float(theta.min()) if len(theta) else None,
        "theta_max": float(theta.max()) if len(theta) else None,
        "dml_mse_mean": float(fold_df["dml_mse"].mean()) if len(fold_df) else None,
        "lgb_mse_mean": float(fold_df["lgb_mse"].mean()) if len(fold_df) else None,
        "rf_mse_mean": float(fold_df["rf_mse"].mean()) if len(fold_df) else None,
        "hit_rate_mean": float(fold_df["hit_rate"].mean()) if len(fold_df) else None,
        "information_coef_mean": (
            float(fold_df["information_coef"].mean()) if len(fold_df) else None
        ),
        "sharpe_annualized": sharpe,
        "max_drawdown": max_drawdown(pooled_pnl),
        "n_periods": int(len(pooled_pnl)),
    }


# ---------------------------------------------------------------------------
# Data loading — S3 (matches mbsSpread.main()) or a local parquet cache
# ---------------------------------------------------------------------------


def load_dataset(bucket: str | None, local_parquet: str | None) -> pd.DataFrame:
    if local_parquet:
        print(f"Loading cached dataset from {local_parquet} …")
        return pd.read_parquet(local_parquet)

    import boto3

    if not bucket:
        raise ValueError("Either --bucket/DATA_BUCKET or --local-parquet is required.")
    s3 = boto3.client("s3")
    print(f"Loading data from s3://{bucket}/raw/ …")
    return data.build_dataset(s3, bucket)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.environ.get("DATA_BUCKET"))
    parser.add_argument(
        "--local-parquet", default=None, help="Bypass S3, use a cached dataset."
    )
    parser.add_argument("--train-weeks", type=int, default=104)
    parser.add_argument("--test-weeks", type=int, default=8)
    parser.add_argument("--step-weeks", type=int, default=8)
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Use a fixed-size rolling train window instead of expanding.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Cap the number of folds — use while iterating locally; GridSearchCV per fold is expensive.",
    )
    parser.add_argument(
        "--no-s3-write",
        action="store_true",
        help="Print results only, don't write to S3 (default behavior if --local-parquet is set).",
    )
    args = parser.parse_args()

    df = load_dataset(args.bucket, args.local_parquet)
    print(f"Dataset shape: {df.shape}")

    fold_df, pooled_pnl = run_backtest(
        df,
        train_size=args.train_weeks,
        test_size=args.test_weeks,
        step=args.step_weeks,
        expanding=not args.rolling,
        max_folds=args.max_folds,
    )
    summary = summarize(fold_df, pooled_pnl)

    print("\n=== Backtest Summary ===")
    print(json.dumps(summary, indent=2, default=str))

    result = {"summary": summary, "folds": fold_df.to_dict(orient="records")}

    write_to_s3 = args.bucket and not args.local_parquet and not args.no_s3_write
    if write_to_s3:
        import boto3

        s3 = boto3.client("s3")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"models/double_ml/backtest_{today}.json"
        s3.put_object(
            Bucket=args.bucket,
            Key=key,
            Body=json.dumps(result, indent=2, default=str).encode(),
        )
        print(f"\nResult written to s3://{args.bucket}/{key}")


if __name__ == "__main__":
    main()
