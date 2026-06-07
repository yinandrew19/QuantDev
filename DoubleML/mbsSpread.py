import json
import os
from datetime import datetime

import boto3
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from lightgbm import LGBMRegressor
from sklearn.base import clone

import data


def run_double_ml(df: pd.DataFrame) -> dict:
    """
    Example: estimate the causal effect of a change in T10 yield on
    MBS-spread proxy (30YMortRate - T10), controlling for the rest.
    """

    y_col = "spread"  # outcome
    d_col = "T10"  # treatment
    x_cols = [c for c in df.columns if c not in (y_col, d_col, "30YMortRate")]

    dml_data = DoubleMLData(df, y_col=y_col, d_cols=d_col, x_cols=x_cols)

    # short tree to accommodate small sample data
    ml_l = LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)
    ml_m = clone(ml_l)

    dml_plr = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=5)
    dml_plr.fit()

    return {
        "treatment": d_col,
        "outcome": y_col,
        "coef": float(dml_plr.coef[0]),
        "se": float(dml_plr.se[0]),
        "ci_low": float(dml_plr.confint().iloc[0, 0]),
        "ci_high": float(dml_plr.confint().iloc[0, 1]),
        "n_obs": int(len(df)),
    }


def main():
    bucket = os.environ["DATA_BUCKET"]
    s3 = boto3.client("s3")

    print(f"Loading data from s3://{bucket}/raw/ …")
    df = data.build_dataset(s3, bucket)
    print(f"Dataset shape: {df.shape}")

    print("Running Double ML …")
    result = run_double_ml(df)
    print(json.dumps(result, indent=2))

    today = datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    s3.put_object(
        Bucket=bucket,
        Key=f"models/double_ml/result_{today}.json",
        Body=json.dumps(result, indent=2).encode(),
    )
    print(f"Result written to s3://{bucket}/models/double_ml/result_{today}.json")


if __name__ == "__main__":
    # import boto3

    # s3 = boto3.client("s3")
    # for o in s3.list_objects_v2(Bucket="mbs-struct-bucket", Prefix="raw/").get(
    #     "Contents", []
    # ):
    #     print(o["Key"])
    main()
