import json
import os
from datetime import datetime
from io import BytesIO, StringIO

import boto3 as boto
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from lightgbm import LGBMRegressor
from sklearn.base import clone


def aws_client(service: str):
    """Return a boto3 client, pointed at LocalStack if AWS_ENDPOINT_URL is set."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL")  # set only in local dev
    return (
        boto.client(service, endpoint_url=endpoint)
        if endpoint
        else boto3.client(service)
    )


def build_dataset(s3, bucket: str) -> pd.DataFrame:
    """Combine all series into one DataFrame indexed by date."""
    series_names = [
        "30YMortRate",
        "MediCPI",
        "VIX",
        "T10-2Curve",
        "T30",
        "T10",
        "FHA30",
        "Jumbo30Y",
        "Conform30Y",
        "FedFundRate",
        "CaseShiller",
        "HighYield",
        "CorpInvestGrade",
        "GoldVol",
        "Unemploy",
        "CommPaperLessFFR",
        "BaaLess10Y",
        "Fin",
        "ConsumeDiscre",
        "ConsumeStaple",
        "RealEstate",
        "Health",
        "Tech",
        "Industrial",
        "MBSYield",
        "OASVol",
    ]
    cols = [load_series(s3, bucket, f"{n}_data") for n in series_names]
    df = pd.concat(cols, axis=1).dropna()
    return df


def run_double_ml(df: pd.DataFrame) -> dict:
    """
    Example: estimate the causal effect of a change in T10 yield on
    MBS-spread proxy (30YMortRate - T10), controlling for the rest.
    """
    df = df.copy()
    df["MtgSpread"] = df["MBSYield"] - df["T10"]

    y_col = "MtgSpread"  # outcome
    d_col = "T10"  # treatment
    x_cols = [c for c in df.columns if c not in (y_col, d_col, "30YMortRate")]

    dml_data = DoubleMLData(df, y_col=y_col, d_cols=d_col, x_cols=x_cols)

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
    bucket = "mbs_struct_bucket"  # since local testing, not pulling from aws
    s3 = boto3.client("s3")

    print(f"Loading data from s3://{bucket}/raw/ …")
    df = build_dataset(s3, bucket)
    print(f"Dataset shape: {df.shape}")

    print("Running Double ML …")
    result = run_double_ml(df)
    print(json.dumps(result, indent=2))

    today = datetime.utcnow().strftime("%Y-%m-%d")
    s3.put_object(
        Bucket=bucket,
        Key=f"models/double_ml/result_{today}.json",
        Body=json.dumps(result, indent=2).encode(),
    )
    print(f"Result written to s3://{bucket}/models/double_ml/result_{today}.json")


if __name__ == "__main__":
    main()
# s3 = aws_client("s3")


# objs = s3.list_objects_v2(Bucket="mbs-struct-bucket", Prefix="raw/")

# for obj in objs.get("Contents", []):
#     print(obj["Key"], obj["Size"], obj["LastModified"])
