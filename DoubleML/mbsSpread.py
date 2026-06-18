import json
import os
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from kFold import PurgedEmbargoedKFold
import boto3
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import numpy as np

import data


def baselineForests(dfTrain: pd.DataFrame, dfTest: pd.DataFrame):
    y_col = "spread"  # outcome
    x_cols = [c for c in dfTrain.columns if c not in (y_col, "30YMortRate", "T10")]
    lgb = LGBMRegressor().fit(dfTrain[x_cols], dfTrain[y_col])
    lgbPred = lgb.predict(dfTest[x_cols])

    rf = RandomForestRegressor().fit(dfTrain[x_cols], dfTrain[y_col])
    rfPred = rf.predict(dfTest[x_cols])
    rfMSE = np.mean((rfPred - dfTest[y_col]) ** 2)
    lgbMSE = np.mean((lgbPred - dfTest[y_col]) ** 2)
    return {"lgb": lgbMSE, "rf": rfMSE}


def run_double_ml(dfTrain: pd.DataFrame, dfTest: pd.DataFrame) -> dict:
    """
    Example: estimate the causal effect of a change in T10 yield on
    MBS-spread proxy (30YMortRate - T10), controlling for the rest.
    """

    y_col = "spread"  # outcome
    d_col = "T10"  # treatment
    x_cols = [c for c in dfTrain.columns if c not in (y_col, d_col, "30YMortRate")]

    cv = PurgedEmbargoedKFold(n_splits=3, embargo=2)

    # 1) Out-of-sample grid search for each nuisance learner.
    #    ml_l predicts Y from X; ml_m predicts D from X. DoubleML wants both
    #    well-fit but not overfit, which is exactly what time-respecting CV
    #    guards against.
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.05],
        "num_leaves": [15, 31],
        "min_child_samples": [10, 20],
    }

    X = dfTrain[x_cols].values
    y = dfTrain[y_col].values
    d = dfTrain[d_col].values

    yHat = np.zeros(y.shape, dtype=float)
    dHat = np.zeros(d.shape, dtype=float)

    ml_l_estimators, ml_m_estimators = [], []

    gs_kw = dict(cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, refit=True)

    # -------------------------------------------------------------------------
    # 2. Manual run to avoid future data leakage
    # -------------------------------------------------------------------------

    for fold, (trainIdx, valIdx) in enumerate(cv.split(X)):
        x_train, y_train, d_train = (X[trainIdx], y[trainIdx], d[trainIdx])
        x_val, y_val, d_val = (X[valIdx], y[valIdx], d[valIdx])

        # Fit ml_l Y ~ X
        ml_l = GridSearchCV(LGBMRegressor(verbose=-1), param_grid, **gs_kw).fit(
            x_train, y_train
        )
        yHat[valIdx] = ml_l.best_estimator_.predict(x_val)
        ml_l_estimators.append(ml_l.best_estimator_)

        # Fit ml_m D~X
        ml_m = GridSearchCV(LGBMRegressor(verbose=-1), param_grid, **gs_kw).fit(
            x_train, d_train
        )
        dHat[valIdx] = ml_m.best_estimator_.predict(x_val)
        ml_m_estimators.append(ml_m.best_estimator_)

    # -------------------------------------------------------------------------
    # 3. ESTIMATE CAUSAL EFFECT (THETA)
    # -------------------------------------------------------------------------
    y_tilde = y - yHat
    d_tilde = d - dHat

    theta = np.dot(d_tilde, y_tilde) / np.dot(d_tilde, d_tilde)

    # Calculate robust standard errors
    residuals = y_tilde - theta * d_tilde
    j_val = np.mean(d_tilde**2)
    omega = np.mean((d_tilde**2) * (residuals**2))
    var_theta = omega / (len(dfTrain) * (j_val**2))
    std_err = np.sqrt(var_theta)

    print("\n=== Historical Causal Inference ===")
    print(f"Estimated Theta (Causal Impact of {d_col}): {theta:.6f}")
    print(f"Standard Error:                          {std_err:.6f}")

    # -------------------------------------------------------------------------
    # 4. FUTURE VALUE PREDICTION (OUT-OF-SAMPLE FORECASTING)
    # -------------------------------------------------------------------------
    print("\nProjecting structural predictions onto future dataset...")
    X_future = dfTest[x_cols].values
    d_future = dfTest[d_col].values  # Realized or scenario-based treatment

    # Generate baseline predictions by averaging the predictions of all fold models
    y_hat_future_folds = np.column_stack(
        [model.predict(X_future) for model in ml_l_estimators]
    )
    d_hat_future_folds = np.column_stack(
        [model.predict(X_future) for model in ml_m_estimators]
    )

    y_hat_future_baseline = np.mean(y_hat_future_folds, axis=1)
    d_hat_future_baseline = np.mean(d_hat_future_folds, axis=1)

    # Compute the future treatment innovation (the unexpected movement in D)
    # D_tilde_future = D_future_actual - E[D_future | X_future]
    d_tilde_future = d_future - d_hat_future_baseline

    # Combine the baseline outcome expectation with the orthogonalized causal impact
    # Y_pred = E[Y | X] + theta * (D - E[D | X])
    yPred = y_hat_future_baseline + theta * d_tilde_future

    MSE = np.mean((dfTest[y_col] - yPred) ** 2)

    return {"True": dfTest[y_col], "Prediction": yPred, "MSE": MSE}


def main():
    bucket = os.environ["DATA_BUCKET"]
    s3 = boto3.client("s3")

    print(f"Loading data from s3://{bucket}/raw/ …")
    df = data.build_dataset(s3, bucket)
    print(f"Dataset shape: {df.shape}")

    print("Running Double ML …")

    # additional 2 weeks to avoid AR
    dfTrain, dfTest = df.iloc[:-12, :], df.iloc[-10:, :]
    result = run_double_ml(dfTrain, dfTest)
    baselineRes = baselineForests(dfTrain, dfTest)

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
