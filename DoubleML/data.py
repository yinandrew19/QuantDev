"""
Data loading and shaping for the DoubleML mortgage-spread model.

All S3 fetching and frame assembly lives here. mbsSpread.py just calls
`build_dataset` and runs the model.
"""

import json
from io import BytesIO

import pandas as pd

# ---------------------------------------------------------------------------
# S3 -> pandas.Series
# ---------------------------------------------------------------------------


def load_series(s3, bucket: str, name: str) -> pd.Series:
    """Load all files for a single series (FRED .json or yfinance .csv)."""
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=f"raw/{name}_data")
    frames = []
    for o in objs.get("Contents", []):
        body = s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()
        if o["Key"].endswith(".json"):
            payload = json.loads(body)
            df = pd.DataFrame(payload["observations"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            frames.append(df.set_index("date")["value"].rename(name))
        elif o["Key"].endswith(".csv"):
            df = pd.read_csv(BytesIO(body), index_col=0)
            # Don't rely on parse_dates=True on read_csv — under newer pandas
            # (especially with pyarrow string storage) the index can land as
            # 'large_string', which breaks .diff()/.dt downstream. Parse it
            # explicitly to UTC-aware datetime and drop anything unparseable.
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df[~df.index.isna()]
            frames.append(
                df["Close"].apply(pd.to_numeric, errors="coerce").rename(name)
                if "Close" in df
                else df.iloc[:, 0].apply(pd.to_numeric, errors="coerce").rename(name)
            )
    if not frames:
        raise FileNotFoundError(
            f"No S3 objects found under prefix 'raw/{name}_data' in bucket "
            f"'{bucket}'. Check the series name and that ingest_data has run."
        )
    return pd.concat(frames).sort_index().groupby(level=0).last()


# ---------------------------------------------------------------------------
# The accumulating frame
# ---------------------------------------------------------------------------


class FinData:
    """
    Accumulates time-series columns aligned to a weekly mortgage-spread target.

    Call ``add_series`` once per data source. On the first call it seeds the
    target column ``spread`` from ``target_y − target_d`` (default
    30YMortRate − T10). Subsequent calls only append features

    For features released less often than weekly (CPI, Case-Shiller, etc.)
    a ``{name}_days_since_release`` column is added automatically so the
    model can see staleness.
    """

    WEEKLY_GAP_DAYS = 7  # release period > 7 days ⇒ sub-weekly

    def __init__(
        self,
        excludes: list | None = None,
        target_y: str = "30YMortRate",
        target_d: str = "T10",
    ):
        self.data = pd.DataFrame()
        self.target_y = target_y
        self.target_d = target_d
        self.excludes = set(excludes or []) | {target_y}

    @property
    def is_initialized(self) -> bool:
        """True once self.data has been seeded with the target."""
        return not self.data.empty

    def add_series(self, series: dict) -> None:
        """
        Attach variables to ``self.data``.

        On the first call (when ``self.data`` is empty) the dict must include
        both ``target_y`` and ``target_d`` to build the target spread.
        Later calls don't need them.
        """
        if not self.is_initialized:
            if self.target_y not in series or self.target_d not in series:
                raise ValueError(
                    f"First add_series batch must include '{self.target_y}' "
                    f"and '{self.target_d}' to seed the target spread."
                )
            spread = (
                (series[self.target_y] - series[self.target_d])
                .dropna()
                .rename("spread")
            )
            self.data = (
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(spread.index, utc=True),
                        "spread": spread.values,
                    }
                )
                .sort_values("date")
                .reset_index(drop=True)
            )

        for name, s in series.items():
            if name in self.excludes:
                continue
            if name in self.data.columns:
                continue  # already attached on an earlier call
            s = s.dropna().sort_index()
            if s.empty:
                raise ValueError(f"the data is empty!")

            # 1. Fill the data with the most recent prior record to avoid data leaking
            right = pd.DataFrame(
                {"date": pd.to_datetime(s.index, utc=True), name: s.values}
            ).sort_values("date")
            self.data[name] = pd.merge_asof(
                self.data[["date"]], right, on="date", direction="backward"
            )[name].values

            # 2. If the native release cadence is slower than weekly,
            #    attach a days-since-last-release column.
            gaps = s.index.to_series().diff().dt.days.dropna()
            if not gaps.empty and gaps.median() > self.WEEKLY_GAP_DAYS:
                # Both columns must be tz-aware so the subtraction below
                # (self.data["date"] - asof) doesn't mix naive and aware.
                s_dates_utc = pd.to_datetime(s.index, utc=True)
                rel = pd.DataFrame(
                    {"date": s_dates_utc, "_rel": s_dates_utc}
                ).sort_values("date")
                asof = pd.merge_asof(
                    self.data[["date"]], rel, on="date", direction="backward"
                )["_rel"]
                self.data[f"{name}DayLag"] = (self.data["date"] - asof).dt.days.values

    def finalize(self) -> pd.DataFrame:
        """Return the frame indexed by date, dropping rows with any NaNs."""
        return self.data.set_index("date").dropna()


# ---------------------------------------------------------------------------
# Series catalogues — keep in sync with lambda/ingest_data.py
# ---------------------------------------------------------------------------

FRED_SERIES = [
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
    "StreeIndex",
    "MAaa",
    "MBaa",
    "GoldVol",
    "Unemploy",
    "CommPaperLessFFR",
    "BaaLess10Y",
]

YFIN_SERIES = [
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


# ---------------------------------------------------------------------------
# Top-level entry point used by mbsSpread.py
# ---------------------------------------------------------------------------


def build_dataset(s3, bucket: str) -> pd.DataFrame:
    """Load FRED + yfinance series from S3 into one weekly-aligned frame."""
    fd = FinData(excludes=["FHA30", "Jumbo30Y", "Conform30Y"])

    # FRED batch — seeds the target spread and adds macro features
    fd.add_series({n: load_series(s3, bucket, n) for n in FRED_SERIES})

    # yfinance batch — appends sector indices and vol measures
    fd.add_series({n: load_series(s3, bucket, n) for n in YFIN_SERIES})

    return fd.finalize()
