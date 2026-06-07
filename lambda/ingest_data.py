import boto3 as boto
import requests
import json
import os
from datetime import datetime, timedelta
import yfinance


def aws_client(service: str):
    """Return a boto3 client, pointed at LocalStack if AWS_ENDPOINT_URL is set."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL")  # set only in local dev
    return (
        boto.client(service, endpoint_url=endpoint)
        if endpoint
        else boto.client(service)
    )


def handler(event, context):
    print(f"AWS_ENDPOINT_URL={os.environ.get('AWS_ENDPOINT_URL')!r}")
    print(f"AWS_ACCESS_KEY_ID={os.environ.get('AWS_ACCESS_KEY_ID')!r}")
    print(f"FRED_API_KEY_PARAM={os.environ.get('FRED_API_KEY_PARAM')!r}")
    FredData()
    yfinData()
    return {"statusCode": 200, "body": "Data ingestion successful"}


def FredData():
    # 1. Fetch the secret key reference from the environment variable
    # defined in your QuantDevStack (ingest_function.add_environment)
    param_name = os.environ["FRED_API_KEY_PARAM"]
    # retrieve AWS bucket name
    bucket_name = os.environ["BUCKET_NAME"]
    s3 = aws_client("s3")
    # add date to file name so files don't overwrite
    # today = datetime.datetime.now().strftime("%Y-%m-%d")

    # 2. Use boto3 to talk to SSM (the "Vault" for your keys)
    ssm = aws_client("ssm")
    response = ssm.get_parameter(Name=param_name, WithDecryption=True)
    api_key = response["Parameter"]["Value"]

    # 3. Pull Data from FRED
    # We use the 'requests' library to call the API
    FredSeries = [
        "MORTGAGE30US",  # weekly
        "MEDCPIM158SFRBCLE",  # monthly 1983
        "VIXCLS",  # daily 1990
        "T10Y2Y",
        "DGS30",
        "DGS10",
        "OBMMIFHA30YF",  # daily 2017
        "OBMMIJUMBO30YF",
        "OBMMIC30YF",
        "DFF",
        "CSUSHPINSA",  # daily 1987
        "STLFSI4",  # weekly 1993
        "DAAA",
        "DBAA",
        "GVZCLS",
        "UNRATE",
        "CPFF",
        "BAA10Y",
    ]
    FredNames = [
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
    FredDict = dict(zip(FredSeries, FredNames))
    # set parameters to get the last 2 years of data
    twenty_years_ago = datetime.now() - timedelta(days=int(365.25 * 50))
    observation_start = fifty_years_ago.strftime("%Y-%m-%d")

    for series_id in FredSeries:
        try:
            series_id = series_id  # 30-Year Fixed Mortgage Rate
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "observation_start": observation_start,
                # "limit": 100,  # Get the most recent 100 data points
            }
            res = requests.get(url, params=params)
            res.raise_for_status()
            mort30 = res.json()
        except requests.exceptions.RequestException as e:
            print("Error getting from FRED")
            raise e  # This triggers a Lambda retry if configured

        # 4. Save to S3
        # The bucket name was passed to Lambda via environment variable
        s3.put_object(
            Bucket=bucket_name,
            # raw/{FredDict[series_id]}_data({today}).json -> to include date
            Key=f"raw/{FredDict[series_id]}_data.json",
            Body=json.dumps(mort30),
        )


# SEC standard fund's 30 day yield
def yieldCal(MBB):
    div = MBB.history("20y").Dividends
    high = MBB.history("20y").High
    return ((div[div > 0] / high[div > 0] + 1).pow(6) - 1) * 200


def yfinData():
    import yfinance as yf

    sectorTicker = [
        "^SP500-40",
        "^SP500-25",
        "^SP500-30",
        "^SP500-60",
        "^SP500-35",
        "^SP500-45",
        "^SP500-20",
        "MBB",
        "^MOVE",
    ]
    sectorName = [
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

    bucket_name = os.environ["BUCKET_NAME"]
    s3 = aws_client("s3")

    # add date to file name so files don't overwrite
    # today = datetime.datetime.now().strftime("%Y-%m-%d")

    equity = dict(zip(sectorTicker, sectorName))
    for series_id in sectorTicker:
        if series_id == "MBB":
            data = yieldCal(yf.Ticker(series_id))
        else:
            data = yf.Ticker(series_id).history(period="20y").Close

        s3.put_object(
            Bucket=bucket_name,
            # raw/{equity[series_id]}_data({today}).json
            Key=f"raw/{equity[series_id]}_data.csv",
            Body=data.to_csv().encode("utf-8"),
        )
