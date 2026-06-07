import os

import sys

sys.path.append("/Users/andrewyin/quantDev/lambda")
from ingest_data import handler

# Manually set the environment variables your code expects
os.environ["FRED_API_KEY_PARAM"] = "/quant/fred-api-key"
os.environ["BUCKET_NAME"] = "mbs-struct-bucket"

# Execute the handler with empty event/context
# Use your profile to ensure boto3 uses your credentials
import boto3

boto3.setup_default_session(profile_name="quant-dev", region_name="us-east-1")

print("Starting local test...")
result = handler({}, None)
print(f"Result: {result}")
