from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_events as events,
    aws_events_targets as targets,
    aws_ssm as ssm,
)
from constructs import Construct


class QuantDevStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. Existing Storage Logic
        self.raw_data_bucket = s3.Bucket(
            self,
            "QuantRawDataBucket",
            bucket_name="mbs-struct-bucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # Refing the existing parameter for FRED API key
        fred_key_param = ssm.StringParameter.from_secure_string_parameter_attributes(
            self, "FredApiKey", parameter_name="/quant/fred-api-key"
        )
        # 1. Define the Layer
        requests_layer = _lambda.LayerVersion(
            self,
            "RequestsLayer",
            code=_lambda.Code.from_asset("lambda_layer"),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_12],
            description="A layer for the requests, yfinance library",
        )

        # 2. Add the layer to your existing IngestFunction
        ingest_function = _lambda.Function(
            self,
            "IngestFunction",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="ingest_data.handler",
            # CDK can know there is a folder named "lambda/" for data ingestion code
            code=_lambda.Code.from_asset("lambda"),
            # Adding more time so data ingestion won't time out in 3 sec
            timeout=Duration.seconds(100),
            environment={
                "BUCKET_NAME": self.raw_data_bucket.bucket_name,
                "FRED_API_KEY_PARAM": fred_key_param.parameter_name,
                # Declared so SAM Local can override it locally via env.json.
                # Empty in production — aws_client() treats empty as "use real AWS".
                "AWS_ENDPOINT_URL": "",
            },
            layers=[requests_layer],  # adding layer for "requests"
        )

        # 3. Security Logic: Grant Lambda write access to your specific bucket
        self.raw_data_bucket.grant_put(ingest_function)

        # Grant read Access to the lambda function
        fred_key_param.grant_read(ingest_function)

        # Pass the parameter name as an environment variable
        ingest_function.add_environment(
            "FRED_API_KEY_PARAM", fred_key_param.parameter_name
        )
        # 4. Automation Logic: Define the Trigger (Daily Schedule)
        rule = events.Rule(
            self,
            "DailyIngestionRule",
            schedule=events.Schedule.rate(Duration.days(7)),
        )
        rule.add_target(targets.LambdaFunction(ingest_function))
