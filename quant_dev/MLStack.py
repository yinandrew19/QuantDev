from aws_cdk import (
    Stack,
    Duration,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_logs as logs,
    aws_s3 as s3,
)
from aws_cdk.aws_ecr_assets import Platform as DockerPlatform
from constructs import Construct


class MLStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        data_bucket: s3.IBucket,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Cheap VPC — public subnets only, no NAT gateway (~$32/mo saved)
        vpc = ec2.Vpc(
            self,
            "MLVpc",
            # 2 availability zones
            max_azs=2,
            # cost saving, everyting public
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                )
            ],
        )

        cluster = ecs.Cluster(self, "MLCluster", vpc=vpc)

        task_definition = ecs.FargateTaskDefinition(
            self,
            "DoubleMLTaskDef",
            cpu=2048,  # 2 vCPU
            memory_limit_mib=8192,  # 8 GB
            runtime_platform=ecs.RuntimePlatform(
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
                cpu_architecture=ecs.CpuArchitecture.ARM64,  # cheaper Graviton
            ),
        )

        # CDK builds the image from ./DoubleML and pushes it to ECR
        # during `cdk deploy`. No manual `docker build`/`docker push`.
        task_definition.add_container(
            "DoubleMLContainer",
            image=ecs.ContainerImage.from_asset(
                "DoubleML", platform=DockerPlatform.LINUX_ARM64
            ),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="DoubleML",
                log_retention=logs.RetentionDays.ONE_MONTH,
            ),
            environment={
                "DATA_BUCKET": data_bucket.bucket_name,
            },
        )

        # Grant task read/write on the data bucket
        data_bucket.grant_read_write(task_definition.task_role)

        # Outputs so you can grab the names easily from the CLI
        self.cluster_name = cluster.cluster_name
        self.task_definition_arn = task_definition.task_definition_arn
        self.subnet_ids = [s.subnet_id for s in vpc.public_subnets]
