.PHONY: layer template up bootstrap invoke down clean

# Rebuild the Lambda layer for Python 3.12 / linux-x86_64
layer:
	rm -rf lambda_layer/python && mkdir -p lambda_layer/python
	docker run --rm --platform linux/x86_64 --entrypoint "" \
	  -v "$$PWD":/work public.ecr.aws/lambda/python:3.12 \
	  pip install -r /work/lambda/requirements.txt -t /work/lambda_layer/python

# Synthesize CloudFormation template for SAM
template:
	cdk synth --no-staging > template.yaml

# Start LocalStack
up:
	docker compose up -d localstack

# Create the S3 bucket and SSM parameter inside LocalStack
bootstrap:
	./scripts/bootstrap-localstack.sh

# Invoke the Lambda end-to-end against LocalStack
invoke: template
	sam local invoke IngestFunction \
	  --template template.yaml \
	  --env-vars env.json \
	  --docker-network quantdev_net

# Tear down
down:
	docker compose down

clean: down
	rm -rf .localstack template.yaml