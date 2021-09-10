# Notebook examples for RL algorithms

## SageMaker
To use SageMaker for training, build a trainng environment and host
it on your AWS ECR account
```sh
export AWS_ACCOUNT=<Your AWS Account>
exprot REGION=<Your Region>

# get ECR credentials for base image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com 

docker build -t ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/pt:cpu \
  -f Dockerfile.pytorch .

# login to your ecr
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com 

# create a repo
aws ecr create-repository --repository-name pt

docker push ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/pt:cpu
```


