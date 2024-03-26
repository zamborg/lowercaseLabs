#!/bin/bash

#setup region
REGION=us-east-1

# auth to aws docker repo:
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-cpu-py310-ubuntu20.04-ec2
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-ec2


