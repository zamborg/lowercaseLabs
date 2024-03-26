#!/bin/bash

# Check if --gpu flag is passed
if [[ "$1" == "--gpu" ]]; then
    echo "gpu"
else
    echo "no gpu"
fi



docker build . -t jupyter:latest -t 343356233979.dkr.ecr.us-east-1.amazonaws.com/lowercaselabs
docker push 343356233979.dkr.ecr.us-east-1.amazonaws.com/lowercaselabs
