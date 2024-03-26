FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-cpu-py310-ubuntu20.04-ec2
# AWS pytorch image ^^ for CPU

RUN mkdir -p /app/home
WORKDIR /app/home

# Copy requirements.txt from parent directory
COPY ./requirements.txt requirements.txt
COPY ./run.sh run.sh
COPY ./jupyter_notebook_config.py /root/.jupyter/

# Install requirements
RUN pip install -r requirements.txt

# expose port 1234
EXPOSE 1234

# Run jupyter notebook
CMD ["bash", "run.sh"]
