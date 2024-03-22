FROM python:3.10
# WORKDIR /amp-regression-predict-flask
# COPY . /amp-regression-predict-flask
WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    pip install -r requirements.txt