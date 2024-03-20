FROM python:3.10
WORKDIR /AMP_Regression_EC_SA_Predict
COPY . /AMP_Regression_EC_SA_Predict

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    pip install -r requirements.txt