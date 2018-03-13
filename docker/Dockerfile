FROM tensorflow/tensorflow:latest-gpu-py3

RUN mkdir /app
WORKDIR /app

COPY ./ ./

RUN pip install keras hyperopt scikit-learn

CMD python optimize.py
