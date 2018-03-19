FROM tensorflow/tensorflow:1.6.0-gpu-py3

RUN mkdir /app
WORKDIR /app

COPY ./ ./

RUN pip install keras hyperopt scikit-learn networkx==1.11

CMD python optimize.py
