FROM tensorflow/tensorflow:1.4.1-gpu-py3

RUN mkdir /app
WORKDIR /app

COPY ./ ./

RUN pip install keras hyperopt scikit-learn

CMD python optimize.py
