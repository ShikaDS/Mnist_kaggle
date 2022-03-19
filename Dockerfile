FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN pip install --upgrade pip setuptools wheel
RUN pip install gast==0.2.2 \
    numpy==1.19.5 \
    Keras==2.3.1 \
    pandas==1.4.1 \
    sklearn

WORKDIR /workspace/mnist_kaggle