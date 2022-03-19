# INSTALL 
```
DOCKER_BUILDKIT=1 docker build -t mnist -f Dockerfile .  
```
# RUN
```
docker run -it --rm -u $(id -u):$(id -g) --gpus all -v "$PWD":/workspace/mnist_kaggle mnist
```

# GET DATA
```
kaggle competitions download -c digit-recognizer
```
