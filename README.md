## TensorRT

docker pull nvcr.io/nvidia/tensorrt:19.12-py3

docker run --privileged --rm -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -v ~/Desktop/python:/py -w /py --runtime=nvidia nvcr.io/nvidia/tensorrt:19.12-py3  bash

