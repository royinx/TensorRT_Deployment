## Environment 

torch - 1.3.0.dev20190910
torchvision - 0.5.0a0+6ddda3a
efficientnet_pytorch - 0.5.1
onnx - 1.5.0

#### Steps

```shell
docker pull nvcr.io/nvidia/tensorrt:<yy>.<mm>-py3
#cd to dir TensorRT_Deployment/efficient2ONNX/

. pre.sh
python transform.py <model_file>
```