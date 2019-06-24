import os
from random import randint
import numpy as np
 
try:
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module({})
Please make sure you have pycuda and the example dependencies installed.
sudo apt-get install python(3)-pycuda
pip install tensorrt[examples]""".format(err))
 
try:
    from PIL import Image
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Pillow installed.
For installation instructions, see:
http://pillow.readthedocs.io/en/stable/installation.html""".format(err))
 
import mnist
 
try:
    import torch
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have PyTorch installed.
For installation instructions, see:
http://pytorch.org/""".format(err))
 
# TensorRT must be imported after any frameworks in the case where
# the framework has incorrect dependencies setup and is not updated
# to use the versions of libraries that TensorRT imports.
try:
    import tensorrt as trt
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have the TensorRT Library installed
and accessible in your LD_LIBRARY_PATH""".format(err))
 
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
 
ITERATIONS = 10
INPUT_LAYERS = ["data"]
OUTPUT_LAYERS = ['prob']
INPUT_H = 28
INPUT_W = 28
OUTPUT_SIZE = 10
 
def create_pytorch_engine(max_batch_size, builder, dt, model):
    network = builder.create_network()
 
    data = network.add_input(INPUT_LAYERS[0], dt, (1, INPUT_H, INPUT_W))
    assert(data)
 
    # #-------------
    # Template
    # w0 = params['conv0_weight'].asnumpy().reshape(-1)
    # b0 = params['conv0_bias'].asnumpy().reshape(-1)
    # conv1 = network.add_convolution(dataLayer, 32, (3,3),  trt.Weights(w0),trt.Weights(b0))
    # conv1.stride = (1,1)
    # conv1.padding = (1,1)

    conv1_w = model['conv1.weight'].cpu().numpy().reshape(-1)
    conv1_b = model['conv1.bias'].cpu().numpy().reshape(-1)
    conv1 = network.add_convolution(data, 20, (5,5),  conv1_w, conv1_b)
    assert(conv1)
    conv1.set_stride((1,1))
 
    #-------------
    pool1 = network.add_pooling(conv1.get_output(0), trt.infer.PoolingType.MAX, (2,2))
    assert(pool1)
    pool1.set_stride((2,2))
 
    #-------------
    conv2_w = model['conv2.weight'].cpu().numpy().reshape(-1)
    conv2_b = model['conv2.bias'].cpu().numpy().reshape(-1)
    conv2 = network.add_convolution(pool1.get_output(0), 50, (5,5), conv2_w, conv2_b)
    assert(conv2)
    conv2.set_stride((1,1))
 
    #-------------
    pool2 = network.add_pooling(conv2.get_output(0), trt.infer.PoolingType.MAX, (2,2))
    assert(pool2)
    pool2.set_stride((2,2))
 
    #-------------
    fc1_w = model['fc1.weight'].cpu().numpy().reshape(-1)
    fc1_b = model['fc1.bias'].cpu().numpy().reshape(-1)
    fc1 = network.add_fully_connected(pool2.get_output(0), 500, fc1_w, fc1_b)
    assert(fc1)
 
    #-------------
    relu1 = network.add_activation(fc1.get_output(0), trt.infer.ActivationType.RELU)
    assert(relu1)
 
    #-------------
    fc2_w = model['fc2.weight'].cpu().numpy().reshape(-1)
    fc2_b = model['fc2.bias'].cpu().numpy().reshape(-1)
    fc2 = network.add_fully_connected(relu1.get_output(0), OUTPUT_SIZE, fc2_w, fc2_b)
    assert(fc2)
 
    #-------------
    # Using log_softmax in training, cutting out log softmax here since no log softmax in TRT
    fc2.get_output(0).set_name(OUTPUT_LAYERS[0])
    network.mark_output(fc2.get_output(0))
 
 
    builder.set_max_batch_size(max_batch_size)
    builder.set_max_workspace_size(1 << 20)
 
    #builder.set_fp16_mode(True)
 
    engine = builder.build_cuda_engine(network)
    network.destroy()
 
    return engine
 
def model_to_engine(model, max_batch_size):
    builder = trt.infer.create_infer_builder(G_LOGGER)
    engine = create_pytorch_engine(max_batch_size, builder, trt.infer.DataType.FLOAT, model)
    assert(engine)
 
    modelstream = engine.serialize()
    engine.destroy()
    builder.destroy()
    return modelstream
 
# Run inference on device
def infer(context, input_img, output_size, batch_size):
    # Load engine
    engine = context.get_engine()
    assert(engine.get_nb_bindings() == 2)
    # Convert input data to Float32
    input_img = input_img.astype(np.float32)
    # Create output array to receive data
    output = np.empty(output_size, dtype = np.float32)
 
    # Allocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.nbytes)
    d_output = cuda.mem_alloc(batch_size * output.nbytes)
 
    bindings = [int(d_input), int(d_output)]
 
    stream = cuda.Stream()
 
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # Execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
 
    # Return predictions
    return output
 
def main():
    path = dir_path = os.path.dirname(os.path.realpath(__file__))
 
    # The mnist package is a simple PyTorch mnist example. mnist.learn() trains a network for
    # PyTorch's provided mnist dataset. mnist.get_trained_model() returns the state dictionary
    # of the trained model. We use this to demonstrate the full training to inference pipeline
    mnist.learn()
    model = mnist.get_trained_model()
 
    # Typically training and inference are seperated so using torch.save() and saving the
    # model's state dictionary and then using torch.load() to load the state dictionary
    #
    # e.g:
    # model = torch.load(path + "/trained_mnist.pyt")
    modelstream = model_to_engine(model, 1)
 
    runtime = trt.infer.create_infer_runtime(G_LOGGER)
    engine = runtime.deserialize_cuda_engine(modelstream.data(), modelstream.size(), None)
 
    if modelstream:
        modelstream.destroy()
 
    img, target = mnist.get_testcase()
    img = img.numpy()
    target = target.numpy()
    print("\n| TEST CASE | PREDICTION |")
    for i in range(ITERATIONS):
        img_in = img[i].ravel()
        target_in = target[i]
        context = engine.create_execution_context()
        out = infer(context, img_in, OUTPUT_SIZE, 1)
        print("|-----------|------------|")
        print("|     " + str(target_in) + "     |      " + str(np.argmax(out)) + "     |")
 
    print('')
    context.destroy()
    engine.destroy()
    runtime.destroy()
 
 
 
if __name__ == "__main__":
    main()
