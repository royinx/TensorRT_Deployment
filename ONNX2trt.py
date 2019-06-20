import tensorrt as trt
'''
				TensorRT				Pytorch / TF / Keras / MXNet / Caffe2
architecture: 	simplified network 		general network for train/eval/test 
lang: 			C++ 					python
dtype: 			FP16 / int8 			FP32
Accuracy:		79.x / 76.x				80

'''
# with builder = trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
# 	with open(model_path, 'rb') as model:
# 		parser.parse(model.read())

from argparse import ArgumentParser
args = ArgumentParser().parse_args()
args.onnx_model_name = "model.onnx"
args.model_dtype = trt.float32
'''
tensorrt.DataType.FLOAT 	tensorrt.float32
tensorrt.DataType.HALF 		tensorrt.float16
tensorrt.DataType.INT32		tensorrt.int32
tensorrt.DataType.INT8 		tensorrt.int8
'''

from tensorrt.parsers import onnxparser
apex = onnxparser.create_onnxconfig()

apex.set_model_file_name(args.onnx_model_name)
apex.set_model_dtype(args.model_dtype)
apex.set_print_layer_info(True)
trt_parser = onnxparser.create_onnxparser(apex)
data_type = apex.get_model_dtype()
onnx_filename = apex.get_model_file_name()
trt_parser.parse(onnx_filename, data_type)

trt_parser.convert_to_trtnetwork()
trt_network = trt_parser.get_trtnetwork()


# calibration_files = create_calibration_dataset()
batchstream = calibrator.ImageBatchStream(args)
int8_calibrator = calibrator.PythonEntropyCalibrator(["data"], batchstream)

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

builder = trt.infer.create_infer_builder(G_LOGGER)
builder.set_max_batch_size(16)
builder.set_max_workspace_size(1 << 20)
builder.set_int8_calibrator(int8_calibrator)
builder.set_int8_mode(True)
engine = builder.build_cuda_engine(trt_network)
modelstream = engine.serialize()
trt.utils.write_engine_to_file(args.trt_model_name, modelstream)
engine.destroy()
builder.destroy()



# class ModelData(object):
#     MODEL_PATH = "1.onnx"
#     INPUT_SHAPE = (3, 224, 224)
#     # We can convert TensorRT data types to numpy types with trt.nptype()
#     DTYPE = trt.int8

'''
# Template
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)

network = builder.create_network()
dataLayer = network.add_input('data',trt.DataType.FLOAT,(c,h,w))
# Add network layer
network.mark_output(outputLayer.get_output(0))

engine = builder.build_cuda_engine(network)
context = engine.create_execution_context()
context.execute_async(bindings=[d_input,d_output])
'''

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)

network = builder.create_network()
dataLayer = network.add_input('data',trt.DataType.FLOAT,(c,h,w))
# Add network layer
network.mark_output(outputLayer.get_output(0))

engine = builder.build_cuda_engine(network)
context = engine.create_execution_context()
context.execute_async(bindings=[d_input,d_output])
modelstream = engine.serialize()
trt.utils.write_engine_to_file(args.trt_model_name, modelstream)
engine.destroy()
builder.destroy()

