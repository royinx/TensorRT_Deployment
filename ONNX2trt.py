import tensorrt as trt
'''
				TensorRT				Pytorch / TF / Keras / MXNet / Caffe2
architecture: 	simplified network 		general network for train/eval/test 
lang: 			C++ 					python
dtype: 			FP16 / int8 			FP32
Accuracy:		79.x / 76.x				80

'''

from argparse import ArgumentParser


def GiB(val):
    return val * 1 << 30

def load_onnx_model1(args):
	# with builder = trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
	# 	with open(model_path, 'rb') as model:
	# 		parser.parse(model.read())
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
	return trt_network

def load_onnx_model2(args):
	with trt.Builder(args.TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, args.TRT_LOGGER) as parser:
		with open(args.onnx_model_name, 'rb') as model:
			parser.parse(model.read())
	return builder.build_cuda_engine(network)
def main():
	args = ArgumentParser().parse_args()
	args.onnx_model_name = "torch_model.onnx"
	args.model_dtype = trt.float32
	args.input_shape = (3,224,224) # (c,h,w)
	args.batch_size = 16

	args.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	'''
	tensorrt.DataType.FLOAT 	tensorrt.float32
	tensorrt.DataType.HALF 		tensorrt.float16
	tensorrt.DataType.INT32		tensorrt.int32
	tensorrt.DataType.INT8 		tensorrt.int8
	'''

	import inspect
	# for i, item in enumerate(inspect.getmembers(load_onnx_model1(args))):
		# print(item)
	print('*******************')
	for i, item in enumerate(inspect.getmembers(load_onnx_model2(args))):
		print(item)	


	# print(load_onnx_model2)

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


	'''
	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
	    builder.max_workspace_size = GiB(2)

		dataLayer = network.add_input('data',args.model_dtype,args.input_shape)

		network.mark_output(outputLayer.get_output(0))

		return builder.build_cuda_engine(network)
	context = engine.create_execution_context()
	context.execute_async(bindings=[d_input,d_output])
	modelstream = engine.serialize()
	trt.utils.write_engine_to_file(args.trt_model_name, modelstream)
	engine.destroy()
	builder.destroy()
	'''

if __name__ == '__main__':
	main()