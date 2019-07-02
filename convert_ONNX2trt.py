import tensorrt as trt   # Version TensotRT '5.1.5.0'
'''
				TensorRT				Pytorch / TF / Keras / MXNet / Caffe2
architecture: 	simplified network 		general network for train/eval/test 
lang: 			C++ 					python
dtype: 			FP16 / int8 			FP32
Accuracy:		79.x / 76.x				80

'''
import numpy as np
from argparse import ArgumentParser
import os 
from utils import common


class CFG(object):
	"""docstring for cfg"""
	def __init__(self,args):
		self.onnx_file_path = args.model_path # gender_model.onnx
		self.model_name = self.onnx_file_path.split('.')[0].split('/')[-1]
		self.engine_file_path = self.model_name + '.trt'
		self.model_dtype = trt.float16
		self.input_shape = (3,224,224) # (c,h,w)
		self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	# def __init__(self,path):
	# 	self.model_name = 'age'
	# 	self.onnx_file_path = .onnx
	# 	self.engine_file_path = .trt
	# 	self.model_dtype = trt.float32
	# 	self.input_shape = (3,224,224) # (c,h,w)
	#	self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

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
			# Set the model dtype to half , fp16
			parser.parse(model.read())
	return builder.build_cuda_engine(network)



def get_engine(args,cfg):
	"""Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
	def build_engine():
		"""Takes an ONNX file and creates a TensorRT engine to run inference with"""
		with trt.Builder(cfg.TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, cfg.TRT_LOGGER) as parser:
			builder.max_workspace_size = GiB(args.model_memory)
			builder.max_batch_size = args.max_batch_size
			
			if args.precision == 'fp16':
				# set to fp16 
				print('force to fp16')
				builder.fp16_mode = True
				builder.strict_type_constraints = True
			elif args.precision == 'int8':
				# set to int8
				builder.int8_mode = True

				'''
				NUM_IMAGES_PER_BATCH = 5 
				batch = ImageBatchStream(NUM_IMAGES_PER_BATCH, calibration_files)
				Int8_calibration = EntropyCalibrator(['input_node_name'],batchstream)
				trt_builder.int8_calibrator = Int8_calibrator
				'''
			else:
				pass

			# Parse model file
			if not os.path.exists(cfg.onnx_file_path):
				print('ONNX file {} not found, please run pytorch2ONNX.py first to generate it.'.format(cfg.onnx_file_path))
				exit(0)
			print('Loading ONNX file from path {}...'.format(cfg.onnx_file_path))
			with open(cfg.onnx_file_path, 'rb') as model:
				print('Beginning ONNX file parsing')
				parser.parse(model.read())
			print('Completed parsing of ONNX file')
			print('Building an engine from file {}; this may take a while...'.format(cfg.onnx_file_path))
			engine = builder.build_cuda_engine(network)
			print("Completed creating Engine")
			with open(cfg.engine_file_path, "wb") as f:
				f.write(engine.serialize())
			return engine

	if not args.build and os.path.exists(cfg.engine_file_path):
		# If a serialized engine exists, use it instead of building an engine.
		print("Reading engine from file {}".format(cfg.engine_file_path))
		with open(cfg.engine_file_path, "rb") as f, trt.Runtime(cfg.TRT_LOGGER) as runtime:
			return runtime.deserialize_cuda_engine(f.read())
	else:
		print('------------------ Building the Engine ------------------')
		print("Building engine from file {}".format(cfg.onnx_file_path))
		return build_engine()
	#both are returning deserialize cuda engine


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
	

def img_input():
	return np.load('data/img.npy')

def parse_args(argv=None):
	parser = ArgumentParser()
	parser.add_argument('-p','--precision', default='fp16', type=str, dest='precision',
						help='inference precision, fp32, fp16, int8 etc.')
	parser.add_argument('--model', type=str, dest='model_path',
						help='model path')
	parser.add_argument('--model_memory', type=int, dest='model_memory',
						help='engine memory')
	parser.add_argument('--model_max_batch_size', type=int, dest='max_batch_size',
						help='engine batch')
	# parser.add_argument('-MP','--model_path', default='fp16', type=str, dest='engine_file_path',
	# 					help='Path to the model')
	parser.add_argument('--build', action='store_true',dest='build',
						help='build the model, (model will be overwrite if model exists)')
	parser.add_argument('--batch',default = 1, type=int ,dest='batch_size',
						help='model batch size')
	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	cfg = CFG(args)
	'''
	tensorrt.DataType.FLOAT 	tensorrt.float32
	tensorrt.DataType.HALF 		tensorrt.float16
	tensorrt.DataType.INT32		tensorrt.int32
	tensorrt.DataType.INT8 		tensorrt.int8
	'''

	assert os.path.exists(args.model_path)

	trt_outputs = []
	output_shapes = (args.batch_size,)
	# get_engine(args,cfg)
	with get_engine(args, cfg) as engine, engine.create_execution_context() as context:
		inputs, outputs, bindings, stream = common.allocate_buffers(engine)
		# Do inference
		print('Running inference on image {}...'.format('data/img.npy'))
		# Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
		# inputs[0].host = img_input()
		temp_img = img_input().flatten()

		for _ in range(args.batch_size):
			if _ == 0:
				images = temp_img
			else:
				images = np.concatenate((images,temp_img),axis=None)
		# for idx,item in enumerate(range(1000)):
		# 	inputs[idx].host = temp_img
		print('before: ',inputs)
		print('------------------------------------')
		print('inputs[0].host: {}'.format(inputs[0].host.shape))
		print('images: {}'.format(images.shape))


		inputs[0].host = images
		print('------------------------------------')
		print(len(trt_outputs))
		print('after: ',inputs)

		import time
		start = time.time()
		trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size = args.batch_size)
		print(time.time()-start)

		print(trt_outputs[0].shape)
	# print(trt_outputs)
	# Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
	trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
	print(trt_outputs.shape)
	# trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]




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


# py convert_ONNX2trt.py --precision fp16 --build --batch 16 --model gender_model.onnx --model_memory 2 --model_max_batch_size 512
# py convert_ONNX2trt.py --precision fp16 --batch 16 --model gender_model.onnx --model_memory 2 --model_max_batch_size 512
