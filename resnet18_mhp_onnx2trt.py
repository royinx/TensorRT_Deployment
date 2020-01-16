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

import cv2

class CFG(object):
	"""docstring for cfg"""
	def __init__(self,args):
		self.onnx_file_path = args.model_path # gender_model.onnx
		self.model_name = self.onnx_file_path.split('.')[0].split('/')[-1]
		self.engine_file_path = args.model_path
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
			

			print(network.num_layers)
			network.mark_output(network.get_layer(network.num_layers-1).get_output(0))
			
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

def normalize(batch_img: np.array):  # support 
	batch_img = np.true_divide(batch_img,255)
	mean = np.array([0.485, 0.456, 0.406]).reshape(1,3,1,1)
	std = np.array([0.229, 0.224, 0.225]).reshape(1,3,1,1)
	batch_img2 = np.subtract(batch_img,mean)
	batch_img3 = np.true_divide(batch_img2,std)
	return batch_img

def batch_resize( images :list): # [HWC, HWC, HWC, HWC]
	resize_shape = (512,320)
	temp_images = np.array([cv2.resize(image, resize_shape) for image in images]) # cv2.INTER_LINEAR) NHWC
	batch_image = np.transpose(temp_images,(0,3,1,2)) # RGB  , NCHW
	return batch_image # array(N,C,H,W)

def decode_segmap(image, nc=21):

	# with open('testing/colors.txt') as infile:
	# 	label_colors = [line.split('\n')[0]for line in infile.readlines()]
	# 	label_colors = np.array([[int(x)for x in color.split(" ")] for color in label_colors])

	label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
 
	r = np.zeros_like(image).astype(np.uint8)
	g = np.zeros_like(image).astype(np.uint8)
	b = np.zeros_like(image).astype(np.uint8)

	for l in range(0, nc):
		idx = image == l
		r[idx] = label_colors[l, 0]
		g[idx] = label_colors[l, 1]
		b[idx] = label_colors[l, 2]
	 
	rgb = np.stack([r, g, b], axis=2)
	return rgb

def main():
	args = parse_args()
	cfg = CFG(args)
	'''
	tensorrt.DataType.FLOAT 	tensorrt.float32
	tensorrt.DataType.HALF 		tensorrt.float16
	tensorrt.DataType.INT32		tensorrt.int32
	tensorrt.DataType.INT8 		tensorrt.int8
	'''

	# assert os.path.exists(args.model_path)

	output_shapes = (64,21,10,16)

	input_img = cv2.imread('trump.jpg') # BGR  , HWC 
	ori_shape = input_img.shape
	print(ori_shape)


	input_img = input_img[:,:,[2,1,0]] # BGR - RGB  , HWC 


	# bgr = input_img[:,:,::-1] # RGB - BGR  , HWC 
	# cv2.imwrite("testing/test2.jpg",bgr)


	batch_img = list(np.tile(input_img,[64,1,1,1]))

	# pre-processing
	print(1,64,batch_img[0].shape)
	batch_img = batch_resize(batch_img)
	print(2,batch_img.shape)
	batch_img = normalize(batch_img)
	print(3,batch_img.shape)

	# TensorRT
	batch_img = np.array(batch_img, dtype=np.float32, order='C')
	with get_engine(args, cfg) as engine, engine.create_execution_context() as context:
		inputs, outputs, bindings, stream = common.allocate_buffers(engine)

		inputs[0].host = batch_img

		trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size = args.batch_size)

		print(trt_outputs)

	trt_outputs = trt_outputs[0].reshape(output_shapes)
	np.save('trt_outputs.npy',trt_outputs)
	print(trt_outputs.shape)
	rs = trt_outputs[0]
	print(rs.shape)




	# om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

	om = np.argmax(rs,axis = 0)
	print(om.shape)

	rgb = decode_segmap(om)

	bgr = rgb[:,:,::-1] # RGB - BGR
	# rgb = rgb[...,[2,0,1]] # RGB2BGR
	
	print('rgb',bgr.shape)
	frame = cv2.resize(bgr, (ori_shape[0],ori_shape[1]), interpolation=cv2.INTER_LINEAR)
	frame = np.transpose(frame,(1,0,2)) # BGR  , HWC
	cv2.imwrite("testing/test.jpg",frame)

	# import matplotlib.pyplot as plt
	# plt.imshow(rgb); plt.show()
	exit()



	# batch_img = np.ascontiguousarray(batch_img)
	# temp_img = temp_img.flatten()

	# get_engine(args,cfg)

	# print(trt_outputs)
	# Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
	# print(trt_outputs.shape)
	# for trt_output in trt_outputs:
	# 	print(trt_output)





	# om = np.argmax(trt_outputs)

	# with open('testing/colors.txt') as infile:
	# 	classes = [line.split('\n')[0]for line in infile.readlines()]
	# 	classes = np.array([[int(x)for x in shape.split(" ")] for shape in classes])
	# print(classes.shape)

	for idx, _class in enumerate(classes):


		'''
		print(idx, _class)
		# frame = np.array([np.ones((10,16))* RGB for RGB in _class])
		# print(trt_outputs[idx])
		frame = np.multiply(trt_outputs[idx],_class.reshape(3,1,1))  # RGB  , CHW
		
		print(frame.shape)
		print(frame)
		# frame = np.dot(frame,trt_outputs[0][idx])
		# print(frame)
	# for idx,value in enumerate(trt_outputs[0]):
		frame = np.transpose(frame,(1,2,0)) # RGB  , HWC
		print(frame.shape, ori_shape)
		frame = cv2.resize(frame, (ori_shape[0],ori_shape[1]), interpolation=cv2.INTER_LINEAR)

		# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		frame = frame[...,[2,0,1]]

		# normalise
		frame *= (255.0/frame.max())

		print(frame)
		# cv2.imwrite("testing/layer_{}.jpg".format(idx),frame)
		'''
		temp = cv2.resize(trt_outputs[idx], (ori_shape[1],ori_shape[0]), interpolation=cv2.INTER_LINEAR)
		# temp += 100
		# print(temp.max(),temp.min())
		# cv2.imwrite("testing/layer_{}.jpg".format(idx),temp)
	# cv2.imwrite("testing/test.jpg",input_img[0])

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



# docker run --privileged --rm -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -v ~/Desktop/python:/py -w /py --runtime=nvidia nvcr.io/nvidia/tensorrt:19.09-py3  bash
# cd TensorRT_Deployment/ && pip3 install opencv-python matplotlib &&  apt-get install -y libsm6 libxext6 libxrender1
# python3 convert_trt_val.py --model resnet18.trt
