import tensorrt as trt
'''
				TensorRT				Pytorch / TF / Keras / MXNet / Caffe2
architecture: 	simplified network 		general network for train/eval/test 
lang: 			C++ 					python
dtype: 			FP16 / int8 			FP32
Accuracy:		79.x / 76.x				80

'''

from argparse import ArgumentParser
import os 
import common

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



def get_engine(args):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(args.TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, args.TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(args.onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(args.onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(args.onnx_file_path))
            with open(args.onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(args.onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(args.engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(args.engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(args.engine_file_path))
        with open(args.engine_file_path, "rb") as f, trt.Runtime(args.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()




def main():
	args = ArgumentParser().parse_args()
	args.model_name = 'age'
	args.onnx_file_path = args.model_name + '_model.onnx'
	args.engine_file_path = args.model_name + '_model.trt'
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


	print('------------------ Building the Engine ------------------')
	print()

	trt_outputs = []
	get_engine(args)
    # with get_engine(args.onnx_file_path, args.engine_file_path) as engine, engine.create_execution_context() as context:
    #     inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    #     # Do inference
    #     print('Running inference on image {}...'.format(input_image_path))
    #     # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
    #     inputs[0].host = image
    #     trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
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