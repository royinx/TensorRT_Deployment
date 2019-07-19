import torch 
from models import *
from torch.autograd import Variable
import argparse
import sys

def parse_args():
	parser = argparse.ArgumentParser(description="Example of convert framework to TensorRT")
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--cuda', type=int, default=0)
	parser.add_argument('--input',type = str ,dest='framework_model_path', help='framework model path')
	parser.add_argument('--output',type = str ,dest='output_path', default=None, help='new model path')
	parser.add_argument('--framework',type = str ,dest='framework' ,default='pytorch', help='model framework type')
	args = parser.parse_args()
	return args

class CFG(object):
	def __init__(self, args):

		self.framework = args.framework.lower()
		self.input_path = args.framework_model_path # pytorch 
		if args.output_path:
			self.output_path = args.output_path # pytorch 
		else:
			model_name = self.framework_model_path.split('.')[-2].split('/')[-1] # gender_model
			self.output_path = 'transform/onnx/{}.onnx'.format(model_name)

		self.input_shape = (3,224,224)  # default
		self.batch_size = args.batch_size 
		self.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available else 'cpu')

def torch_to_ONNX(cfg:CFG):
	print(cfg.input_path)
	model = torch.load(cfg.input_path)
	print(type(model))
	model.to(cfg.device)
	print('================ Device: {} ================'.format(cfg.device))

	model.eval()
	dummy_input = Variable(torch.randn(cfg.batch_size, *cfg.input_shape, device=cfg.device))  # (batch , ch , h , w)    N,C,H,W
	print('****')
	output = torch.onnx.export(	model,
								dummy_input,
								cfg.output_path,
								output_names = ['output_classes'],
								verbose=True)#.cpu()
	print("Export of {} complete".format(cfg.output_path))

def torch_pt_to_pth(cfg:CFG):
	print(cfg.input_path)
	model = torch.load(cfg.input_path)
	print(type(model))
	model.to(cfg.device)
	print('================ Device: {} ================'.format(cfg.device))

	model.eval()
	dummy_input = torch.randn(cfg.batch_size, *cfg.input_shape, device=cfg.device)  # (batch , ch , h , w)    NCHW
	print(cfg.input_shape)
	traced_script_module = torch.jit.trace(model, dummy_input)
	print(type(traced_script_module))
	torch.jit.save(traced_script_module,cfg.output_path)
	print("Export of {} complete".format(cfg.output_path))

if __name__ == '__main__':
	args = parse_args()
	cfg = CFG(args)
	model_out_suffix = cfg.output_path.split('.')[-1]
	model_in_suffix = cfg.input_path.split('.')[-1]
	if model_out_suffix == 'onnx':
		if cfg.framework == 'pytorch':
			torch_to_ONNX(cfg)
	elif model_out_suffix == 'pth':
		if cfg.framework == 'pytorch' and model_in_suffix == 'pt':
			torch_pt_to_pth(cfg)
	else:
		pass

'''
py convert_ONNX_fullmodel.py 	--cuda 6 \
								--batch_size 1024 \
								--input transform/framework_weight/yolov3.pt \
								--output transform/onnx/yolov3.pt
'''
# py converter.py 	--cuda 0 --batch_size 32 --input transform/framework_weight/yolov3-ssp.pt --output transform/onnx/yolov3-ssp.onnx
# py converter.py 	--cuda 0 --batch_size 1 --input transform/framework_weight/age_full_model.pt --output transform/onnx/age_full_model.onnx
# py converter.py 	--cuda 0 --batch_size 1 --input transform/framework_weight/yolov3-ssp.pt --output transform/other/yolov3-ssp.pth
# py converter.py 	--cuda 0 --batch_size 1 --input transform/framework_weight/gender_full.pt --output transform/other/gender_full.pth