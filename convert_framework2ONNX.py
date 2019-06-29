import torch 
from torch.autograd import Variable
import argparse
import sys

def parse_args():
	parser = argparse.ArgumentParser(description="Example of convert framework to TensorRT")
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--cuda', type=int, default=0)
	parser.add_argument('--input',type = str ,dest='framework_model_path', help='framework model path')
	parser.add_argument('--output',type = str ,dest='ONNX_path', default=None, help='ONNX output path')
	parser.add_argument('--framework',type = str ,dest='framework' ,default='pytorch', help='model framework type')
	args = parser.parse_args()
	return args

class CFG(object):
	def __init__(self, args):

		self.framework = args.framework.lower()
		self.input_path = args.framework_model_path # pytorch 
		if args.ONNX_path:
			self.output_path = args.ONNX_path # pytorch 
		else:
			model_name = self.framework_model_path.split('.')[-2].split('/')[-1] # gender_model
			self.output_path = 'weights/{}.onnx'.format(model_name)

		self.input_shape = (3,224,224)  # default
		self.batch_size = args.batch_size 
		self.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available else 'cpu')

def torch_to_ONNX(cfg:CFG):
	print(cfg.input_path)
	model = torch.load(cfg.input_path)
	print(type(model))
	model.to(cfg.device)

	model.eval()
	dummy_input = Variable(torch.randn(cfg.batch_size, *cfg.input_shape, device=cfg.device))  # (batch , ch , h , w)    N,C,H,W
	print('****')
	output = torch.onnx.export(	model,
								dummy_input,
								cfg.output_path,
								output_names = ['output_classes'],
								verbose=True)
	print("Export of {} complete".format(cfg.output_path))



if __name__ == '__main__':
	args = parse_args()
	cfg = CFG(args)
	if cfg.framework == 'pytorch':
		torch_to_ONNX(cfg)

'''
py convert_ONNX_fullmodel.py 	--cuda 6 \
								--batch_size 1024 \
								--input transform/framework_weight/yolov3.pt \
								--output transform/onnx/yolov3.pt
'''
# py convert_framework2ONNX.py 	--cuda 0 --batch_size 1024 --input transform/framework_weight/yolov3.pt --output transform/onnx/yolov3.onnx
# py convert_framework2ONNX.py 	--cuda 0 --batch_size 32 --input transform/framework_weight/age_full_model.pt --output transform/onnx/age_full_model.onnx