import torch 
from torch.autograd import Variable
# import models
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Example of running detection, tracking and heatmap generation")
    # parser.add_argument('--display', action='store_true', help='display the video')
    # parser.add_argument('--monitor', action='store_true', help='Monitor the CPU and GPU resources usage')
    # parser.add_argument('--log_level', type=str, default='WARNING', choices=LOG_LEVELS.keys())
    # parser.add_argument('--video', type=str, help='video path or path to directory')
    # parser.add_argument('--video_level', type=int, default=0, help='number of directory level to traverse, 0 for reading a video file directly')
    # parser.add_argument('--stream', type=str, help='stream url')
    # parser.add_argument('--base_gui_config', type=str, default=os.path.join(project_path, 'base_config.json'), help='base gui config json file path')
    # parser.add_argument('--tracker_gui_config', type=str, default=os.path.join(project_path, 'tracker_config.json'), help='tracker gui config json file path')
    # parser.add_argument('--heatmap_gui_config', type=str, default=os.path.join(project_path, 'heatmap_config.json'), help='heatmap gui config json file path')
    # parser.add_argument('--detector_config', type=str, default=os.path.join(project_path, 'yolo_config.json'), help='detector config json file path')
    # parser.add_argument('--tracker_config', type=str, default=os.path.join(project_path, 'euclidean_distance_tracker.json'), help='tracker config json file path')
    # parser.add_argument('--video_out', type=str, help='file path for video output')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cuda', type=int, default=6)
    parser.add_argument('--input',type = str ,dest='framework_model_path', help='framework model path')
    parser.add_argument('--output',type = str ,dest='ONNX_path', default=None, help='ONNX output path')
    parser.add_argument('--framework',type = str ,dest='framework' ,default='pytorch', help='model framework type')
    args = parser.parse_args()


class CFG(object):
	def __init__(self, args):

		self.framework = args.framework.lower()
		self.input_path = args.framework_model_path # pytorch 
		if args.ONNX_path:
			self.  = args.ONNX_path # pytorch 
		else:
			model_name = self.framework_model_path.split('.')[-2].split('/')[-1] # gender_model
 			self.output_path = 'weights/{}.onnx'.format(model_name)

		self.input_shape = (3,224,224)  # default
		self.batch_size = args.batch_size 
		self.device = torch.device('cuda:{}'.format(args.cuda))

def torch_to_ONNX(cfg):
	model = torch.load(cfg.framework_model_path)
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
		torch_to_ONNX(args)

# py convert_ONNX_fullmodel.py weights/full_models/gender_full_model.pt 16