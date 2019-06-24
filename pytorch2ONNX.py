import torch 
from torch.autograd import Variable
import models
import sys


framework_model_path = sys.argv[1]
model_name = 'weights/gender_model.pt'.split('.')[-2].split('/')[-1] # gender_model

# Config
input_shape = (3,224,224)
batch_size = int(sys.argv[2])
model_onnx_path = '{}.onnx'.format(model_name)

# Load the model
device = torch.device("cuda")
model = models.resnet.resnext101_32x8d(pretrained = False,progress = True)
model.load_state_dict(torch.load(framework_model_path,map_location=device),strict = False)
model.to(device)

# Export the model to an ONNX file
dummy_input = Variable(torch.randn(batch_size, *input_shape, device='cuda'))  # (batch , ch , h , w)    N,C,H,W
print('****')

output = torch.onnx.export(	model,
							dummy_input,
							model_onnx_path,
							output_names = ['output_classes'],
							verbose=True)

print("Export of {} complete".format(model_onnx_path))

# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]

# print(input_names, output_names)
# torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

# py pytorch2ONNX.py weights/gender_model.pt 16