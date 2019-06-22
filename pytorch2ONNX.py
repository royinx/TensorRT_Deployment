import torch 
from torch.autograd import Variable
import models

# Config
input_shape = (3,224,224)
batch_size = 16
model_onnx_path = 'torch_model.onnx'

# Load the model
device = torch.device("cuda")
model = models.resnet.resnext101_32x8d(pretrained = False,progress = True)
model.load_state_dict(torch.load('weights/weights/age_model.pt',map_location=device),strict = False)
model.to(device)

# Export the model to an ONNX file
dummy_input = Variable(torch.randn(batch_size, *input_shape, device='cuda'))  # (batch , ch , h , w)    N,C,H,W
print('****')

output = torch.onnx.export(	model,
							dummy_input,
							model_onnx_path,
							verbose=True)

print("Export of torch_model.onnx complete")



# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]

# print(input_names, output_names)
# torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)