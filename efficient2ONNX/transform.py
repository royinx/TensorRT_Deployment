#========================torch to onnx  Emotion  ===================================================
'''
import torch
from efficientnet_pytorch import EfficientNet
import sys
import torch.nn as nn


# Specify which model to use
model_file = sys.argv[1]
model_name = model_file.split('.')[0]
output_name = "{}.onnx".format(model_name)
backbone = 'efficientnet-b7'
image_size = EfficientNet.get_image_size(backbone)
print('Image size: ', image_size)

# Load model
model = EfficientNet.from_pretrained(backbone)
num_features = model._fc.in_features
print(num_features)
# model._fc = nn.Linear(num_features, 4) #for emotion
model._fc = nn.Linear(num_features, 7)
model.load_state_dict(torch.load(model_file)['model'])
model.set_swish(memory_efficient=False)

model.eval()
print('Model image size: ', model._global_params.image_size)


# torch.save(model, '{}_entire.pth'.format(model_name))

# Dummy input for ONNX
dummy_input = torch.randn(1, 3, 224, 224)

# Export with ONNX
torch.onnx.export(model, dummy_input, output_name,input_names=['input'], output_names = ['output_classes'], verbose=True)

# Test export 
import onnx

model = onnx.load(output_name)

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)

# Things should look good

print("Model saved as {}".format(output_name))

# python transform.py <model_file>
'''

#========================torch to onnx   Age  ===================================================
'''


# checkpoint = torch.load(model_file)
# model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2)
# new_state_dict = OrderedDict()
# for k, v in checkpoint['model'].items():
#     name = k.replace("module.", "")
#     new_state_dict[name]= v
# model.load_state_dict(new_state_dict)

import torch
from efficientnet_pytorch import EfficientNet
import sys
import torch.nn as nn
from collections import OrderedDict


# Specify which model to use
model_file = sys.argv[1]
model_name = model_file.split('.')[0]
output_name = "{}.onnx".format(model_name)
backbone = 'efficientnet-b7'
image_size = EfficientNet.get_image_size(backbone)
print('Image size: ', image_size)

# Load model
model = EfficientNet.from_pretrained(backbone, num_classes = 7)

checkpoint = torch.load(model_file)
new_state_dict = OrderedDict()
for k, v in checkpoint['model'].items():
    name = k.replace("module.", "")
    new_state_dict[name]= v
# num_features = model._fc.in_features
# print(num_features)
# model._fc = nn.Linear(num_features, 4) #for emotion
# model._fc = nn.Linear(num_features, 7)
model.load_state_dict(new_state_dict)
model.set_swish(memory_efficient=False)

model.eval()
print('Model image size: ', model._global_params.image_size)


# torch.save(model, '{}_entire.pth'.format(model_name))

# Dummy input for ONNX
dummy_input = torch.randn(1, 3, 224, 224)

# Export with ONNX
torch.onnx.export(model, dummy_input, output_name,input_names=['input.1'], output_names = ['output_classes'], verbose=True)

# Test export 
import onnx

model = onnx.load(output_name)

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)

# Things should look good

print("Model saved as {}".format(output_name))

# python transform.py <model_file>
'''


#========================torch to onnx   Gender  ===================================================
# '''


import torch
from efficientnet_pytorch import EfficientNet
import sys
import torch.nn as nn
from collections import OrderedDict


# Specify which model to use
model_file = sys.argv[1]
model_name = model_file.split('.')[0]
output_name = "{}.onnx".format(model_name)
backbone = 'efficientnet-b7'
image_size = EfficientNet.get_image_size(backbone)
print('Image size: ', image_size)

# Load model
model = EfficientNet.from_pretrained(backbone, num_classes = 2)

checkpoint = torch.load(model_file)
# new_state_dict = OrderedDict()
# for k, v in checkpoint['model'].items():
#     name = k.replace("module.", "")
#     new_state_dict[name]= v


# num_features = model._fc.in_features
# print(num_features)
# model._fc = nn.Linear(num_features, 4) #for emotion
# model._fc = nn.Linear(num_features, 7)
model.load_state_dict(checkpoint)
model.set_swish(memory_efficient=False)

model.eval()
print('Model image size: ', model._global_params.image_size)


# torch.save(model, '{}_entire.pth'.format(model_name))

# Dummy input for ONNX
dummy_input = torch.randn(1, 3, 224, 224)

# Export with ONNX
torch.onnx.export(model, dummy_input, output_name,input_names=['input.1'], output_names = ['output_classes'], verbose=True)

# Test export 
import onnx

model = onnx.load(output_name)

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)

# Things should look good

print("Model saved as {}".format(output_name))

# python transform.py <model_file>
# '''


#==========================torch to trt================================


# import torch
# from efficientnet_pytorch import EfficientNet
# from torch2trt import torch2trt
# import sys
# import torch.nn as nn

# # Specify which model to use
# model_file = sys.argv[1]
# model_name = model_file.split('.')[0]
# output_name = "{}.onnx".format(model_name)
# backbone = 'efficientnet-b7'
# image_size = EfficientNet.get_image_size(backbone)
# print('Image size: ', image_size)

# # Load model
# model = EfficientNet.from_pretrained(backbone)
# num_features = model._fc.in_features
# print(num_features)
# model._fc = nn.Linear(num_features, 4)
# model.load_state_dict(torch.load(model_file)['model'])
# model.set_swish(memory_efficient=False)

# model.eval().cuda()

# # create example data
# x = torch.randn((1, 3, 224, 224)).cuda()

# # convert to TensorRT feeding sample data as input
# model_trt = torch2trt(model, [x])


#======================================================================================




# '''

'''
import torch
from efficientnet_pytorch import EfficientNet

# Specify which model to use
model_name = 'efficientnet-b1'
image_size = EfficientNet.get_image_size(model_name)
print('Image size: ', image_size)

# Load model
model = EfficientNet.from_pretrained(model_name)
# model = EfficientNet.from_name(model_name)
model.eval()
model.set_swish(memory_efficient=False)
print('Model image size: ', model._global_params.image_size)

# Dummy input for ONNX
dummy_input = torch.randn(1, 3, 224, 224)


# Export with ONNX
torch.onnx.export(model, dummy_input, "efficientnet-b1.onnx", verbose=True)

# Test export 
import onnx

model = onnx.load("efficientnet-b1.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)

print("Done")

# Things should look good
'''