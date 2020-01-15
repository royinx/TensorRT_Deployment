# '''
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
model._fc = nn.Linear(num_features, 4)
model.load_state_dict(torch.load(model_file)['model'])

model.eval()
print('Model image size: ', model._global_params.image_size)

# Dummy input for ONNX
dummy_input = torch.randn(10, 3, 224, 224)

# Export with ONNX
torch.onnx.export(model, dummy_input, output_name, verbose=True)

# Test export 
import onnx

model = onnx.load(output_name)

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)

# Things should look good

# python transform.py 
# '''

'''
import torch
from efficientnet_pytorch import EfficientNet

# Specify which model to use
model_name = 'efficientnet-b7'
image_size = EfficientNet.get_image_size(model_name)
print('Image size: ', image_size)

# Load model
model = EfficientNet.from_pretrained(model_name)
model.eval()
print('Model image size: ', model._global_params.image_size)

# Dummy input for ONNX
dummy_input = torch.randn(10, 3, 224, 224)

# Export with ONNX
torch.onnx.export(model, dummy_input, "efficientnet-b1.onnx", verbose=True)

# Test export 
import onnx

model = onnx.load("efficientnet-b1.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)

# Things should look good
'''