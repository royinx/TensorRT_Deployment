import onnx
onnx_model = onnx.load('model.onnx')
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '8'
onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = '8'
print(onnx_model.graph.input[0])
print(onnx_model.graph.output[0])
onnx.save(onnx_model, 'batch8_model.onnx')