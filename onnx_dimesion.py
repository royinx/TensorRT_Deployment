import onnx 
model = onnx.load('model.onnx')
model.graph.input[0].type.tensor_type.shjape.dim[0].dim_param = '?'
onnx.save(model,'demo.onnx')

