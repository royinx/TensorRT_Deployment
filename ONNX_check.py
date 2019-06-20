import onnx
model = onnx.load('torch_model.onnx')
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))