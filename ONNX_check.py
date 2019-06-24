import onnx
import sys
onnx_model_path = sys.argv[1]
model = onnx.load(onnx_model_path)
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))

# py ONNX_check.py gender_model.onnx