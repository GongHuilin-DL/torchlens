import onnx
m = onnx.load(r"C:\Users\ADMINI~1\AppData\Local\Temp\torchinfo_jju8qtwl\model.onnx")
print("opset:", m.opset_import[0].version)      # 应为 18
print("input:", m.graph.input[0].name)           # 应为 "input"
# 检查 batch 维是否为动态（dim_param 而非 dim_value）
print(m.graph.input[0].type.tensor_type.shape)