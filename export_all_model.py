from app_v3 import MODEL_REGISTRY
from analysis import analyze_model
import tempfile

for name, (cls, shape) in MODEL_REGISTRY.items():
    # d = tempfile.mkdtemp()
    d = "./onnx_model"
    try:
        _, p, m, onnx_path = analyze_model(cls(), shape, save_dir=d, export_onnx=True)
        print(f"{name:20s} params={p:>10,} macs={m:>12,} onnx={'OK' if onnx_path else 'FAIL'}")
    except Exception as e:
        print(f"{name:20s} 分析失败: {e}")