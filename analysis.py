# analysis.py
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import inspect
import tempfile


def _calculate_flops(layer_info):
    """估算单层的 FLOPs，填补激活层/池化层/归一化层的计算量空白"""
    macs = layer_info.macs
    class_name = layer_info.class_name
    flops = 2 * macs  # 默认：卷积和全连接层 FLOPs ≈ 2 * MACs
    
    # 对于不含乘加的操作，根据输出张量的元素个数估算 FLOPs
    if macs == 0 and hasattr(layer_info, 'output_size') and layer_info.output_size:
        try:
            num_elements = int(np.prod(layer_info.output_size))
            
            # 1. 激活函数层：约 1 FLOPs / element (如 max(0,x))
            activations = ['ReLU', 'ReLU6', 'LeakyReLU', 'Sigmoid', 'Tanh', 'GELU', 'SiLU', 'Hardswish', 'Hardsigmoid']
            if any(act in class_name for act in activations):
                flops = num_elements
            
            # 2. 归一化层：约 4 FLOPs / element (减均值、除方差、乘gamma、加beta)
            elif any(norm in class_name for norm in ['BatchNorm', 'LayerNorm', 'GroupNorm', 'InstanceNorm']):
                flops = 4 * num_elements
                
            # 3. 池化层：约 1 FLOPs / element
            elif 'Pool' in class_name:
                flops = num_elements
                
        except Exception:
            pass
            
    return flops


# def _summary_to_dataframe(stats):
#     """
#     将 torchinfo 的 ModelStatistics 转换为扁平的 DataFrame。
#     每一行对应一个被 summary 捕获的层。
#     """
#     rows = []
#     for layer in stats.summary_list:
#         rows.append({
#             "层名称": layer.var_name, # ✅ 改为 var_name
#             "类型": layer.class_name,
#             "输入形状": str(tuple(layer.input_size)) if layer.input_size else "-",   # ✅ 改为 input_size
#             "输出形状": str(tuple(layer.output_size)) if layer.output_size else "-", # ✅ 改为 output_size
#             "参数量": layer.num_params,
#             "Mult-Adds": layer.macs,
#             "FLOPs": _calculate_flops(layer),
#         })
#     return pd.DataFrame(rows)


def _summary_to_dataframe(stats):
    """
    将 torchinfo 的 ModelStatistics 转换为扁平的 DataFrame。
    特别处理 nn.ModuleList 类型，汇总其内部子模块的参数与计算量。
    """
    rows = []
    # 用于记录已经处理过的 ModuleList 变量名，防止重复添加
    processed_modulelists = set()

    for layer in stats.summary_list:
        var_name = getattr(layer, 'var_name', getattr(layer, 'layer_name', 'Unknown'))

        # 检查当前层是否属于一个 ModuleList（通过类名或变量名判断）
        is_modulelist_container = (
            layer.class_name == 'ModuleList' or
            (isinstance(var_name, str) and 'ModuleList' in var_name)
        )
        
        if is_modulelist_container:
            # 如果这个 ModuleList 容器本身还没有被记录过
            if var_name not in processed_modulelists:
                # 初始化汇总值
                total_params = 0
                total_macs = 0
                total_flops = 0
                
                # 遍历所有层，寻找属于这个 ModuleList 的子层并累加
                for sub_layer in stats.summary_list:
                    sub_var_name = getattr(sub_layer, 'var_name', getattr(sub_layer, 'layer_name', ''))
                    # 判断子层是否属于当前 ModuleList (通过变量名前缀匹配)
                    if isinstance(sub_var_name, str) and sub_var_name.startswith(var_name + '.'):
                        total_params += sub_layer.num_params
                        total_macs += sub_layer.macs
                        total_flops += _calculate_flops(sub_layer) # 调用你之前定义的 FLOPs 计算函数
                
                # 只有当汇总到数据时，才添加这一行
                if total_params > 0 or total_macs > 0 or total_flops > 0:
                    rows.append({
                        "层名称": var_name,
                        "类型": "ModuleList (汇总)",
                        "输入形状": "-",
                        "输出形状": "-",
                        "参数量": total_params,
                        "Mult-Adds": total_macs,
                        "FLOPs": total_flops,
                    })
                    processed_modulelists.add(var_name)
            continue # 跳过容器本身，因为它不执行具体计算

        # 对于非 ModuleList 的层，正常处理
        # 这里同时排除了已经被汇总过的 ModuleList 子层，防止数据重复计算
        is_child_of_processed_list = False
        for ml_name in processed_modulelists:
            if isinstance(var_name, str) and var_name.startswith(ml_name + '.'):
                is_child_of_processed_list = True
                break
        
        if not is_child_of_processed_list:
            rows.append({
                "层名称": var_name,
                "类型": layer.class_name,
                "输入形状": str(tuple(layer.input_size)) if layer.input_size else "-",
                "输出形状": str(tuple(layer.output_size)) if layer.output_size else "-",
                "参数量": layer.num_params,
                "Mult-Adds": layer.macs,
                "FLOPs": _calculate_flops(layer)  # 使用你之前定义的 FLOPs 计算函数
            })
            
    return pd.DataFrame(rows)
    

def analyze_model(
    model: nn.Module,
    input_size: tuple,
    save_dir: str | None = None,
    export_onnx: bool = False,
    dtypes: list | None = None,
):
    """
    对模型做逐层结构分析。

    参数
    ----
    model : nn.Module
    input_size : tuple，包含 batch 维，例如 (1, 1, 28, 28)
    save_dir : 若非 None，则在该目录下保存 summary.csv
    export_onnx : 若 True，则尝试导出 model.onnx 供 Netron 使用

    返回
    ----
    df : pd.DataFrame，逐层表格
    total_params : int
    total_mult_adds : int
    onnx_path : str | None
    """
    model.eval()
    # stats = summary(model, input_size=input_size, verbose=0, device="cpu")
    stats = summary(model, input_size=input_size, dtypes=dtypes, verbose=0, device="cpu")
    df = _summary_to_dataframe(stats)

    total_params = int(stats.total_params)
    total_mult_adds = int(stats.total_mult_adds)

    onnx_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, "summary.csv"), index=False, encoding="utf-8-sig")

        if export_onnx:
            onnx_path = os.path.join(save_dir, "model.onnx")
            try:
                dummy = torch.randn(*input_size)
                batch_dim = torch.export.Dim("batch_size")
                first_param = next(iter(inspect.signature(model.forward).parameters))
                torch.onnx.export(
                    model,
                    dummy,
                    onnx_path,
                    export_params=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},

                    # dynamic_shapes={            # ✅ 替代 dynamic_axes
                    #                     "x": {0: batch_dim},    # ✅ 键是 forward() 的参数名 "x"
                    #                },

                    dynamic_shapes={first_param: {0: batch_dim}},
                )
            except Exception as e:
                print(f"ONNX 导出失败（该模型可能包含 ONNX 不支持的自定义算子）：{e}")
                onnx_path = None

    return df, total_params, total_mult_adds, onnx_path


# def profile_model_speed(model, input_size, device="cuda", num_iters=50):
#     """
#     实测模型的真实推理速度和显存占用
#     """
#     if device == "cuda" and not torch.cuda.is_available():
#         device = "cpu"
        
#     model = model.to(device)
#     model.eval()
    
#     # 创建模拟输入 (这里默认是浮点型，如果是 NLP 模型需处理 long 型)
#     dummy_input = torch.randn(*input_size).to(device)
    
#     # 预热 (Warm-up)，排除初次加载的初始化开销
#     with torch.no_grad():
#         for _ in range(10):
#             model(dummy_input)
            
#     # 记录显存峰值（仅限 CUDA）
#     if device == "cuda":
#         torch.cuda.reset_peak_memory_stats(device)
        
#     # 使用 profiler 包装
#     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
#         with torch.no_grad():
#             start_time = time.time()
#             for _ in range(num_iters):
#                 model(dummy_input)
#             if device == "cuda":
#                 torch.cuda.synchronize() # 等待 GPU 真正执行完毕
#             end_time = time.time()
            
#     # 计算指标
#     avg_latency = (end_time - start_time) / num_iters * 1000 # ms
#     peak_vram = 0
#     if device == "cuda":
#         peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # MB
        
#     return avg_latency, peak_vram


def export_tensorboard(model, input_size, dtypes=None):
    """
    将模型结构图导出到 TensorBoard 日志目录。
    """
    model.eval()
    
    # dtype = dtypes[0] if dtypes else torch.float32
    device = next(model.parameters()).device  # 模型实际所在设备
    # dtype = next(model.parameters()).dtype    # 顺便对齐 dtype，避免类型不匹配
    dtype = dtypes[0] if dtypes else torch.float32

    if dtype in [torch.long, torch.int, torch.int64, torch.int32]:
        # dummy_input = torch.randint(0, 1000, input_size, dtype=dtype, device=device)
        # 为 Embedding 层生成合法的词索引（0-1000）
        dummy_input = torch.randint(0, 1000, input_size, dtype=dtype, device=device)
    else:
        dummy_input = torch.randn(*input_size, dtype=dtype, device=device)
        
    # 创建临时目录保存 TensorBoard 日志
    save_dir = tempfile.mkdtemp(prefix="tb_log_")
    writer = SummaryWriter(save_dir)
    
    # tensorboard --logdir=解压后的目录路径 即可查看网络的计算图。
    try:
        # add_graph 底层依赖 JIT trace，对于复杂控制流可能失败
        writer.add_graph(model, dummy_input)
        writer.close()
        return save_dir, None
    except Exception as e:
        writer.close()
        return None, str(e)


def profile_model_speed(model, input_size, dtypes=None, device="cpu", num_iters=50):
    """
    实测模型的真实推理速度和显存占用。
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    model = model.to(device)
    model.eval()
    
    # 1. 智能生成测试输入
    dtype = dtypes[0] if dtypes else torch.float32
    if dtype in [torch.long, torch.int, torch.int64, torch.int32]:
        # 文本/Embedding 模型，生成合法的索引（假设词表 1000）
        dummy_input = torch.randint(0, 1000, input_size, dtype=dtype).to(device)
    else:
        # 视觉/特征模型
        dummy_input = torch.randn(*input_size, dtype=dtype).to(device)
        
    # 2. 预热 (Warm-up)，排除初次运行和显存分配的开销
    with torch.no_grad():
        for _ in range(5):
            model(dummy_input)
            
    # 重置显存统计
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        
    # 3. 循环测试
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_iters):
            model(dummy_input)
        if device == "cuda":
            torch.cuda.synchronize() # 等待 GPU 真正执行完毕
        end_time = time.time()
        
    # 4. 计算指标
    avg_latency = (end_time - start_time) / num_iters * 1000 # 毫秒
    peak_vram = 0
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # MB
        
    return avg_latency, peak_vram