# app.py
import ast
import os
import tempfile
import importlib.util
import inspect
import shutil
import sys
import tempfile

import gradio as gr
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis import analyze_model, profile_model_speed, export_tensorboard
from models import (
    MLP, CNN1D, CNN2D, SimpleResNet2D,
    CNN2D_ChannelAttention, CNN2D_SpatialAttention,
    TinyUNet, CNN3D, TinyConvLSTM,
    SimpleRNN, SimpleLSTM, BiRNN, BiLSTM, LSTMAttention, TinyTransformer,
    VGG11Small, ResNet18Small, ViTTiny,
    YOLODetectionHead, DeepLabV3Small, MaskRCNNLight,
    ImageCaptioning, DCGANGenerator,
    BERTClassifier, GPT2Small, Seq2SeqTranslator,
    TextCNN, BiGRUSequenceLabeler,
)

TEXT_MODELS = {"BERT 文本分类", "GPT-2 文本生成", "Seq2Seq 机器翻译", "TextCNN 文本分类"}

MODEL_REGISTRY = {
    # 原有
    "MLP (BP)": (MLP, (1, 784)),
    "1D-CNN": (CNN1D, (1, 1, 100)),
    "2D-CNN": (CNN2D, (1, 1, 28, 28)),
    "2D-ResNet": (SimpleResNet2D, (1, 1, 28, 28)),
    "2D + 通道注意力": (CNN2D_ChannelAttention, (1, 1, 28, 28)),
    "2D + 空间注意力": (CNN2D_SpatialAttention, (1, 1, 28, 28)),
    "UNet": (TinyUNet, (1, 1, 32, 32)),
    "3D-CNN": (CNN3D, (1, 1, 8, 16, 16)),
    "ConvLSTM": (TinyConvLSTM, (1, 4, 1, 16, 16)),
    "RNN": (SimpleRNN, (1, 10, 16)),
    "LSTM": (SimpleLSTM, (1, 10, 16)),
    "双向 RNN": (BiRNN, (1, 10, 16)),
    "双向 LSTM": (BiLSTM, (1, 10, 16)),
    "LSTM + Attention": (LSTMAttention, (1, 10, 16)),
    "Transformer": (TinyTransformer, (1, 10, 16)),

    # 新增 CV
    "VGG11 (Small)": (VGG11Small, (1, 3, 64, 64)),
    "ResNet18 (Small)": (ResNet18Small, (1, 3, 64, 64)),
    "ViT (Tiny)": (ViTTiny, (1, 3, 32, 32)),
    "YOLO 检测头": (YOLODetectionHead, (1, 3, 128, 128)),
    "DeepLabV3 (Small)": (DeepLabV3Small, (1, 3, 128, 128)),
    "Mask R-CNN (Light)": (MaskRCNNLight, (1, 3, 128, 128)),
    "图像描述 (CNN+LSTM)": (ImageCaptioning, (1, 3, 64, 64)),
    "DCGAN 生成器": (DCGANGenerator, (1, 100)),

    # 新增 NLP
    "BERT 文本分类": (BERTClassifier, (1, 32)),
    "GPT-2 文本生成": (GPT2Small, (1, 32)),
    "Seq2Seq 机器翻译": (Seq2SeqTranslator, (1, 20)),
    "TextCNN 文本分类": (TextCNN, (1, 32)),
    "BiGRU 序列标注": (BiGRUSequenceLabeler, (1, 20, 16)),
}


# ---------- 单模型分析（保持不变） ----------

def _parse_shape(shape_str, default):
    if not shape_str or not shape_str.strip():
        return default
    try:
        parsed = ast.literal_eval(shape_str.strip())
    except Exception as e:
        raise gr.Error(f"无法解析输入形状 '{shape_str}'：{e}")
    if not isinstance(parsed, (list, tuple)) or len(parsed) == 0 or not all(isinstance(x, int) and x > 0 for x in parsed):
        raise gr.Error(f"输入形状必须是由正整数组成的元组，例如 (1, 3, 64, 64)，收到：{shape_str}")
    return tuple(parsed)


def get_default_shape(model_name):
    return str(MODEL_REGISTRY[model_name][1])


def _get_output_files(csv_path, onnx_path=None, tb_dir=None):
    """打包输出文件供 Gradio 下载"""
    files = [csv_path]
    if onnx_path and os.path.exists(onnx_path):
        files.append(onnx_path)
    if tb_dir and os.path.exists(tb_dir):
        # 将 TensorBoard 的日志目录打包成 zip
        zip_path = shutil.make_archive(tb_dir, 'zip', tb_dir)
        files.append(zip_path)
    return files


# def create_layer_visualizations(df):
#     """
#     根据 torchinfo 的 DataFrame 生成两个可视化图表：
#     1. 按网络深度的双 Y 轴图 (层序 vs 计算量/参数量)
#     2. 参数量 vs Mult-Adds 的散点权衡图 (颜色代表层序)
#     """
#     if df is None or len(df) == 0:
#         return None, None
        
#     # 过滤掉既没有参数也没有计算量的空层（例如单纯的 Input 或 ReLU），使图表更紧凑
#     df_plot = df[(df["参数量"] > 0) | (df["Mult-Adds"] > 0)].reset_index(drop=True)
    
#     if len(df_plot) == 0:
#         return None, None
        
#     df_plot["层序"] = df_plot.index + 1
#     df_plot["显示名称"] = df_plot["层序"].astype(str) + ". " + df_plot["类型"]
    
#     # 对数坐标系中不能出现 0，因此做 clip 保护
#     df_plot["参数量_显示"] = df_plot["参数量"].clip(lower=1)
#     df_plot["Mult_Adds_显示"] = df_plot["Mult-Adds"].clip(lower=1)

#     # ================= 图 1：按网络深度的双 Y 轴图 =================
#     fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
#     # 参数量（柱状图）
#     fig1.add_trace(
#         go.Bar(x=df_plot["显示名称"], y=df_plot["参数量_显示"], name="参数量 (Params)", marker_color="rgba(55, 128, 191, 0.6)"),
#         secondary_y=False,
#     )
    
#     # Mult-Adds（折线图）
#     fig1.add_trace(
#         go.Scatter(x=df_plot["显示名称"], y=df_plot["Mult_Adds_显示"], name="计算量 (Mult-Adds)", mode="lines+markers", line=dict(color="crimson", width=2)),
#         secondary_y=True,
#     )
    
#     fig1.update_layout(
#         title="逐层参数量与计算量（按网络深度排列）",
#         xaxis_title="网络层 (由浅入深)",
#         xaxis_tickangle=-45,
#         legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
#         margin=dict(b=150),
#         height=450
#     )
#     fig1.update_yaxes(title_text="参数量 (log)", secondary_y=False, type="log")
#     fig1.update_yaxes(title_text="Mult-Adds (log)", secondary_y=True, type="log")

#     # ================= 图 2：参数量 vs Mult-Adds 权衡图 =================
#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(
#         x=df_plot["参数量_显示"], 
#         y=df_plot["Mult_Adds_显示"], 
#         mode="markers+text",
#         text=df_plot["层序"],
#         textposition="top center",
#         marker=dict(
#             size=14,
#             color=df_plot["层序"], # 颜色映射层序，由浅入深
#             colorscale="Viridis",
#             showscale=True,
#             colorbar=dict(title="网络层序")
#         ),
#         hovertext=df_plot["显示名称"],
#         hoverinfo="text+x+y",
#         name="网络层"
#     ))
    
#     fig2.update_layout(
#         title="参数量 vs Mult-Adds 分布（颜色代表输入到输出的先后）",
#         xaxis_title="参数量 (log scale)",
#         yaxis_title="Mult-Adds (log scale)",
#         xaxis_type="log",
#         yaxis_type="log",
#         height=450,
#         margin=dict(t=50, b=50)
#     )
    
#     return fig1, fig2
    

def create_layer_visualizations(df):
    if df is None or len(df) == 0:
        return None, None
        
    # 过滤掉既没有参数也没有计算量的空层（使图表更紧凑）
    df_plot = df[(df["参数量"] > 0) | (df["Mult-Adds"] > 0) | (df["FLOPs"] > 0)].reset_index(drop=True)
    if len(df_plot) == 0:
        return None, None
        
    df_plot["层序"] = df_plot.index + 1
    df_plot["显示名称"] = df_plot["层序"].astype(str) + ". " + df_plot["类型"]
    
    # 对数坐标保护
    df_plot["参数量_显示"] = df_plot["参数量"].clip(lower=1)
    df_plot["FLOPs_显示"] = df_plot["FLOPs"].clip(lower=1)

    # ================= 图 1：按网络深度的双 Y 轴图 =================
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 参数量（柱状图）
    fig1.add_trace(
        go.Bar(x=df_plot["显示名称"], y=df_plot["参数量_显示"], name="参数量 (Params)", marker_color="rgba(55, 128, 191, 0.6)"),
        secondary_y=False,
    )
    
    # FLOPs（折线图，取代原来的 Mult-Adds，使得激活层不再断线）
    fig1.add_trace(
        go.Scatter(x=df_plot["显示名称"], y=df_plot["FLOPs_显示"], name="计算量 (FLOPs)", mode="lines+markers", line=dict(color="crimson", width=2)),
        secondary_y=True,
    )
    
    fig1.update_layout(
        title="逐层参数量与计算量 FLOPs（按网络深度排列）",
        xaxis_title="网络层 (由浅入深)",
        xaxis_tickangle=-45,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(b=150),
        height=450
    )
    fig1.update_yaxes(title_text="参数量 (log)", secondary_y=False, type="log")
    fig1.update_yaxes(title_text="FLOPs (log)", secondary_y=True, type="log")

    # ================= 图 2：参数量 vs FLOPs 权衡图 =================
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_plot["参数量_显示"], 
        y=df_plot["FLOPs_显示"], 
        mode="markers+text",
        text=df_plot["层序"],
        textposition="top center",
        marker=dict(
            size=14,
            color=df_plot["层序"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="网络层序")
        ),
        hovertext=df_plot["显示名称"],
        hoverinfo="text+x+y",
        name="网络层"
    ))
    
    fig2.update_layout(
        title="参数量 vs FLOPs 分布（颜色代表输入到输出的先后）",
        xaxis_title="参数量 (log scale)",
        yaxis_title="FLOPs (log scale)",
        xaxis_type="log",
        yaxis_type="log",
        height=450,
        margin=dict(t=50, b=50)
    )
    
    return fig1, fig2


def run_single(model_name, shape_str, export_onnx, export_tb, num_iters):
    model_cls, default_shape = MODEL_REGISTRY[model_name]
    input_size = _parse_shape(shape_str, default_shape)
    model = model_cls()
    
    save_dir = tempfile.mkdtemp(prefix="torchinfo_")
    dtypes = [torch.long] if model_name in TEXT_MODELS else None

    try:
        df, total_params, total_mult_adds, onnx_path = analyze_model(
            model, input_size, save_dir=save_dir, export_onnx=export_onnx, dtypes=dtypes
        )
    except Exception as e:
        raise gr.Error(f"结构分析失败：{e}")

    # 性能测试
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perf_text = ""
    try:
        avg_latency, peak_vram = profile_model_speed(
            model, input_size, dtypes=dtypes, device=device, num_iters=num_iters
        )
        perf_text = f"\n--- 真实性能测试 ---\n运行设备：{device.upper()}\n循环次数：{num_iters}\n平均延迟：{avg_latency:.2f} ms\n"
        if device == "cuda":
            perf_text += f"峰值显存：{peak_vram:.2f} MB"
    except Exception as e:
        perf_text = f"\n性能测试失败：{e}"

    # TensorBoard 导出
    tb_dir = None
    tb_text = ""
    if export_tb:
        tb_dir, tb_err = export_tensorboard(model, input_size, dtypes=dtypes)
        if tb_err:
            tb_text = f"\nTensorBoard 导出失败（通常因为模型包含动态控制流）：{tb_err}"
        else:
            tb_text = "\nTensorBoard 图导出成功！可通过下载的 zip 或命令行 tensorboard 查看。"

    summary_text = (
        f"模型：{model_name}\n"
        f"输入形状：{input_size}\n"
        f"总参数量：{total_params:,}\n"
        f"总 Mult-Adds：{total_mult_adds:,}\n"
        f"CSV 保存路径：{os.path.join(save_dir, 'summary.csv')}"
    )
    if onnx_path:
        summary_text += f"\nONNX 保存路径：{onnx_path}"
    summary_text += perf_text + tb_text

    files = _get_output_files(os.path.join(save_dir, "summary.csv"), onnx_path, tb_dir)
    fig1, fig2 = create_layer_visualizations(df)  
    return df, summary_text, files, fig1, fig2   


def run_custom(file_obj, class_name, shape_str, is_text_model, export_onnx, export_tb, num_iters):
    if file_obj is None:
        raise gr.Error("请先上传一个包含 PyTorch 模型的 .py 文件！")
    
    input_size = _parse_shape(shape_str, None)
    if input_size is None:
        raise gr.Error("请输入有效的输入形状！")

    file_path = file_obj.name
    module_name = "user_custom_model_module"
    
    # 1. 动态加载用户模块
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ValueError("无法加载文件，请确保是有效的 Python 文件。")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise gr.Error(f"导入 Python 文件失败：{e}")

    # 2. 寻找目标模型类
    # target_cls = None
    # if class_name.strip():
    #     if hasattr(module, class_name.strip()):
    #         target_cls = getattr(module, class_name.strip())
    #     else:
    #         raise gr.Error(f"在文件中找不到名为 '{class_name}' 的类。")
    # else:
    #     # 自动扫描所有继承自 nn.Module 的类，排除 nn.Module 本身
    #     for name, obj in inspect.getmembers(module, inspect.isclass):
    #         if issubclass(obj, nn.Module) and obj is not nn.Module:
    #             target_cls = obj
    #             break
                
    # if target_cls is None:
    #     raise gr.Error("在文件中未找到继承自 nn.Module 的模型类。")


    target_cls = None
    if class_name.strip():
        if hasattr(module, class_name.strip()):
            target_cls = getattr(module, class_name.strip())
        else:
            raise gr.Error(f"在文件中找不到名为 '{class_name}' 的类。")
    else:
        # ✅ 优化：自动扫描时，尝试无参实例化，跳过需要传参的辅助类
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, nn.Module) and obj is not nn.Module:
                try:
                    obj() # 尝试无参实例化
                    target_cls = obj
                    break # 找到能成功实例化的类就退出
                except TypeError:
                    continue # 如果报错（缺少参数），说明是内部辅助类，跳过
                
    if target_cls is None:
        raise gr.Error("在文件中未找到可以直接无参实例化的 nn.Module 模型类，请尝试手动指定类名。")


    # 3. 实例化并分析
    try:
        model = target_cls()
    except Exception as e:
        raise gr.Error(f"实例化模型 {target_cls.__name__} 失败：{e}")

    save_dir = tempfile.mkdtemp(prefix="torchinfo_custom_")
    dtypes = [torch.long] if is_text_model else None

    try:
        df, total_params, total_mult_adds, onnx_path = analyze_model(
            model, input_size, save_dir=save_dir, export_onnx=export_onnx, dtypes=dtypes
        )
    except Exception as e:
        raise gr.Error(f"结构分析失败：{e}")

    # 性能测试
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perf_text = ""
    try:
        avg_latency, peak_vram = profile_model_speed(
            model, input_size, dtypes=dtypes, device=device, num_iters=num_iters
        )
        perf_text = f"\n--- 真实性能测试 ---\n运行设备：{device.upper()}\n平均延迟：{avg_latency:.2f} ms\n"
        if device == "cuda":
            perf_text += f"峰值显存：{peak_vram:.2f} MB"
    except Exception as e:
        perf_text = f"\n性能测试失败：{e}"

    # TensorBoard 导出
    tb_dir = None
    tb_text = ""
    if export_tb:
        tb_dir, tb_err = export_tensorboard(model, input_size, dtypes=dtypes)
        if tb_err:
            tb_text = f"\nTensorBoard 导出失败：{tb_err}"
        else:
            tb_text = "\nTensorBoard 图导出成功！"

    summary_text = (
        f"自定义模型：{target_cls.__name__}\n"
        f"输入形状：{input_size}\n"
        f"总参数量：{total_params:,}\n"
        f"总 Mult-Adds：{total_mult_adds:,}\n"
    )
    summary_text += perf_text + tb_text

    files = _get_output_files(os.path.join(save_dir, "summary.csv"), onnx_path, tb_dir)
    fig1, fig2 = create_layer_visualizations(df) 
    return df, summary_text, files, fig1, fig2   


# ---------- 多模型对比 ----------

def run_compare(model_names, progress=gr.Progress()):
    if not model_names:
        raise gr.Error("请至少选择一个模型")

    rows = []
    for i, name in enumerate(model_names):
        progress((i, len(model_names)), desc=f"分析 {name}")
        cls, shape = MODEL_REGISTRY[name]
        try:
            model = cls()
            # 拿到包含逐层 FLOPs 的 DataFrame，以及总参数量(p)和总 Mult-Adds(m)
            df, p, m, _ = analyze_model(model, shape, save_dir=None, export_onnx=False)
            
            # 计算总 FLOPs
            total_flops = int(df["FLOPs"].sum()) if not df.empty else 0
            
            rows.append({
                "模型": name, 
                "输入形状": str(shape),
                "参数量": int(p), 
                "Mult-Adds": int(m),
                "FLOPs": total_flops
            })
        except Exception as e:
            rows.append({
                "模型": name, 
                "输入形状": str(shape),
                "参数量": 0, 
                "Mult-Adds": 0, 
                "FLOPs": 0,
                "错误": str(e)[:80]
            })

    df_res = pd.DataFrame(rows)
    df_plot = df_res.sort_values("FLOPs", ascending=False).copy()
    
    # 对数坐标保护
    df_plot["参数量(显示)"] = df_plot["参数量"].clip(lower=1)
    df_plot["Mult-Adds(显示)"] = df_plot["Mult-Adds"].clip(lower=1)
    df_plot["FLOPs(显示)"] = df_plot["FLOPs"].clip(lower=1)

    # ================= 图 1：多指标柱状图 =================
    long_df = df_plot.melt(
        id_vars="模型",
        value_vars=["参数量(显示)", "Mult-Adds(显示)", "FLOPs(显示)"],
        var_name="指标", value_name="数值",
    )
    bar = px.bar(
        long_df, x="模型", y="数值", color="指标", barmode="group",
        title="各模型参数量 vs Mult-Adds vs FLOPs（对数 Y 轴）",
        color_discrete_sequence=["#3780bf", "#d62728", "#2ca02c"] # 蓝、红、绿对应不同指标
    )
    bar.update_yaxes(type="log")
    bar.update_layout(
        xaxis_tickangle=-35, height=480, 
        margin=dict(b=120),
        legend_title_text="指标",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # ================= 图 2：参数量 vs FLOPs 散点图 =================
    scatter = px.scatter(
        df_plot, x="参数量(显示)", y="FLOPs(显示)", text="模型",
        size="Mult-Adds(显示)", size_max=60, # 气泡大小依然代表传统 MACs，颜色可以区分模型
        title="参数量 vs FLOPs 散点图（气泡大小为 Mult-Adds）",
        color="模型"
    )
    scatter.update_traces(textposition="top center", textfont_size=10)
    scatter.update_xaxes(type="log", title_text="参数量（log）")
    scatter.update_yaxes(type="log", title_text="FLOPs（log）")
    scatter.update_layout(
        height=480, 
        showlegend=False,
        margin=dict(t=50, b=50)
    )

    # 保存对比汇总 CSV
    save_dir = tempfile.mkdtemp(prefix="torchinfo_compare_")
    csv_path = os.path.join(save_dir, "compare.csv")
    df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    return df_res, bar, scatter, csv_path


def run_compare_all(progress=gr.Progress()):
    return run_compare(list(MODEL_REGISTRY.keys()), progress=progress)


def select_all():
    return list(MODEL_REGISTRY.keys())


def invert_selection(current):
    """反选：全集减去当前已选。"""
    return [m for m in MODEL_REGISTRY if m not in current]


# =======================
# Gradio UI 界面
# =======================

with gr.Blocks(title="PyTorch 模型结构分析器") as demo:
    gr.Markdown("## PyTorch 逐层结构 · 计算量分析器")
    
    with gr.Tabs():
        # Tab 1：单模型详细分析（增加了 TensorBoard 勾选框）
        with gr.Tab("单模型详细分析"):
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=list(MODEL_REGISTRY.keys()), value="2D-CNN", label="选择模型")
                shape_input = gr.Textbox(value=str(MODEL_REGISTRY["2D-CNN"][1]), label="输入形状")
                num_iters_slider = gr.Slider(minimum=10, maximum=500, value=50, step=10, label="性能测试循环次数")
            
            with gr.Row():
                onnx_checkbox = gr.Checkbox(value=False, label="导出 ONNX")
                tb_checkbox = gr.Checkbox(value=False, label="导出 TensorBoard 图")
                run_btn = gr.Button("运行分析", variant="primary")

            df_output = gr.Dataframe(label="逐层结构表", headers=["层名称", "类型", "输入形状", "输出形状", "参数量", "Mult-Adds", "FLOPs"], wrap=True)
            with gr.Row():
                layer_bar_plot = gr.Plot(label="逐层计算量与参数量（按网络深度）")
                layer_scatter_plot = gr.Plot(label="参数量与计算量权衡图")
            summary_output = gr.Textbox(label="摘要", lines=8)
            file_output = gr.File(label="下载产出文件", file_count="multiple")

            model_dropdown.change(fn=lambda x: str(MODEL_REGISTRY[x][1]), inputs=model_dropdown, outputs=shape_input)
            run_btn.click(
                            fn=run_single, 
                            inputs=[model_dropdown, shape_input, onnx_checkbox, tb_checkbox, num_iters_slider], 
                            outputs=[df_output, summary_output, file_output, layer_bar_plot, layer_scatter_plot])

        # Tab 2：自定义模型分析（全新功能）
        with gr.Tab("自定义模型上传分析"):
            gr.Markdown("上传一个 `.py` 文件（其中必须包含继承自 `nn.Module` 的类）。系统会自动寻找第一个找到的类，或通过类名指定。")
            with gr.Row():
                file_input = gr.File(label="上传 .py 文件", file_types=[".py"])
                class_name_input = gr.Textbox(label="类名 (可选，留空自动寻找)", placeholder="例如 MyModel")
                custom_shape_input = gr.Textbox(value="(1, 3, 32, 32)", label="输入形状")
                
            with gr.Row():
                is_text_checkbox = gr.Checkbox(value=False, label="NLP/文本模型 (整型输入)")
                custom_onnx_checkbox = gr.Checkbox(value=False, label="导出 ONNX")
                custom_tb_checkbox = gr.Checkbox(value=False, label="导出 TensorBoard 图")
                custom_num_iters = gr.Slider(minimum=10, maximum=500, value=50, step=10, label="性能循环次数")
                custom_run_btn = gr.Button("分析自定义模型", variant="primary")

            custom_df_output = gr.Dataframe(label="逐层结构表", headers=["层名称", "类型", "输入形状", "输出形状", "参数量", "Mult-Adds", "FLOPs"], wrap=True)
            with gr.Row():
                custom_layer_bar_plot = gr.Plot(label="逐层计算量与参数量（按网络深度）")
                custom_layer_scatter_plot = gr.Plot(label="参数量与计算量权衡图")
            custom_summary_output = gr.Textbox(label="摘要", lines=8)
            custom_file_output = gr.File(label="下载产出文件", file_count="multiple")

            custom_run_btn.click(
                fn=run_custom, 
                inputs=[file_input, class_name_input, custom_shape_input, is_text_checkbox, custom_onnx_checkbox, custom_tb_checkbox, custom_num_iters], 
                outputs=[custom_df_output, custom_summary_output, custom_file_output, custom_layer_bar_plot, custom_layer_scatter_plot]
            )

        # Tab 3：多模型对比 (保持不变)
        with gr.Tab("多模型对比"):
            gr.Markdown(
                "勾选模型后点击「开始对比」。新增**反选**按钮：一键把当前选中集合取反。"
            )
            compare_models = gr.CheckboxGroup(
                choices=list(MODEL_REGISTRY.keys()),
                value=list(MODEL_REGISTRY.keys()),
                label="选择要对比的模型",
            )
            with gr.Row():
                select_all_btn = gr.Button("全选")
                invert_btn = gr.Button("反选")
                compare_btn = gr.Button("开始对比", variant="primary")
                compare_all_btn = gr.Button("全部模型对比", variant="secondary")

            compare_df = gr.Dataframe(
                label="对比汇总表",
                headers=["模型", "输入形状", "参数量", "Mult-Adds", "FLOPs"],
            )
            with gr.Row():
                bar_plot = gr.Plot(label="参数量 vs Mult-Adds（柱状图）")
                scatter_plot = gr.Plot(label="参数量 vs Mult-Adds（散点图）")
            compare_file = gr.File(label="下载对比汇总 CSV")

            select_all_btn.click(fn=select_all, outputs=compare_models)
            invert_btn.click(fn=invert_selection, inputs=compare_models, outputs=compare_models)
            compare_btn.click(fn=run_compare, inputs=compare_models,
                              outputs=[compare_df, bar_plot, scatter_plot, compare_file])
            compare_all_btn.click(fn=run_compare_all,
                                  outputs=[compare_df, bar_plot, scatter_plot, compare_file])


if __name__ == "__main__":
    demo.launch()