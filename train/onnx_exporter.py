# onnx_exporter.py
import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer


# ============================================================
# Optimum-based 量化函数
# ============================================================

def _quantize_int8_optimum(model_dir, output_dir):
    """
    使用 onnxruntime 进行 INT8 动态量化（绕过 Optimum 的 shape inference 问题）。

    Args:
        model_dir (str): 包含 ONNX 模型的目录
        output_dir (str): 量化后模型保存目录

    Returns:
        bool: 是否成功
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        import onnx
        import shutil

        print("[ONNX] 正在进行 INT8 动态量化...")

        # 查找模型文件（支持 model.onnx 或 model_optimized.onnx）
        model_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model_optimized.onnx")
        if not os.path.exists(model_path):
            print(f"[ONNX] 错误: 找不到模型文件 {model_dir}/model.onnx 或 model_optimized.onnx")
            return False

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "model.onnx")

        # INT8 动态量化 - 使用 extra_options 解决 shape inference 问题
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
            extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT}
        )

        # 复制必要文件到量化目录
        files_to_copy = ["config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "vocab.txt", "label_mapping.json"]
        for f in files_to_copy:
            src = os.path.join(model_dir, f)
            if os.path.exists(src):
                shutil.copy(src, output_dir)

        # 显示模型大小
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[ONNX] 原始模型大小: {original_size:.2f} MB")
        print(f"[ONNX] INT8 量化后大小: {quantized_size:.2f} MB")
        print(f"[ONNX] 压缩比例: {original_size/quantized_size:.2f}x")

        print(f"[ONNX] INT8 量化模型已保存至: {output_dir}")
        return True

    except ImportError as e:
        print(f"[ONNX] 警告: 缺少依赖库: {e}")
        print("[ONNX] 请安装: pip install onnxruntime onnx")
        return False
    except Exception as e:
        print(f"[ONNX] INT8 量化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def _quantize_fp16_optimum(model_dir, output_dir):
    """
    使用 onnxconverter-common 进行 FP16 转换。

    Args:
        model_dir (str): 包含 ONNX 模型的目录
        output_dir (str): 量化后模型保存目录

    Returns:
        bool: 是否成功
    """
    try:
        import onnx
        from onnxconverter_common import float16
        import shutil

        print("[ONNX] 正在进行 FP16 量化...")

        # 查找模型文件（支持 model.onnx 或 model_optimized.onnx）
        model_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model_optimized.onnx")
        if not os.path.exists(model_path):
            print(f"[ONNX] 错误: 找不到模型文件 {model_dir}/model.onnx 或 model_optimized.onnx")
            return False

        # 加载并转换为 FP16
        model = onnx.load(model_path)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "model.onnx")
        onnx.save(model_fp16, output_path)

        # 复制必要文件到量化目录
        files_to_copy = ["config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "vocab.txt", "label_mapping.json"]
        for f in files_to_copy:
            src = os.path.join(model_dir, f)
            if os.path.exists(src):
                shutil.copy(src, output_dir)

        # 显示模型大小
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[ONNX] 原始模型大小: {original_size:.2f} MB")
        print(f"[ONNX] FP16 量化后大小: {quantized_size:.2f} MB")
        print(f"[ONNX] 压缩比例: {original_size/quantized_size:.2f}x")

        print(f"[ONNX] FP16 量化模型已保存至: {output_dir}")
        return True

    except ImportError as e:
        print(f"[ONNX] 警告: 缺少依赖库: {e}")
        print("[ONNX] 请安装: pip install onnx onnxconverter-common")
        return False
    except Exception as e:
        print(f"[ONNX] FP16 量化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# 兼容旧版 onnxruntime 量化函数
# ============================================================

def _quantize_model_int8(model_path, output_path, optimize=True):
    """
    使用 onnxruntime 进行 INT8 动态量化（兼容旧版）。

    Args:
        model_path (str): 原始ONNX模型路径
        output_path (str): 量化后模型保存路径
        optimize (bool): 是否先优化模型再进行量化
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        print("[ONNX] 正在进行INT8动态量化...")

        import inspect
        sig = inspect.signature(quantize_dynamic)

        kwargs = {
            "model_input": model_path,
            "model_output": output_path,
            "weight_type": QuantType.QInt8,
        }

        if "optimize_model" in sig.parameters:
            kwargs["optimize_model"] = optimize
            print(f"[ONNX] 使用 optimize_model={optimize} 参数")

        extra_params = {
            "per_channel": False,
            "reduce_range": False,
            "op_types_to_quantize": None,
            "nodes_to_quantize": None,
            "nodes_to_exclude": None,
            "use_external_data_format": False,
            "extra_options": None,
        }

        for param, default_value in extra_params.items():
            if param in sig.parameters:
                kwargs[param] = default_value

        print(f"[ONNX] 量化参数: {list(kwargs.keys())}")
        quantize_dynamic(**kwargs)

        print(f"[ONNX] INT8量化模型已保存至: {output_path}")

        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[ONNX] 原始模型大小: {original_size:.2f} MB")
        print(f"[ONNX] INT8量化后模型大小: {quantized_size:.2f} MB")
        if quantized_size > 0:
            print(f"[ONNX] 压缩比例: {original_size/quantized_size:.2f}x")

        return True

    except ImportError as e:
        print(f"[ONNX] 警告: 缺少量化依赖库: {e}")
        print("[ONNX] 请安装: pip install onnxruntime onnx")
        return False
    except Exception as e:
        print(f"[ONNX] 量化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def _quantize_model_fp16(model_path, output_path):
    """
    使用 onnxconverter-common 进行 FP16 量化。

    Args:
        model_path (str): 原始ONNX模型路径
        output_path (str): 量化后模型保存路径
    """
    try:
        import onnx
        from onnxconverter_common import float16

        print("[ONNX] 正在进行FP16量化...")

        model = onnx.load(model_path)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, output_path)

        print(f"[ONNX] FP16量化模型已保存至: {output_path}")

        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[ONNX] 原始模型大小: {original_size:.2f} MB")
        print(f"[ONNX] FP16量化后模型大小: {quantized_size:.2f} MB")
        if quantized_size > 0:
            print(f"[ONNX] 压缩比例: {original_size/quantized_size:.2f}x")

        return True

    except ImportError as e:
        print(f"[ONNX] 警告: 缺少量化依赖库: {e}")
        print("[ONNX] 请安装: pip install onnx onnxconverter-common")
        return False
    except Exception as e:
        print(f"[ONNX] FP16量化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def _quantize_model(model_path, output_path, optimize=True):
    """向后兼容的 INT8 量化接口。"""
    return _quantize_model_int8(model_path, output_path, optimize)


# ============================================================
# 交互式选择函数
# ============================================================

def _interactive_optimize_and_quantize(model_dir, base_output_path):
    """
    交互式选择优化和量化选项。

    Args:
        model_dir (str): 模型目录路径
        base_output_path (str): 基础输出路径

    Returns:
        bool: 是否成功
    """
    print("\n" + "=" * 60)
    print("模型优化与量化选项")
    print("=" * 60)

    # 第一步：选择优化等级
    print("\n【步骤 1/2】选择优化等级:")
    print("  [0] 不优化      - 保持原始模型")
    print("  [1] O1 基础优化 - 常量折叠、冗余节点消除")
    print("  [2] O2 中等优化 - O1 + 算子融合")
    print("  [3] O3 激进优化 - O2 + 额外融合 (推荐)")
    print("-" * 60)

    try:
        optimize_choice = input("请选择优化等级 [0/1/2/3，默认3]: ").strip()
        if optimize_choice == "":
            optimize_choice = "3"
    except EOFError:
        print("\n[ONNX] 非交互式环境，使用默认配置。")
        optimize_choice = "3"

    optimize_level = {"0": None, "1": "O1", "2": "O2", "3": "O3"}.get(optimize_choice, "O3")

    # 第二步：选择量化方式
    print("\n【步骤 2/2】选择量化方式:")
    print("  [0] 不量化        - 保持 FP32 精度")
    print("  [1] INT8 动态量化 - 推荐 CPU 推理，精度损失小 (~1-2%)")
    print("  [2] FP16 量化     - 推荐 GPU 推理，精度损失极小")
    print("  [3] 两者都生成    - 同时生成 INT8 和 FP16 版本")
    print("-" * 60)

    try:
        quantize_choice = input("请选择量化方式 [0/1/2/3，默认1]: ").strip()
        if quantize_choice == "":
            quantize_choice = "1"
    except EOFError:
        quantize_choice = "1"

    # 执行优化
    current_model_dir = model_dir
    optimized_dir = None

    if optimize_level:
        print(f"\n[Optimum] 正在进行 {optimize_level} 优化...")
        optimized_dir = _optimize_model_optimum(model_dir, optimize_level)
        if optimized_dir:
            current_model_dir = optimized_dir
            print(f"[Optimum] 优化完成，模型保存在: {optimized_dir}")
        else:
            print("[Optimum] 优化失败，使用原始模型继续。")

    # 执行量化
    success = False
    base_dir = os.path.dirname(base_output_path) if os.path.dirname(base_output_path) else "."

    if quantize_choice == '0':
        print("\n[ONNX] 跳过量化步骤。")
        if optimized_dir:
            print(f"[ONNX] 优化后的 FP32 模型位于: {optimized_dir}")
        success = True

    elif quantize_choice == '1':
        output_dir = os.path.join(base_dir, "model_int8")
        success = _quantize_int8_optimum(current_model_dir, output_dir)

    elif quantize_choice == '2':
        output_dir = os.path.join(base_dir, "model_fp16")
        success = _quantize_fp16_optimum(current_model_dir, output_dir)

    elif quantize_choice == '3':
        # INT8
        int8_dir = os.path.join(base_dir, "model_int8")
        success_int8 = _quantize_int8_optimum(current_model_dir, int8_dir)
        print()

        # FP16
        fp16_dir = os.path.join(base_dir, "model_fp16")
        success_fp16 = _quantize_fp16_optimum(current_model_dir, fp16_dir)

        success = success_int8 or success_fp16

    else:
        print(f"[ONNX] 无效的选择: {quantize_choice}，跳过量化。")

    if success:
        print("\n" + "=" * 60)
        print("[ONNX] 处理完成！")
        print("=" * 60)

    return success


def _optimize_model_optimum(model_dir, optimization_level="O3"):
    """
    使用 Optimum 进行模型优化。

    Args:
        model_dir (str): 模型目录路径
        optimization_level (str): 优化等级 O1/O2/O3

    Returns:
        str: 优化后的模型目录路径，失败返回 None
    """
    try:
        from optimum.onnxruntime import ORTOptimizer
        from optimum.onnxruntime.configuration import OptimizationConfig
        import shutil

        # 映射字符串优化等级到整数（onnxruntime 要求）
        # opt_level 必须是 0, 1, 2, 99 中的一个整数
        level_map = {"O1": 1, "O2": 2, "O3": 99}
        opt_level_int = level_map.get(optimization_level, 99)

        print(f"[Optimum] 正在进行 {optimization_level} 图优化 (opt_level={opt_level_int})...")

        optimizer = ORTOptimizer.from_pretrained(model_dir)

        # 创建优化配置（使用整数 opt_level）
        optimization_config = OptimizationConfig(optimization_level=opt_level_int)

        # 输出目录
        optimized_dir = os.path.join(model_dir, f"optimized_{optimization_level.lower()}")

        # 执行优化 - file_suffix=None 确保输出文件名为 model.onnx
        optimizer.optimize(save_dir=optimized_dir, optimization_config=optimization_config, file_suffix=None)

        # 复制 label_mapping.json 到优化目录
        label_mapping_src = os.path.join(model_dir, "label_mapping.json")
        if os.path.exists(label_mapping_src):
            shutil.copy(label_mapping_src, optimized_dir)
            print(f"[Optimum] 已复制 label_mapping.json 到 {optimized_dir}")

        # 显示模型大小
        original_model = os.path.join(model_dir, "model.onnx")
        optimized_model = os.path.join(optimized_dir, "model.onnx")

        if os.path.exists(original_model) and os.path.exists(optimized_model):
            original_size = os.path.getsize(original_model) / (1024 * 1024)
            optimized_size = os.path.getsize(optimized_model) / (1024 * 1024)
            print(f"[Optimum] 原始模型大小: {original_size:.2f} MB")
            print(f"[Optimum] 优化后模型大小: {optimized_size:.2f} MB")

        print(f"[Optimum] 优化模型已保存至: {optimized_dir}")
        return optimized_dir

    except ImportError as e:
        print(f"[Optimum] 警告: 缺少依赖库: {e}")
        print("[Optimum] 请安装: pip install optimum[onnxruntime]")
        return None
    except Exception as e:
        print(f"[Optimum] 优化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


# 保留旧版交互函数以兼容
def _interactive_quantize(model_path, base_output_path):
    """
    交互式选择量化档位（兼容旧版）。
    """
    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "."
    return _interactive_optimize_and_quantize(model_dir, base_output_path)


# ============================================================
# 模型导出函数
# ============================================================

def convert_to_onnx(model_dir, output_path, max_length=128):
    """
    将保存的 Transformers 模型转换为 ONNX 格式。

    Args:
        model_dir (str): 包含 pytorch/safetensors 模型和配置文件的目录路径
        output_path (str): ONNX 文件的保存路径 (包含文件名, 例如 model.onnx)
        max_length (int): 模型输入的最大序列长度，用于生成 dummy input
    """
    print(f"\n[ONNX] 开始转换模型...")
    print(f"[ONNX] 加载模型自: {model_dir}")

    try:
        # 1. 加载模型和分词器 (使用 CPU 进行导出即可)
        device = torch.device("cpu")
        model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()

        # 2. 准备虚拟输入 (Dummy Input)
        dummy_input_ids = torch.randint(0, 1000, (1, max_length), device=device)
        dummy_attention_mask = torch.ones((1, max_length), device=device)
        dummy_inputs = (dummy_input_ids, dummy_attention_mask)

        # 3. 定义输入输出名称
        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]

        # 4. 定义动态轴 (允许 batch_size 变化)
        dynamic_axes = {
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }

        # 5. 执行导出
        print(f"[ONNX] 正在导出到: {output_path} ...")
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=20,
            do_constant_folding=True
        )

        print(f"[ONNX] 转换成功！模型已保存至: {output_path}")

        # 验证生成的 ONNX 模型
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("[ONNX] 模型结构检查通过。")
        except ImportError:
            print("[ONNX] 未安装 'onnx' 库，跳过结构检查。")
        except Exception as e:
            print(f"[ONNX] 警告: 模型检查失败: {e}")

    except Exception as e:
        print(f"[ONNX] 转换过程中发生错误: {e}")
        raise e

    # 交互式选择优化和量化
    model_dir_path = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    _interactive_optimize_and_quantize(model_dir_path, output_path)


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ONNX 模型导出与量化工具")
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录路径")
    parser.add_argument("--output", type=str, default="model.onnx", help="输出文件名")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--optimize", type=str, choices=["O1", "O2", "O3"], help="优化等级")
    parser.add_argument("--quantize", type=str, choices=["int8", "fp16", "both"], help="量化方式")

    args = parser.parse_args()

    # 非交互模式
    if args.optimize or args.quantize:
        output_path = os.path.join(args.model_dir, args.output)
        convert_to_onnx(args.model_dir, output_path, args.max_length)
    else:
        # 交互模式
        output_path = os.path.join(args.model_dir, args.output)
        convert_to_onnx(args.model_dir, output_path, args.max_length)