import os
import json
from datetime import datetime

# =========================
# 0) 环境变量（保持不变）
# =========================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from llamafactory.train.tuner import run_exp
import shutil
import glob

# =========================
# 1) 定义训练参数（你的原始配置）
# =========================
args = {
    # --- 模型路径 ---
    "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    # "use_unsloth": True,

    # --- 4bit 量化 ---
    "quantization_bit": 4,
    "quantization_method": "bitsandbytes",

    # --- Attention ---
    "flash_attn": "auto",
    "packing": True,

    # --- 阶段设置 ---
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "template": "qwen",

    # --- 数据设置 ---
    "dataset": "finance_alpaca",
    "dataset_dir": "./data",
    "cutoff_len": 2048,

    # --- 训练超参数 ---
    "output_dir": "saves/qwen_2.5_7b_finance_v1",
    "overwrite_output_dir": True,

    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2.0e-4,
    "num_train_epochs": 3.0,

    # --- 监控与硬件 ---
    "logging_steps": 100,
    "save_steps": 200,
    "plot_loss": True,

    "bf16": True,
    "fp16": False,

    "gradient_checkpointing": False,
    "preprocessing_num_workers": 16,
}

# =========================
# ✅ 精简：用已注册的 val.json 做验证 + 训练结束回滚到最优 checkpoint
# =========================
def _choose_eval_steps(save_steps: int) -> int:
    """
    load_best_model_at_end + steps 模式要求：
    - eval_strategy == save_strategy
    - save_steps 必须是 eval_steps 的整数倍
    所以这里返回一个能整除 save_steps 的 eval_steps。
    """
    preferred = [1000, 500, 200, 100, 50, 20, 10, 5, 1]
    for s in preferred:
        if save_steps % s == 0:
            return s
    return save_steps  # 兜底：至少满足整除

def enable_val_and_best_model(args: dict, eval_name: str = "val"):
    save_steps = int(args.get("save_steps", 500))

    # ---- (A) eval / save 策略对齐 ----
    args.setdefault("save_strategy", "steps")
    args.setdefault("eval_strategy", "steps")
    args.setdefault("eval_steps", _choose_eval_steps(save_steps))
    args.setdefault("per_device_eval_batch_size", 1)

    # 双保险：确保整除
    if save_steps % int(args["eval_steps"]) != 0:
        args["eval_steps"] = save_steps

    # ---- (B) 明确使用已注册的 val 数据集（你的 dataset_info.json 里 key 就叫 "val"） ----
    args["eval_dataset"] = eval_name
    args["val_size"] = 0.0  # 禁止从训练集再切分出“伪验证集”

    # ---- (C) 训练结束自动加载验证集最优 checkpoint ----
    args["load_best_model_at_end"] = True
    args.setdefault("metric_for_best_model", "eval_loss")
    args.setdefault("greater_is_better", False)

enable_val_and_best_model(args, eval_name="val")


# =========================
# 2) 改动①：输出目录唯一化（避免覆盖）
# =========================
tag = datetime.now().strftime("%m%d_%H%M")
epochs_str = str(args["num_train_epochs"]).replace(".", "p")
lr_str = f"{args['learning_rate']:.0e}".replace("-", "")  # 2e-4 -> 2e4(简化显示)
cut_str = str(args["cutoff_len"])

args["output_dir"] = (
    f"saves/qwen2.5-7b_finance_{tag}_e{epochs_str}_lr{lr_str}_cut{cut_str}_loraall"
)

# 强制不覆盖（避免误删）
args["overwrite_output_dir"] = False

# =========================
# 3) 改动②：保存本次实验配置到输出目录
# =========================
os.makedirs(args["output_dir"], exist_ok=True)
with open(os.path.join(args["output_dir"], "run_args.json"), "w", encoding="utf-8") as f:
    json.dump(args, f, ensure_ascii=False, indent=2)




def archive_artifacts(output_dir: str):
    """
    把可能生成在当前目录的 loss 图、日志等，统一搬到 output_dir 里。
    """
    patterns = [
        "*loss*.png",
        "*loss*.jpg",
        "*loss*.jpeg",
        "*loss*.pdf",
        "*loss*.json",
        "*trainer_state*.json",
        "*all_results*.json",
        "*log_history*.json",
        "events.out.tfevents.*",  # tensorboard
    ]

    moved = []
    for pat in patterns:
        for p in glob.glob(pat):
            # 避免把 output_dir 里面的文件又搬一次
            if os.path.abspath(p).startswith(os.path.abspath(output_dir)):
                continue
            dst = os.path.join(output_dir, os.path.basename(p))
            try:
                shutil.move(p, dst)
                moved.append(os.path.basename(p))
            except Exception:
                # move 失败就 copy（例如跨盘/权限）
                try:
                    shutil.copy2(p, dst)
                    moved.append(os.path.basename(p))
                except Exception:
                    pass

    if moved:
        print(f"✅ 已归档到 {output_dir}: {moved}")
    else:
        print("ℹ️ 未在当前目录发现需要归档的 loss/日志文件（可能已直接生成在 output_dir 内）。")

# =========================
# 4) 开始运行
# =========================
if __name__ == "__main__":
    print("开始加载 LLaMA-Factory (Int4 QLoRA 加速模式)...")
    print(f"本次输出目录: {args['output_dir']}")

    # 保存本次配置
    os.makedirs(args["output_dir"], exist_ok=True)
    with open(os.path.join(args["output_dir"], "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(args, f, ensure_ascii=False, indent=2)

    try:
        run_exp(args)
        # 训练结束后归档
        archive_artifacts(args["output_dir"])
        print(f"训练结束！模型已保存至: {args['output_dir']}")
    except Exception as e:
        # 即使出错也尝试归档，方便你排查
        archive_artifacts(args["output_dir"])
        print(f"训练发生错误: {e}")