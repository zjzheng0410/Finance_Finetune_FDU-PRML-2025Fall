import os
from datasets import load_dataset, DatasetDict

# =========================
# 配置
# =========================
SEED = 42
VAL_TEST_RATIO = 0.10  # 先切出10%再平分 -> val/test各5%

# 当前脚本所在目录（确保“同目录”读写）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 优先尝试常见的清洗后文件名（兼容笔误/不同命名）
possible_names = [
    "Cleaned_data.json",
    "Cleaned_date.json",  # 有时写成 date
    "Cleaned.json",
    "cleaned_data.json",
]
INPUT_FILE = None
for name in possible_names:
    p = os.path.join(BASE_DIR, name)
    if os.path.exists(p):
        INPUT_FILE = p
        break
if INPUT_FILE is None:
    # 如果没有找到，保留原来的期望路径以便错误信息清晰
    INPUT_FILE = os.path.join(BASE_DIR, "Cleaned_data.json")

# 输出文件名（按你要求）
TRAIN_OUT = os.path.join(BASE_DIR, "finance_alpaca.json")
VAL_OUT   = os.path.join(BASE_DIR, "val.json")
TEST_OUT  = os.path.join(BASE_DIR, "test.json")

# =========================
# 读取本地数据
# =========================
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"找不到输入文件: {INPUT_FILE}")

# 支持 json / jsonl（这里扩展名不重要，datasets 会按 json 读）
ds = load_dataset("json", data_files={"train": INPUT_FILE})

# =========================
# 切分 train/val/test
# =========================
tmp = ds["train"].train_test_split(test_size=VAL_TEST_RATIO, seed=SEED)  # 90/10
tmp2 = tmp["test"].train_test_split(test_size=0.50, seed=SEED)          # 10 -> 5/5

splits = DatasetDict({
    "train": tmp["train"],
    "val":   tmp2["train"],
    "test":  tmp2["test"],
})

# =========================
# 保存到同目录（覆盖写）
# =========================
splits["train"].to_json(TRAIN_OUT, force_ascii=False)
splits["val"].to_json(VAL_OUT, force_ascii=False)
splits["test"].to_json(TEST_OUT, force_ascii=False)

print("✅ 切分完成：")
print(f"  train -> {TRAIN_OUT} (n={len(splits['train'])})")
print(f"  val   -> {VAL_OUT}   (n={len(splits['val'])})")
print(f"  test  -> {TEST_OUT}  (n={len(splits['test'])})")
print(f"  seed={SEED}, val/test 各占 {VAL_TEST_RATIO/2:.0%}")
