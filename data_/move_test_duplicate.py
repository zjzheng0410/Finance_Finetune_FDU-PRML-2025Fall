# remove_test_from_train.py
# 功能：从 final_dataset_clean.json 中剔除 Finance_test.json 里的所有样本（防止测试集泄漏进训练集）
# 用法：把本脚本放在两个json同目录下，然后直接运行：
#   python remove_test_from_train.py

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union
from tqdm import tqdm

def _first_non_ws_char(fp: Path, peek_bytes: int = 4096) -> str:
    with fp.open("rb") as f:
        chunk = f.read(peek_bytes)
    try:
        s = chunk.decode("utf-8", errors="ignore")
    except Exception:
        s = str(chunk)
    for ch in s:
        if not ch.isspace():
            return ch
    return ""

def load_json_or_jsonl(fp: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """
    返回 (fmt, records)
    fmt in {"json_array", "jsonl"}
    """
    ch = _first_non_ws_char(fp)
    records: List[Dict[str, Any]] = []

    if ch == "[":
        # JSON array
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{fp} looks like JSON but is not a list.")
        # 只保留 dict 项
        for item in data:
            if isinstance(item, dict):
                records.append(item)
        return "json_array", records

    # JSONL（逐行读取）
    bad_lines = 0
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                bad_lines += 1

    if bad_lines > 0:
        print(f"[Warn] {fp.name}: skipped {bad_lines} non-JSON lines (kept {len(records)} records).")

    return "jsonl", records

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return " ".join(x.strip().split())

def fingerprint(rec: Dict[str, Any]) -> str:
    """
    用 instruction/input/output 生成稳定指纹。
    注意：不把 text 字段纳入，因为你很多数据 text 为空；而 instruction/input/output 才是训练语义主体。
    """
    obj = {
        "instruction": normalize_text(rec.get("instruction", "")),
        "input": normalize_text(rec.get("input", "")),
        "output": normalize_text(rec.get("output", "")),
    }
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def save_like_input(fmt: str, fp_out: Path, records: List[Dict[str, Any]]) -> None:
    fp_out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json_array":
        with fp_out.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    else:
        with fp_out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    root = Path(__file__).resolve().parent

    train_path = root / "final_dataset_clean.json"
    test_path  = root / "Finance_test.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    print(f"[1/4] Loading train: {train_path.name}")
    train_fmt, train_records = load_json_or_jsonl(train_path)
    print(f"      train format={train_fmt}, records={len(train_records)}")

    print(f"[2/4] Loading test : {test_path.name}")
    test_fmt, test_records = load_json_or_jsonl(test_path)
    print(f"      test format ={test_fmt}, records={len(test_records)}")

    print("[3/4] Building test fingerprints...")
    test_fp_set = set()
    for r in tqdm(test_records, desc="Fingerprint(test)", ncols=100):
        test_fp_set.add(fingerprint(r))

    print("[4/4] Filtering train (remove overlaps with test)...")
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for r in tqdm(train_records, desc="Filter(train)", ncols=100):
        fp = fingerprint(r)
        if fp in test_fp_set:
            removed.append(r)
        else:
            kept.append(r)

    out_path = root / "final_dataset_clean_no_test.json"
    report_path = root / "remove_report.json"

    save_like_input(train_fmt, out_path, kept)

    # 统计报告（含少量样例，方便你确认删对了）
    report = {
        "train_in": len(train_records),
        "test_in": len(test_records),
        "removed_from_train": len(removed),
        "train_out": len(kept),
        "output_file": str(out_path),
        "note": "removed items are exact matches based on (instruction,input,output) fingerprint",
        "removed_examples": removed[:5],
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Done ===")
    print(f"Saved: {out_path.name} (records={len(kept)})")
    print(f"Report: {report_path.name} (removed={len(removed)})")

if __name__ == "__main__":
    main()
