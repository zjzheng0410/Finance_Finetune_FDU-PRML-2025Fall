import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple, List

from tqdm import tqdm


# -----------------------------
# 1) 金融关键词规则（可按需增删）
# -----------------------------
# 强金融词：出现一个基本就可以判为 Finance
STRONG_FINANCE_TERMS = [
    # 投资/交易
    r"\bstock(s)?\b", r"\bshares?\b", r"\bequity\b", r"\bETF(s)?\b", r"\bmutual fund(s)?\b",
    r"\bdividend(s)?\b", r"\bportfolio\b", r"\basset(s)?\b", r"\bvaluation\b", r"\bmarket cap\b",
    r"\btrading\b", r"\btrade\b", r"\boptions?\b", r"\bfutures?\b", r"\bshort sell(ing)?\b",
    r"\bexchange\b", r"\bbroker(s)?\b",

    # 债务/借贷/利率
    r"\bloan(s)?\b", r"\bmortgage(s)?\b", r"\bdebt\b", r"\bcredit\b", r"\bAPR\b", r"\bAPY\b",
    r"\binterest rate(s)?\b", r"\byield\b", r"\bbond(s)?\b", r"\bcoupon\b",

    # 税务/会计/监管
    r"\btax(es)?\b", r"\bIRS\b", r"\b1040\b", r"\b1099\b", r"\b1042-S\b", r"\bdeduction(s)?\b",
    r"\bCPA\b", r"\baccounting\b", r"\bbalance sheet\b", r"\bincome statement\b",
    r"\bSEC\b", r"\bprospectus\b",

    # 货币/外汇/宏观
    r"\bforex\b", r"\bFX\b", r"\bcurrency\b", r"\binflation\b", r"\bCPI\b",
    r"\bGDP\b", r"\binterest\b",  # 单独 interest 也比较金融向
    r"\bbank(s)?\b", r"\bchecking account\b", r"\bsavings account\b",

    # 退休账户/理财账户
    r"\b401k\b", r"\bIRA\b", r"\bRoth\b", r"\brecharacteriz(e|ation)\b",
]

# 弱金融词：容易误判（比如 “account” 太泛），所以需要“出现 >=2 次”才算 Finance
WEAK_FINANCE_TERMS = [
    r"\baccount(s)?\b",
    r"\bfinancial(ly)?\b",
    r"\binvest(ment|ing|or|ors)?\b",
    r"\bprice(s|ing)?\b",
    r"\brevenue\b",
    r"\bprofit(s)?\b",
    r"\bbudget(s)?\b",
]

STRONG_PATTERNS = [re.compile(p, flags=re.IGNORECASE) for p in STRONG_FINANCE_TERMS]
WEAK_PATTERNS = [re.compile(p, flags=re.IGNORECASE) for p in WEAK_FINANCE_TERMS]


def normalize_text(example: Dict) -> str:
    """把 instruction/input/output/text 拼起来，便于关键词匹配。"""
    parts = []
    for k in ("instruction", "input", "output", "text"):
        v = example.get(k, "")
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return "\n".join(parts)


def is_finance(example: Dict, weak_threshold: int = 2) -> Tuple[bool, Dict]:
    """
    规则：
      - 强金融词命中 >= 1 => Finance
      - 否则弱金融词命中次数 >= weak_threshold => Finance
      - 否则 => General
    返回：(是否金融, 诊断信息)
    """
    text = normalize_text(example)

    strong_hits = []
    for pat in STRONG_PATTERNS:
        if pat.search(text):
            strong_hits.append(pat.pattern)

    weak_hit_count = 0
    weak_hits = []
    for pat in WEAK_PATTERNS:
        m = pat.findall(text)
        if m:
            weak_hit_count += len(m)
            weak_hits.append((pat.pattern, len(m)))

    finance = (len(strong_hits) > 0) or (weak_hit_count >= weak_threshold)
    debug = {
        "strong_hits": strong_hits[:10],   # 防止太长
        "weak_hits": weak_hits[:10],
        "weak_hit_count": weak_hit_count,
    }
    return finance, debug


def read_jsonl(path: Path) -> Iterable[Dict]:
    """逐行读取 JSONL。遇到坏行会跳过并抛出提示。"""
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # 你也可以选择 raise，这里默认跳过坏行
                print(f"[WARN] Line {idx} JSONDecodeError: {e}")
                continue


def write_jsonl(path: Path, items: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="input test.json (JSONL)")
    ap.add_argument("--out_dir", type=str, default=".", help="output directory")
    ap.add_argument("--weak_threshold", type=int, default=2, help="weak finance hit threshold")
    ap.add_argument("--save_borderline", action="store_true",
                    help="Save borderline cases (weak hits only) into borderline.json")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)

    finance_items = []
    general_items = []
    borderline = []

    # 先统计行数用来让 tqdm 更准确（大文件也能跑，只是会多扫一遍）
    total_lines = sum(1 for _ in in_path.open("r", encoding="utf-8", errors="ignore"))

    for ex in tqdm(read_jsonl(in_path), total=total_lines, desc="Splitting"):
        finance, dbg = is_finance(ex, weak_threshold=args.weak_threshold)

        if finance:
            finance_items.append(ex)
            # borderline：没有强命中，但靠弱命中判为金融（你可以人工检查规则）
            if (len(dbg["strong_hits"]) == 0) and (dbg["weak_hit_count"] >= args.weak_threshold):
                borderline.append({"example": ex, "debug": dbg})
        else:
            general_items.append(ex)

    finance_path = out_dir / "Finance_test.json"
    general_path = out_dir / "general_test.json"

    write_jsonl(finance_path, finance_items)
    write_jsonl(general_path, general_items)

    print("\n==== Done ====")
    print(f"Finance:  {len(finance_items)} -> {finance_path}")
    print(f"General:  {len(general_items)} -> {general_path}")

    if args.save_borderline:
        borderline_path = out_dir / "borderline.json"
        write_jsonl(borderline_path, borderline)
        print(f"Borderline (weak-only): {len(borderline)} -> {borderline_path}")


if __name__ == "__main__":
    main()
