# eval_multi_lora_adapters_localmetrics.py
# - Always evaluate baseline (Qwen/Qwen2.5-7B-Instruct by default)
# - Optionally evaluate multiple LoRA adapters (no merge) by hot-switching adapters via PEFT
# - Optionally evaluate additional full finetuned models (merged checkpoints) too
# - Save into timestamped folder each run: eval_runs/YYYYMMDD_HHMMSS[_tag]/
#
# Files expected in current folder (default):
#   Finance_test.json
#   general_test.json

import os
# If you use hf-mirror for models, keep this; otherwise you can remove it.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Optional offline flags (only effective if your models/tokenizers are already cached locally)
# os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import csv
import json
import re
import time
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.switch_backend("Agg")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from rouge_score import rouge_scorer
import sacrebleu


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# IO: support JSON array or JSONL
# -----------------------------
def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore").lstrip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} is JSON but not a list.")
        return [x for x in data if isinstance(x, dict)]

    out = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except json.JSONDecodeError:
                print(f"[WARN] {path.name} line {i} invalid JSON; skipped.")
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# Prompting (Qwen Instruct via chat template)
# -----------------------------
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_chat_prompt(tokenizer, instruction: str, user_input: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": instruction if not user_input else f"{instruction}\n\nInput: {user_input}",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


# -----------------------------
# Generation
# -----------------------------
@dataclass
class GenConfig:
    max_new_tokens: int = 256
    do_sample: bool = False      # reproducible default (greedy)
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


def batched_generate(
    model,
    tokenizer,
    prompts: List[str],
    gen_cfg: GenConfig,
    max_input_tokens: int,
    batch_size: int,
    desc: str,
) -> List[str]:
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    outputs: List[str] = []
    model.eval()

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(prompts), batch_size),
                  desc=desc,
                  total=total_batches,
                  ncols=100):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=gen_cfg.max_new_tokens,
                do_sample=gen_cfg.do_sample,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                repetition_penalty=gen_cfg.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j in range(gen_ids.size(0)):
            nonpad = int(inputs["attention_mask"][j].sum().item())
            gen_part = gen_ids[j][nonpad:]
            text = tokenizer.decode(gen_part, skip_special_tokens=True)
            outputs.append(text.strip())

    return outputs


# -----------------------------
# Metrics (local-only): ROUGE via rouge-score, BLEU/chrF via sacrebleu
# -----------------------------
_ROUGE_SCORER = None


def exact_match(pred: str, ref: str) -> int:
    return int(normalize_text(pred).lower() == normalize_text(ref).lower())


def _get_rouge_scorer():
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        # use_stemmer=True is common for English ROUGE
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return _ROUGE_SCORER


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, Any]:
    """
    Returns:
      rouge: rouge1/rouge2/rougeL F1 averaged over examples
      bleu: corpus BLEU (sacrebleu)
      chrf: corpus chrF (sacrebleu)
      exact_match: mean EM
      length stats
    """
    if not preds:
        return {
            "n": 0,
            "exact_match": 0.0,
            "rouge": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            "bleu": {"bleu": 0.0},
            "chrf": {"chrf": 0.0},
            "length_pred_words": {"mean": 0.0, "p50": 0.0, "p90": 0.0},
            "length_ref_words": {"mean": 0.0, "p50": 0.0, "p90": 0.0},
        }

    scorer = _get_rouge_scorer()

    r1, r2, rl = 0.0, 0.0, 0.0
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)  # (target, prediction)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure

    n = len(preds)
    rouge_scores = {
        "rouge1": float(r1 / n),
        "rouge2": float(r2 / n),
        "rougeL": float(rl / n),
    }

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    em = float(np.mean([exact_match(p, r) for p, r in zip(preds, refs)]))

    pred_lens = [len(p.split()) for p in preds]
    ref_lens = [len(r.split()) for r in refs]

    def _stats(xs: List[int]) -> Dict[str, float]:
        arr = np.array(xs) if xs else np.array([0])
        return {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    return {
        "n": n,
        "exact_match": em,
        "rouge": rouge_scores,
        "bleu": {"bleu": float(bleu)},
        "chrf": {"chrf": float(chrf)},
        "length_pred_words": _stats(pred_lens),
        "length_ref_words": _stats(ref_lens),
    }


# -----------------------------
# Helpers: model naming / adapter detection
# -----------------------------
def safe_name(s: str) -> str:
    s = s.strip().replace("\\", "/")
    s = s.rstrip("/")
    s = s.split("/")[-1] if "/" in s else s
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:120] if s else "model"


def try_get_adapter_base(adapter_path: str) -> Optional[str]:
    try:
        cfg = PeftConfig.from_pretrained(adapter_path)
        return getattr(cfg, "base_model_name_or_path", None)
    except Exception:
        return None


def load_base_model_and_tokenizer(base_model: str, dtype: str, attn_impl: str):
    # dtype
    torch_dtype = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype, "auto")
    if dtype == "auto":
        # On RTX 4090D, bf16 is usually a good default if supported by your stack
        torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # attn implementation is optional; if unsupported, fallback
    kwargs = dict(
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if attn_impl and attn_impl != "auto":
        kwargs["attn_implementation"] = attn_impl

    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    except TypeError:
        # transformers version may not support attn_implementation
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    return model, tokenizer


# -----------------------------
# Split evaluation runner
# -----------------------------
def run_split(
    model,
    tokenizer,
    split_name: str,
    data_path: Path,
    out_dir: Path,
    gen_cfg: GenConfig,
    max_input_tokens: int,
    batch_size: int,
) -> Dict[str, Any]:
    data = read_json_or_jsonl(data_path)

    prompts, refs, metas = [], [], []
    for idx, ex in enumerate(tqdm(data, desc=f"Build {split_name}", ncols=100)):
        inst = ex.get("instruction", "")
        inp = ex.get("input", "")
        ref = ex.get("output", "")

        prompts.append(build_chat_prompt(tokenizer, inst, inp))
        refs.append(ref)
        metas.append({"id": idx, "instruction": inst, "input": inp, "reference": ref})

    t0 = time.time()
    preds = batched_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        gen_cfg=gen_cfg,
        max_input_tokens=max_input_tokens,
        batch_size=batch_size,
        desc=f"Generate {split_name}",
    )
    elapsed = time.time() - t0

    metrics = compute_metrics(preds, refs)
    metrics["latency_sec_total"] = float(elapsed)
    metrics["latency_sec_per_sample"] = float(elapsed / max(1, len(preds)))

    pred_path = out_dir / f"pred_{split_name}.jsonl"
    rows = []
    for m, p in zip(metas, preds):
        r = dict(m)
        r["prediction"] = p
        rows.append(r)
    write_jsonl(pred_path, rows)
    metrics["pred_file"] = str(pred_path)
    return metrics


def evaluate_with_given_model(
    model_label: str,
    model,
    tokenizer,
    finance_path: Path,
    general_path: Path,
    out_dir: Path,
    gen_cfg: GenConfig,
    max_input_tokens: int,
    batch_size: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    finance_metrics = run_split(
        model, tokenizer, "finance", finance_path, out_dir,
        gen_cfg, max_input_tokens, batch_size
    )
    general_metrics = run_split(
        model, tokenizer, "general", general_path, out_dir,
        gen_cfg, max_input_tokens, batch_size
    )

    summary = {
        "model": model_label,
        "gen_cfg": gen_cfg.__dict__,
        "max_input_tokens": max_input_tokens,
        "batch_size": batch_size,
        "finance": finance_metrics,
        "general": general_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def flatten_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for split in ["finance", "general"]:
        m = summary[split]
        rows.append({
            "model": summary["model"],
            "split": split,
            "n": m.get("n", 0),
            "exact_match": m.get("exact_match", 0.0),
            "rougeL": m.get("rouge", {}).get("rougeL", None),
            "rouge1": m.get("rouge", {}).get("rouge1", None),
            "rouge2": m.get("rouge", {}).get("rouge2", None),
            "bleu": m.get("bleu", {}).get("bleu", None),
            "chrf": m.get("chrf", {}).get("chrf", None),
            "latency_sec_per_sample": m.get("latency_sec_per_sample", None),
            "pred_file": m.get("pred_file", ""),
        })
    return rows


def compute_delta_vs_baseline(all_summaries: List[Dict[str, Any]], baseline_label: str) -> Dict[str, Any]:
    base = None
    for s in all_summaries:
        if s["model"] == baseline_label:
            base = s
            break
    if base is None:
        return {}

    delta = {}
    for s in all_summaries:
        if s["model"] == baseline_label:
            continue
        delta[s["model"]] = {}
        for split in ["finance", "general"]:
            b = base[split]
            f = s[split]
            delta[s["model"]][split] = {
                "exact_match": f["exact_match"] - b["exact_match"],
                "rougeL": f["rouge"]["rougeL"] - b["rouge"]["rougeL"],
                "bleu": f["bleu"]["bleu"] - b["bleu"]["bleu"],
                "chrf": f["chrf"]["chrf"] - b["chrf"]["chrf"],
            }
    return delta


def plot_metrics(rows: List[Dict[str, Any]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["finance", "general"]:
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue

        labels = [safe_name(r["model"]) for r in rs]
        rougeL = [float(r["rougeL"] or 0.0) for r in rs]
        bleu = [float(r["bleu"] or 0.0) for r in rs]
        chrf = [float(r["chrf"] or 0.0) for r in rs]
        em = [float(r["exact_match"] or 0.0) for r in rs]

        x = np.arange(len(labels))
        width = 0.20

        plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
        plt.bar(x - 1.5 * width, rougeL, width, label="ROUGE-L")
        plt.bar(x - 0.5 * width, bleu, width, label="BLEU")
        plt.bar(x + 0.5 * width, chrf, width, label="chrF")
        plt.bar(x + 1.5 * width, em, width, label="ExactMatch")
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.title(f"Model Comparison on {split}")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"compare_{split}.png", dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finance_path", type=str, default="Finance_test.json")
    ap.add_argument("--general_path", type=str, default="general_test.json")

    ap.add_argument("--baseline_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")

    ap.add_argument("--lora_adapters", nargs="*", default=[],
                    help="LoRA adapter dirs/repos (contain adapter_config.json). Optional.")

    ap.add_argument("--full_models", nargs="*", default=[],
                    help="Optional full finetuned models (merged). Optional.")

    ap.add_argument("--adapter_base_model", type=str, default="",
                    help="Base model for adapters. Default: baseline_model. If empty, will try read from adapter_config.json.")

    ap.add_argument("--out_root", type=str, default="eval_runs")
    ap.add_argument("--run_name", type=str, default="", help="Optional tag appended to the timestamp folder name")

    ap.add_argument("--max_input_tokens", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    ap.add_argument("--attn_impl", type=str, default="auto",
                    help="Attention implementation: auto | flash_attention_2 | sdpa. Use auto if unsure.")

    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)

    args = ap.parse_args()
    set_seed(args.seed)

    finance_path = Path(args.finance_path)
    general_path = Path(args.general_path)
    if not finance_path.exists():
        raise FileNotFoundError(f"Finance test file not found: {finance_path.resolve()}")
    if not general_path.exists():
        raise FileNotFoundError(f"General test file not found: {general_path.resolve()}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = timestamp + (f"_{args.run_name}" if args.run_name else "")
    run_dir = Path(args.out_root) / run_folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
    )

    all_summaries: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    models_root = run_dir / "models"

    # -----------------------------
    # (A) Baseline
    # -----------------------------
    print(f"\n[BASELINE] Loading base model: {args.baseline_model}")
    base_model, base_tokenizer = load_base_model_and_tokenizer(
        args.baseline_model, dtype=args.dtype, attn_impl=args.attn_impl
    )

    baseline_label = f"BASE::{args.baseline_model}"
    baseline_out_dir = models_root / safe_name(baseline_label)
    baseline_summary = evaluate_with_given_model(
        model_label=baseline_label,
        model=base_model,
        tokenizer=base_tokenizer,
        finance_path=finance_path,
        general_path=general_path,
        out_dir=baseline_out_dir,
        gen_cfg=gen_cfg,
        max_input_tokens=args.max_input_tokens,
        batch_size=args.batch_size,
    )
    all_summaries.append(baseline_summary)
    all_rows.extend(flatten_summary(baseline_summary))

    # -----------------------------
    # (B) Full merged models (optional)
    # -----------------------------
    for fm in tqdm(args.full_models, desc="Full models", ncols=100):
        print(f"\n[FULL MODEL] Loading: {fm}")
        model, tokenizer = load_base_model_and_tokenizer(fm, dtype=args.dtype, attn_impl=args.attn_impl)
        label = f"FULL::{fm}"
        out_dir = models_root / safe_name(label)
        s = evaluate_with_given_model(
            model_label=label,
            model=model,
            tokenizer=tokenizer,
            finance_path=finance_path,
            general_path=general_path,
            out_dir=out_dir,
            gen_cfg=gen_cfg,
            max_input_tokens=args.max_input_tokens,
            batch_size=args.batch_size,
        )
        all_summaries.append(s)
        all_rows.extend(flatten_summary(s))

    # -----------------------------
    # (C) LoRA adapters (no merge): load once, switch with set_adapter()
    # -----------------------------
    if args.lora_adapters:
        adapter_base = args.adapter_base_model.strip()
        if not adapter_base:
            inferred = try_get_adapter_base(args.lora_adapters[0])
            if inferred:
                adapter_base = inferred
        if not adapter_base:
            adapter_base = args.baseline_model

        # Load adapter base model (reuse baseline if same)
        if adapter_base == args.baseline_model:
            adapter_base_model = base_model
            adapter_base_tokenizer = base_tokenizer
        else:
            print(f"\n[ADAPTER BASE] Loading adapter base model: {adapter_base}")
            adapter_base_model, adapter_base_tokenizer = load_base_model_and_tokenizer(
                adapter_base, dtype=args.dtype, attn_impl=args.attn_impl
            )

        peft_model = None
        adapter_names: List[str] = []

        print("\n[ADAPTERS] Loading LoRA adapters (no merge) ...")
        for k, adapter_path in enumerate(args.lora_adapters):
            name = f"lora_{k+1}_{safe_name(adapter_path)}"
            adapter_names.append(name)

            if peft_model is None:
                peft_model = PeftModel.from_pretrained(
                    adapter_base_model,
                    adapter_path,
                    adapter_name=name,
                    is_trainable=False,
                )
            else:
                peft_model.load_adapter(adapter_path, adapter_name=name)

            inferred = try_get_adapter_base(adapter_path)
            if inferred and inferred != adapter_base:
                print(f"[WARN] Adapter {adapter_path} base_model_name_or_path={inferred} "
                      f"but you are using adapter_base_model={adapter_base}")

        assert peft_model is not None

        for name, adapter_path in zip(adapter_names, args.lora_adapters):
            print(f"\n[ADAPTER EVAL] set_adapter({name}) from {adapter_path}")
            peft_model.set_adapter(name)

            label = f"LoRA::{adapter_base}::{adapter_path}"
            out_dir = models_root / safe_name(label)
            s = evaluate_with_given_model(
                model_label=label,
                model=peft_model,
                tokenizer=adapter_base_tokenizer,
                finance_path=finance_path,
                general_path=general_path,
                out_dir=out_dir,
                gen_cfg=gen_cfg,
                max_input_tokens=args.max_input_tokens,
                batch_size=args.batch_size,
            )
            all_summaries.append(s)
            all_rows.extend(flatten_summary(s))

    # -----------------------------
    # Save run artifacts
    # -----------------------------
    (run_dir / "run_config.json").write_text(
        json.dumps({
            "baseline_model": args.baseline_model,
            "lora_adapters": args.lora_adapters,
            "full_models": args.full_models,
            "adapter_base_model": args.adapter_base_model,
            "finance_path": str(finance_path),
            "general_path": str(general_path),
            "gen_cfg": gen_cfg.__dict__,
            "max_input_tokens": args.max_input_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_csv(run_dir / "summary.csv", all_rows)
    (run_dir / "summary.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    delta = compute_delta_vs_baseline(all_summaries, baseline_label=baseline_label)
    (run_dir / "delta_vs_baseline.json").write_text(json.dumps(delta, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_metrics(all_rows, out_dir=run_dir / "plots")

    print("\n=== Finished ===")
    print(f"Run dir: {run_dir.resolve()}")
    print(f"- summary.csv: {run_dir / 'summary.csv'}")
    print(f"- delta_vs_baseline.json: {run_dir / 'delta_vs_baseline.json'}")
    print(f"- plots: {run_dir / 'plots'}")
    print(f"- per-model artifacts under: {run_dir / 'models'}")


if __name__ == "__main__":
    main()
