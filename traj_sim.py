import argparse
import json
import math
import pickle as pkl
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

try:
    from llama_attn_replace_sft import replace_llama_attn
except Exception:  # pragma: no cover
    replace_llama_attn = None

try:
    from gptneox_attn_replace import replace_gpt_neox_attn
except Exception:  # pragma: no cover
    replace_gpt_neox_attn = None


def parse_config():
    parser = argparse.ArgumentParser(description="Compute KQT-style trajectory similarity.")
    parser.add_argument("--dataset_name", type=str, default="nyc", choices=["nyc", "tky", "ca"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None, help="Directory containing train/test_kq_pairs.json.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--context_size", type=int, default=32768)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "attention"])
    parser.add_argument("--exclude_same_user", action="store_true")
    parser.add_argument("--save_embeddings", action="store_true")
    return parser.parse_args()


def jload(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[info] wrote: {path}")


def compute_features(hidden, attention=None, attention_mask=None, pooling="mean"):
    # Original traj_sim used an attention-weighted hidden-state feature. Keep it
    # available, but default to masked mean because attention output is very
    # memory-heavy on long contexts.
    if pooling == "attention" and attention is not None:
        averaged_attention = attention.mean(dim=1)
        weighted_hidden_states = torch.zeros_like(hidden)
        batch_size, sequence_length, _ = hidden.shape
        for i in range(batch_size):
            for j in range(sequence_length):
                weighted_hidden_states[i, j, :] = torch.matmul(averaged_attention[i, j, :], hidden[i, :, :])
        pooled = weighted_hidden_states.mean(dim=(0, 1), keepdim=True)
        return F.normalize(pooled.float(), p=2, dim=-1).squeeze(0)

    if attention_mask is None:
        pooled = hidden.mean(dim=1)
    else:
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.normalize(pooled.float(), p=2, dim=-1)


def dtype_from_name(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.context_size,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = transformers.AutoConfig.from_pretrained(
        args.model_path,
        output_hidden_states=True,
        output_attentions=args.pooling == "attention",
        trust_remote_code=True,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype_from_name(args.torch_dtype),
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    return model, tokenizer


def encode_texts(model, tokenizer, texts, args):
    if args.pooling == "attention" and args.batch_size != 1:
        raise ValueError("--pooling attention requires --batch_size 1 to preserve one feature per trajectory.")
    features = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Encoding K/Q text", unit="batch"):
        batch_texts = texts[start : start + args.batch_size]
        batch = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.seq_len,
        ).to(args.device)
        with torch.no_grad():
            output = model(
                **batch,
                output_hidden_states=True,
                output_attentions=args.pooling == "attention",
            )
            last_attention = output.attentions[-1] if args.pooling == "attention" else None
            pooled = compute_features(
                output.hidden_states[-1],
                attention=last_attention,
                attention_mask=batch["attention_mask"],
                pooling=args.pooling,
            )
            features.append(pooled.detach().cpu())
        del batch, output
        torch.cuda.empty_cache()
    return torch.cat(features, dim=0)


def load_kq(path):
    rows = jload(path)
    rows = [row for row in rows if row.get("key") and row.get("query")]
    return rows


def build_feature_payload(rows, key_embeddings, query_embeddings):
    payload = {}
    for idx, row in enumerate(rows):
        payload[str(row["traj_id"])] = {
            "key": key_embeddings[idx],
            "query": query_embeddings[idx],
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
            "user_id": int(str(row.get("user_id", -1))) if str(row.get("user_id", "-1")).isdigit() else -1,
        }
    return payload


def compute_similarity(target_rows, train_rows, target_key_embeddings, train_query_embeddings, args):
    train_start = torch.tensor([float(row["start_time"]) for row in train_rows])
    train_end = torch.tensor([float(row["end_time"]) for row in train_rows])
    train_user = torch.tensor([int(row.get("user_id", -1)) for row in train_rows])
    target_start = torch.tensor([float(row["start_time"]) for row in target_rows])
    target_user = torch.tensor([int(row.get("user_id", -1)) for row in target_rows])
    train_ids = [str(row["traj_id"]) for row in train_rows]
    results = {}

    train_query_embeddings = train_query_embeddings.to(args.device)
    for idx in tqdm(range(len(target_rows)), desc="Computing trajectory topK", unit="traj"):
        key = target_key_embeddings[idx : idx + 1].to(args.device)
        scores = (key @ train_query_embeddings.T).squeeze(0).detach().cpu()
        valid = train_end < target_start[idx]
        if args.exclude_same_user:
            valid = valid & (train_user != target_user[idx])
        scores[~valid] = -float("inf")
        k = min(args.top_k, int(valid.sum().item()))
        if k <= 0:
            results[str(target_rows[idx]["traj_id"])] = []
            continue
        values, indices = torch.topk(scores, k=k)
        selected = []
        for value, train_idx in zip(values.tolist(), indices.tolist()):
            if math.isfinite(value):
                selected.append(train_ids[train_idx])
        results[str(target_rows[idx]["traj_id"])] = selected
    return results


def main(args):
    seed = 2
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data_path = Path(args.data_path) if args.data_path else Path("datasets") / "processed" / args.dataset_name
    train_kq_path = data_path / "train_kq_pairs.json"
    test_kq_path = data_path / "test_kq_pairs.json"
    print(f"[info] data_path={data_path}")
    print(f"[info] model_path={args.model_path}")
    print(f"[info] pooling={args.pooling} top_k={args.top_k}")

    train_rows = load_kq(train_kq_path)
    test_rows = load_kq(test_kq_path)
    print(f"[info] train K/Q rows={len(train_rows)}")
    print(f"[info] test K/Q rows={len(test_rows)}")

    model, tokenizer = load_model_and_tokenizer(args)
    train_key = encode_texts(model, tokenizer, [row["key"] for row in train_rows], args)
    train_query = encode_texts(model, tokenizer, [row["query"] for row in train_rows], args)
    test_key = encode_texts(model, tokenizer, [row["key"] for row in test_rows], args)

    if args.save_embeddings:
        with (data_path / "train_kqt.pkl").open("wb") as fp:
            pkl.dump(build_feature_payload(train_rows, train_key, train_query), fp)
        with (data_path / "test_kqt.pkl").open("wb") as fp:
            pkl.dump(build_feature_payload(test_rows, test_key, test_key), fp)

    train_results = compute_similarity(train_rows, train_rows, train_key, train_query, args)
    test_results = compute_similarity(test_rows, train_rows, test_key, train_query, args)
    dump_json(data_path / "train_key_top200.json", train_results)
    dump_json(data_path / "test_key_top200.json", test_results)


if __name__ == "__main__":
    main(parse_config())
