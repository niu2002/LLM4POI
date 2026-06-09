#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


IGNORE_INDEX = -100


@dataclass
class Example:
    prompt_ids: List[int]
    input_ids: List[int]


def build_example(tokenizer, question: str, answer: str, max_length: int) -> Example:
    prompt_messages = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids

    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[-max_length:]
        prompt_ids = []
    else:
        available_prompt_len = max_length - len(answer_ids)
        prompt_ids = prompt_ids[-available_prompt_len:]

    input_ids = prompt_ids + answer_ids
    return Example(prompt_ids=prompt_ids, input_ids=input_ids)


def collate(batch: List[Example], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x.input_ids) for x in batch)
    input_rows = []
    label_rows = []
    attn_rows = []
    for ex in batch:
        pad_len = max_len - len(ex.input_ids)
        labels = list(ex.input_ids)
        prompt_len = min(len(ex.prompt_ids), len(labels))
        labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
        input_rows.append(ex.input_ids + [pad_token_id] * pad_len)
        label_rows.append(labels + [IGNORE_INDEX] * pad_len)
        attn_rows.append([1] * len(ex.input_ids) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attn_rows, dtype=torch.long),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--train_file", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=8)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    df = pd.read_parquet(args.train_file)
    if not {"question", "answer"}.issubset(df.columns):
        raise ValueError("Training parquet must contain 'question' and 'answer' columns")

    examples = [
        build_example(tokenizer, str(row["question"]), str(row["answer"]), args.max_length)
        for _, row in df.iterrows()
    ]

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device if torch.cuda.is_available() else "cpu",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    for _ in range(args.epochs):
        for idx in range(0, len(examples), args.batch_size):
            batch = collate(examples[idx : idx + args.batch_size], tokenizer.pad_token_id or tokenizer.eos_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            print(f"[train] step={step} loss={loss.item():.4f}")
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[info] saved adapter/tokenizer to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
