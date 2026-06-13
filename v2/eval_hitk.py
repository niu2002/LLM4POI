#!/usr/bin/env python
"""Evaluate generative Hit@K for an OpenAI-compatible vLLM server.

This script asks the model for multiple sampled answers per example and reports
whether the gold POI id appears in the first K generated candidates. It is a
practical v2 metric for the current generative serving path; it is not a full
candidate-pool logprob reranker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

from tqdm import tqdm

try:
    from openai import AsyncOpenAI, BadRequestError
except ImportError as exc:  # pragma: no cover
    raise SystemExit("openai package is required. Install via `pip install openai`.") from exc


POI_HEX_RE = re.compile(r"\b[0-9a-fA-F]{24}\b")
POI_INT_RE = re.compile(r"\b\d+\b")


def load_messages(dataset_path: Path) -> List[dict]:
    if dataset_path.suffix.lower() == ".parquet":
        import pandas as pd

        df = pd.read_parquet(dataset_path)
        records: List[dict] = []
        if "messages" in df.columns:
            for entry in df["messages"]:
                records.append(json.loads(entry) if isinstance(entry, str) else entry)
        else:
            question_col = "question" if "question" in df.columns else df.columns[0]
            answer_col = "answer" if "answer" in df.columns else df.columns[1]
            for _, row in df.iterrows():
                records.append(
                    {
                        "messages": [
                            {"role": "user", "content": str(row[question_col])},
                            {"role": "assistant", "content": str(row[answer_col])},
                        ]
                    }
                )
        return records

    records = []
    with dataset_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def message_to_text(message: Any) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    pieces.append(part["text"])
                elif "content" in part:
                    pieces.append(str(part["content"]))
            else:
                pieces.append(str(part))
        return "".join(pieces)
    return str(content)


def normalize_answer(text: Any, extract_poi_id: bool = True) -> str:
    value = message_to_text(text).strip()
    if "<" in value:
        value = value.split("<", 1)[0].strip()
    value = value.splitlines()[0].strip() if value else value
    if extract_poi_id:
        match = POI_HEX_RE.search(value) or POI_INT_RE.search(value)
        if match:
            return match.group(0).lower()
    return value.strip()


def prepare_prompt(record: dict, system_prompt: str) -> tuple[List[dict], str]:
    messages = record["messages"]
    gt_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    if not gt_messages:
        raise ValueError("No assistant answer found in dataset entry.")
    gold = normalize_answer(gt_messages[-1].get("content"))

    prompt_messages = [msg for msg in messages if msg.get("role") != "assistant"]
    if prompt_messages and prompt_messages[0].get("role") == "system":
        prompt_messages = prompt_messages[1:]
    prompt_messages = [{"role": "system", "content": system_prompt}] + prompt_messages
    return prompt_messages, gold


async def evaluate_one(
    client: AsyncOpenAI,
    record: dict,
    args: argparse.Namespace,
) -> dict:
    prompt_messages, gold = prepare_prompt(record, args.system_prompt)

    async def create_completion(prompt: str, max_tokens: int):
        return await client.completions.create(
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=max_tokens,
            n=args.num_return_sequences,
        )

    try:
        response = await client.chat.completions.create(
            model=args.model,
            messages=prompt_messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            n=args.num_return_sequences,
        )
        predictions = [
            normalize_answer(choice.message, extract_poi_id=args.extract_poi_id)
            for choice in response.choices
        ]
    except BadRequestError:
        # Some models only expose the completions endpoint. Keep the fallback
        # simple so the output remains comparable to chat-completion runs.
        prompt = "\n".join(
            f"{msg.get('role', 'user')}: {message_to_text(msg.get('content'))}"
            for msg in prompt_messages
        )
        prompt = f"{prompt}\nassistant:"
        try:
            response = await create_completion(prompt, args.max_new_tokens)
        except BadRequestError as exc:
            if "maximum context length" not in str(exc):
                raise
            response = await create_completion(prompt, args.context_retry_max_tokens)
        predictions = [
            normalize_answer(choice.text, extract_poi_id=args.extract_poi_id)
            for choice in response.choices
        ]

    hits = {f"hit@{k}": gold in predictions[:k] for k in args.k_values}
    return {
        "gold": gold,
        "predictions": predictions,
        **hits,
        "prompt": prompt_messages,
    }


def parse_k_values(raw: str, num_return_sequences: int) -> List[int]:
    values = sorted({int(item) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("--k-values must contain at least one integer.")
    too_large = [k for k in values if k > num_return_sequences]
    if too_large:
        raise ValueError(
            f"K values {too_large} require --num-return-sequences >= {max(too_large)}."
        )
    return values


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL file to store detailed predictions.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--context-retry-max-tokens",
        type=int,
        default=16,
        help="Retry with fewer output tokens when a long prompt exceeds context length.",
    )
    parser.add_argument("--num-return-sequences", type=int, default=20)
    parser.add_argument("--k-values", default="1,5,10,20")
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    parser.add_argument(
        "--no-extract-poi-id",
        dest="extract_poi_id",
        action="store_false",
        help="Disable 24-hex POI id extraction from generated text.",
    )
    parser.set_defaults(extract_poi_id=True)
    args = parser.parse_args(argv)
    args.k_values = parse_k_values(args.k_values, args.num_return_sequences)

    data = load_messages(args.dataset)
    if args.max_examples is not None:
        data = data[: args.max_examples]

    async def run() -> List[dict]:
        client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
        sem = asyncio.Semaphore(max(1, args.concurrency))
        out: List[Optional[dict]] = [None] * len(data)

        async def guarded(i: int, rec: dict) -> tuple[int, dict]:
            async with sem:
                return i, await evaluate_one(client, rec, args)

        tasks = [asyncio.create_task(guarded(i, rec)) for i, rec in enumerate(data)]
        for fut in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Evaluating",
            unit="example",
            disable=not args.show_progress,
        ):
            i, result = await fut
            out[i] = result
        return [item for item in out if item is not None]

    results = asyncio.run(run())
    total = len(results)
    print(f"Examples evaluated: {total}")
    for k in args.k_values:
        key = f"hit@{k}"
        hit = sum(int(item[key]) for item in results)
        score = hit / total if total else 0.0
        print(f"Hit@{k}: {score:.4f} ({hit}/{total})")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Detailed predictions written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
