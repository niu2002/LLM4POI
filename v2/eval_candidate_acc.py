#!/usr/bin/env python
"""Evaluate candidate-constrained Acc@1 for an OpenAI-compatible vLLM server.

This metric gives the model a fixed candidate POI set and asks it to return one
POI id. It is closer to ranking-style Acc@1 than free-form generation, but it is
still a generative approximation rather than the paper's original scorer.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
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


def load_records(dataset_path: Path) -> List[dict]:
    if dataset_path.suffix.lower() == ".parquet":
        import pandas as pd

        df = pd.read_parquet(dataset_path)
        records: List[dict] = []
        question_col = "question" if "question" in df.columns else df.columns[0]
        answer_col = "answer" if "answer" in df.columns else df.columns[1]
        for _, row in df.iterrows():
            records.append({"question": str(row[question_col]), "answer": str(row[answer_col])})
        return records

    records = []
    with dataset_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "question" in item and "answer" in item:
                records.append({"question": str(item["question"]), "answer": str(item["answer"])})
                continue
            messages = item["messages"]
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
            records.append(
                {
                    "question": str(user_messages[-1]["content"]),
                    "answer": str(assistant_messages[-1]["content"]),
                }
            )
    return records


def extract_poi_id(text: Any) -> str:
    value = str(text).strip()
    if "<" in value:
        value = value.split("<", 1)[0].strip()
    value = value.splitlines()[0].strip() if value else value
    match = POI_HEX_RE.search(value) or POI_INT_RE.search(value)
    if match:
        return match.group(0).lower()
    return value.lower()


def build_candidate_pool(records: List[dict]) -> List[str]:
    return sorted({extract_poi_id(record["answer"]) for record in records if extract_poi_id(record["answer"])})


def stable_rng(seed: int, key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def choose_candidates(
    gold: str,
    all_candidates: List[str],
    candidate_size: int,
    seed: int,
) -> List[str]:
    if candidate_size <= 1:
        return [gold]
    distractors = [candidate for candidate in all_candidates if candidate != gold]
    rng = stable_rng(seed, gold)
    if len(distractors) > candidate_size - 1:
        distractors = rng.sample(distractors, candidate_size - 1)
    candidates = [gold] + distractors
    rng.shuffle(candidates)
    return candidates


def build_prompt(question: str, candidates: List[str]) -> str:
    candidate_text = "\n".join(f"- {candidate}" for candidate in candidates)
    return (
        f"{question}\n"
        "<candidate_poi_ids>\n"
        f"{candidate_text}\n"
        "</candidate_poi_ids>\n"
        "Choose exactly one POI id from candidate_poi_ids. Return only the POI id."
    )


async def evaluate_one(client: AsyncOpenAI, record: dict, all_candidates: List[str], args: argparse.Namespace) -> dict:
    gold = extract_poi_id(record["answer"])
    candidates = choose_candidates(gold, all_candidates, args.candidate_size, args.seed)
    question = build_prompt(record["question"], candidates)
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        response = await client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        prediction = extract_poi_id(response.choices[0].message.content)
    except BadRequestError:
        prompt = f"system: {args.system_prompt}\nuser: {question}\nassistant:"
        response = await client.completions.create(
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        prediction = extract_poi_id(response.choices[0].text)

    return {
        "gold": gold,
        "prediction": prediction,
        "correct": prediction == gold,
        "in_candidates": prediction in candidates,
        "candidates": candidates,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate-source", type=Path, default=None)
    parser.add_argument("--candidate-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    args = parser.parse_args(argv)

    data = load_records(args.dataset)
    candidate_records = load_records(args.candidate_source or args.dataset)
    all_candidates = build_candidate_pool(candidate_records)
    if args.max_examples is not None:
        data = data[: args.max_examples]

    async def run() -> List[dict]:
        client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
        sem = asyncio.Semaphore(max(1, args.concurrency))
        out: List[Optional[dict]] = [None] * len(data)

        async def guarded(i: int, rec: dict) -> tuple[int, dict]:
            async with sem:
                return i, await evaluate_one(client, rec, all_candidates, args)

        tasks = [asyncio.create_task(guarded(i, rec)) for i, rec in enumerate(data)]
        for fut in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="CandidateAcc",
            unit="example",
            disable=not args.show_progress,
        ):
            i, result = await fut
            out[i] = result
        return [item for item in out if item is not None]

    results = asyncio.run(run())
    total = len(results)
    correct = sum(int(item["correct"]) for item in results)
    invalid = sum(int(not item["in_candidates"]) for item in results)
    print(f"Examples evaluated: {total}")
    print(f"Candidate Acc@1: {correct / total if total else 0.0:.4f} ({correct}/{total})")
    print(f"Predictions outside candidate set: {invalid}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Detailed predictions written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
