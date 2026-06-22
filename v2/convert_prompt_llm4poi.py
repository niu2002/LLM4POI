#!/usr/bin/env python3
"""
This script converts a check-in CSV (train/test) into a GSM8K-like parquet where:
  - question: a prompt containing user's historical check-ins + current trajectory context
  - answer: the ground-truth next POI id (PoiId) as a string

It is designed to be reproducible and portable:
  - no hardcoded absolute paths
  - configurable via CLI flags
  - minimal dependencies: pandas, pyarrow, tqdm

Expected input CSV columns (train/test):
  - UserId
  - pseudo_session_trajectory_id
  - UTCTimeOffset (parseable datetime string)
  - PoiId
  - PoiCategoryName
  - Latitude, Longitude (optional; not used in this prompt format)

Example:
  python convert_prompt_llm4poi_repro.py \
    --dataset NYC \
    --train_csv datasets/NYC/train.csv \
    --test_csv datasets/NYC/test.csv \
    --out_dir datasets/NYC \
    --history_limit 50
"""

from __future__ import annotations

import argparse
import ast
import json
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def normalize_category(value: Any) -> str:
    """
    Extract a human-readable category from multiple possible formats.
    Handles plain strings and Python-literal-like strings such as:
      "[{'url': '/categories/79', 'name': 'Stadium'}]"
    """
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
    except Exception:
        pass

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return s
        # Try to parse list/dict literals.
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            # Undo common CSV escaping: ""Trader Joe's"" -> "Trader Joe's"
            s2 = s.replace('""', '"')
            try:
                parsed = ast.literal_eval(s2)
            except Exception:
                return s
            return normalize_category(parsed)
        return s

    if isinstance(value, dict):
        for key in ("name", "Name", "category", "Category", "title", "Title"):
            v = value.get(key)
            if v:
                return normalize_category(v)
        # fallback: stringify values
        return ", ".join([normalize_category(v) for v in value.values() if v])

    if isinstance(value, (list, tuple, set)):
        parts = [normalize_category(v) for v in value]
        parts = [p for p in parts if p]
        # de-dup while preserving order
        parts = list(dict.fromkeys(parts))
        return ", ".join(parts) if parts else ""

    return str(value)


def prepare_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("UserId", "pseudo_session_trajectory_id", "UTCTimeOffset", "PoiId", "PoiCategoryName"):
        if col not in df.columns:
            raise ValueError(f"Missing required column {col!r} in {path}")
    df["UTCTimeOffset"] = pd.to_datetime(df["UTCTimeOffset"])
    df.sort_values(["UserId", "UTCTimeOffset", "pseudo_session_trajectory_id"], inplace=True)
    df["category_name"] = df["PoiCategoryName"].apply(normalize_category)
    return df


@dataclass
class HistoryEntry:
    datetime: Any
    time_str: str
    poiid: str
    category: str


@dataclass
class TrajectoryContext:
    trajectory_id: str
    user_id: int
    start_time: Any
    end_time: Any
    entries: List[Dict[str, Any]]
    poi_ids: Set[str]
    categories: Set[str]


def build_history(train_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    history_map[user_id] = {"entries": [HistoryEntry...], "datetimes": [datetime...]}
    Used to fetch user history up to a cutoff time via bisect.
    """
    history: Dict[int, Dict[str, Any]] = {}
    for user_id, group in train_df.groupby("UserId"):
        group = group.sort_values("UTCTimeOffset")
        entries: List[Dict[str, Any]] = []
        datetimes: List[Any] = []
        for _, row in group.iterrows():
            dt = row["UTCTimeOffset"]
            entries.append(
                {
                    "datetime": dt,
                    "time_str": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "poiid": str(row["PoiId"]),
                    "category": row["category_name"],
                }
            )
            datetimes.append(dt)
        history[int(user_id)] = {"entries": entries, "datetimes": datetimes}
    return history


def build_trajectory_contexts(train_df: pd.DataFrame) -> Dict[str, TrajectoryContext]:
    contexts: Dict[str, TrajectoryContext] = {}
    for trajectory_id, group in train_df.groupby("pseudo_session_trajectory_id"):
        group = group.sort_values("UTCTimeOffset")
        if group.empty:
            continue
        entries: List[Dict[str, Any]] = []
        for _, row in group.iterrows():
            dt = row["UTCTimeOffset"]
            entries.append(
                {
                    "time_str": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "poiid": str(row["PoiId"]),
                    "category": row["category_name"],
                }
            )
        contexts[str(trajectory_id)] = TrajectoryContext(
            trajectory_id=str(trajectory_id),
            user_id=int(group.iloc[0]["UserId"]),
            start_time=group.iloc[0]["UTCTimeOffset"],
            end_time=group.iloc[-1]["UTCTimeOffset"],
            entries=entries,
            poi_ids={str(v) for v in group["PoiId"].tolist()},
            categories={str(v) for v in group["category_name"].tolist()},
        )
    return contexts


def build_poi_trajectory_index(contexts: Dict[str, TrajectoryContext]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = defaultdict(list)
    for trajectory_id, ctx in contexts.items():
        for poi_id in ctx.poi_ids:
            index[poi_id].append(trajectory_id)
    return dict(index)


def jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def get_similar_trajectories(
    trajectory_contexts: Dict[str, TrajectoryContext],
    poi_index: Dict[str, List[str]],
    query_entries: Sequence[Dict[str, Any]],
    user_id: int,
    cutoff_time: Any,
    limit: int,
) -> List[TrajectoryContext]:
    if limit <= 0 or not query_entries:
        return []

    query_pois = {str(entry["poiid"]) for entry in query_entries}
    query_categories = {str(entry["category"]) for entry in query_entries}

    candidate_ids: Set[str] = set()
    for poi_id in query_pois:
        candidate_ids.update(poi_index.get(poi_id, []))
    if len(candidate_ids) < limit * 3:
        for trajectory_id, ctx in trajectory_contexts.items():
            if ctx.categories & query_categories:
                candidate_ids.add(trajectory_id)

    scored = []
    for trajectory_id in candidate_ids:
        ctx = trajectory_contexts[trajectory_id]
        if ctx.user_id == user_id or ctx.end_time >= cutoff_time:
            continue
        poi_score = jaccard(query_pois, ctx.poi_ids)
        category_score = jaccard(query_categories, ctx.categories)
        score = poi_score * 2.0 + category_score
        if score <= 0:
            continue
        scored.append((score, ctx.end_time, ctx))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [ctx for _, _, ctx in scored[:limit]]


def format_similar_trajectories(trajectories: List[TrajectoryContext], entry_limit: int) -> str:
    if not trajectories:
        return "Other-User Similar Trajectories:\nNone.\n"

    lines = ["Other-User Similar Trajectories:"]
    for rank, ctx in enumerate(trajectories, start=1):
        lines.append(f"[{rank}] user={ctx.user_id}, trajectory={ctx.trajectory_id}")
        for entry in ctx.entries[-entry_limit:]:
            lines.append(f"{entry['time_str']} | {entry['poiid']} | {entry['category']}")
    return "\n".join(lines) + "\n"


def load_kqt_top_map(path: Path | None) -> Dict[str, List[str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"KQT top map must be a JSON object: {path}")
    top_map: Dict[str, List[str]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            top_map[str(key)] = [str(item) for item in value]
    return top_map


def get_kqt_trajectories(
    trajectory_contexts: Dict[str, TrajectoryContext],
    kqt_top_map: Dict[str, List[str]],
    trajectory_id: str,
    limit: int,
) -> List[TrajectoryContext]:
    if not kqt_top_map:
        return []
    trajectories: List[TrajectoryContext] = []
    for similar_id in kqt_top_map.get(str(trajectory_id), [])[:limit]:
        ctx = trajectory_contexts.get(str(similar_id))
        if ctx is not None:
            trajectories.append(ctx)
    return trajectories


def get_history_entries(history_map: Dict[int, Dict[str, Any]], user_id: int, cutoff_time, limit: int) -> List[Dict[str, Any]]:
    user_history = history_map.get(int(user_id))
    if not user_history:
        return []
    idx = bisect_left(user_history["datetimes"], cutoff_time)
    entries = user_history["entries"][:idx]
    return entries[-limit:]


def format_entries(entries: List[Dict[str, Any]], header: str) -> str:
    if not entries:
        return header + "\nNone.\n"
    lines = [header]
    for e in entries:
        lines.append(f"{e['time_str']} | {e['poiid']} | {e['category']}")
    return "\n".join(lines) + "\n"


def build_samples(
    df: pd.DataFrame,
    history_map: Dict[int, Dict[str, Any]],
    trajectory_contexts: Dict[str, TrajectoryContext],
    poi_index: Dict[str, List[str]],
    dataset_name: str,
    dataset_split: str,
    history_limit: int,
    include_other_users: bool,
    similar_trajectory_limit: int,
    similar_entry_limit: int,
    kqt_top_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    poi_ids = df["PoiId"].astype(str).str.strip()
    if poi_ids.str.fullmatch(r"[0-9a-fA-F]{24}").all():
        poi_hint = "POI id is a 24-character hexadecimal identifier."
    elif poi_ids.str.fullmatch(r"\d+").all():
        numeric_ids = poi_ids.astype(int)
        poi_hint = f"POI id is an integer in the range from {numeric_ids.min()} to {numeric_ids.max()}."
    else:
        poi_hint = "Return the POI id exactly as it appears in the trajectory data."

    user_template = (
        "You will be given history and current trajectory data of a user from {dataset}.\n"
        "<history>\n"
        "{history_section_header}\n"
        "{history_section}"
        "</history>\n"
        "{other_users_block}"
        "<current>\n"
        "The following is the current trajectory of user {user_id}:\n"
        "The trajectories consist of check-in records and each check-in record is represented as a tuple "
        "as (time, poi_id, poi category):\n"
        "{current_section}"
        "</current>\n"
        "Given the data, at {target_time}, which POI id will user {user_id} visit? "
        "{poi_hint} Return only the POI id without explanation.\n"
    )

    samples: List[Dict[str, Any]] = []

    for trajectory_id, group in tqdm(
        df.groupby("pseudo_session_trajectory_id"),
        desc=f"Building {dataset_name} {dataset_split} samples",
        unit="traj",
    ):
        group = group.sort_values("UTCTimeOffset")
        if len(group) < 2:
            continue

        user_id = int(group.iloc[0]["UserId"])
        target_row = group.iloc[-1]
        current_rows = group.iloc[:-1]
        if current_rows.empty:
            continue

        start_time = current_rows.iloc[0]["UTCTimeOffset"]
        history_entries = get_history_entries(history_map, user_id, start_time, limit=history_limit)
        history_header = (
            "Same-User Historical Trajectories (from the same user's past trajectories):\n"
            "Each entry is formatted as (time, poi_id, poi category)."
        )
        history_text = format_entries(history_entries, header="Entries:")

        current_entries: List[Dict[str, Any]] = []
        for _, row in current_rows.iterrows():
            current_entries.append(
                {
                    "time_str": row["UTCTimeOffset"].strftime("%Y-%m-%d %H:%M:%S"),
                    "poiid": str(row["PoiId"]),
                    "category": row["category_name"],
                }
            )
        current_text = format_entries(current_entries, "the most recent entries (time, poi_id, poi category):")
        other_users_block = ""
        other_users_source = "none"
        if include_other_users:
            similar_trajectories = get_kqt_trajectories(
                trajectory_contexts=trajectory_contexts,
                kqt_top_map=kqt_top_map,
                trajectory_id=str(trajectory_id),
                limit=similar_trajectory_limit,
            )
            other_users_source = "kqt" if similar_trajectories else "jaccard"
            if not similar_trajectories:
                similar_trajectories = get_similar_trajectories(
                    trajectory_contexts=trajectory_contexts,
                    poi_index=poi_index,
                    query_entries=current_entries,
                    user_id=user_id,
                    cutoff_time=start_time,
                    limit=similar_trajectory_limit,
                )
            other_users_block = (
                "<other_users>\n"
                f"The following trajectories come from historical users before the current time and are similar to the current trajectory. Source: {other_users_source}.\n"
                f"{format_similar_trajectories(similar_trajectories, entry_limit=similar_entry_limit)}"
                "</other_users>\n"
            )

        user_prompt = user_template.format(
            dataset=dataset_name,
            history_section_header=history_header,
            history_section=history_text,
            other_users_block=other_users_block,
            current_section=current_text,
            user_id=user_id,
            target_time=target_row["UTCTimeOffset"].strftime("%Y-%m-%d %H:%M:%S"),
            poi_hint=poi_hint,
        )

        target_poiid = str(target_row["PoiId"])
        samples.append(
            {
                "user_prompt": user_prompt,
                "assistant_prompt": target_poiid,
                "ground_truth": target_poiid,
                "user_id": user_id,
                "trajectory_id": int(trajectory_id) if str(trajectory_id).isdigit() else trajectory_id,
                "target_time": target_row["UTCTimeOffset"].strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_split": dataset_split,
                "include_other_users": include_other_users,
                "other_users_source": other_users_source,
            }
        )

    return samples


def write_parquet(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="NYC", help="Dataset name used in prompt text (default: NYC).")
    ap.add_argument("--train_csv", type=Path, required=True)
    ap.add_argument("--test_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--history_limit", type=int, default=50, help="Max number of history entries per prompt (default: 50).")
    ap.add_argument("--include_other_users", action="store_true", help="Add similar trajectories from other users to each prompt.")
    ap.add_argument("--similar_trajectory_limit", type=int, default=20, help="Max other-user trajectories per prompt.")
    ap.add_argument("--similar_entry_limit", type=int, default=5, help="Max entries shown for each similar trajectory.")
    ap.add_argument("--kqt_train_json", type=Path, default=None, help="Optional train_key_top200.json from traj_sim.py.")
    ap.add_argument("--kqt_test_json", type=Path, default=None, help="Optional test_key_top200.json from traj_sim.py.")
    ap.add_argument("--write_jsonl", action="store_true", help="Also write raw samples as jsonl for debugging.")
    args = ap.parse_args()

    train_df = prepare_dataframe(args.train_csv)
    test_df = prepare_dataframe(args.test_csv)

    history_map = build_history(train_df)
    trajectory_contexts = build_trajectory_contexts(train_df)
    poi_index = build_poi_trajectory_index(trajectory_contexts)
    kqt_train_top_map = load_kqt_top_map(args.kqt_train_json)
    kqt_test_top_map = load_kqt_top_map(args.kqt_test_json)

    train_samples = build_samples(
        train_df,
        history_map,
        trajectory_contexts,
        poi_index,
        dataset_name=args.dataset,
        dataset_split="train",
        history_limit=args.history_limit,
        include_other_users=args.include_other_users,
        similar_trajectory_limit=args.similar_trajectory_limit,
        similar_entry_limit=args.similar_entry_limit,
        kqt_top_map=kqt_train_top_map,
    )
    test_samples = build_samples(
        test_df,
        history_map,
        trajectory_contexts,
        poi_index,
        dataset_name=args.dataset,
        dataset_split="test",
        history_limit=args.history_limit,
        include_other_users=args.include_other_users,
        similar_trajectory_limit=args.similar_trajectory_limit,
        similar_entry_limit=args.similar_entry_limit,
        kqt_top_map=kqt_test_top_map,
    )

    def to_gsm(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"question": s["user_prompt"], "answer": s["assistant_prompt"]} for s in samples]

    gsm_train = to_gsm(train_samples)
    gsm_test = to_gsm(test_samples)

    out_train = args.out_dir / f"{args.dataset.lower()}_gsm8k_train_llm4poi.parquet"
    out_test = args.out_dir / f"{args.dataset.lower()}_gsm8k_test_llm4poi.parquet"
    write_parquet(out_train, gsm_train)
    write_parquet(out_test, gsm_test)
    print(f"[info] wrote: {out_train}")
    print(f"[info] wrote: {out_test}")

    if args.write_jsonl:
        write_jsonl(args.out_dir / f"{args.dataset.lower()}_llm4poi_train_samples.jsonl", train_samples)
        write_jsonl(args.out_dir / f"{args.dataset.lower()}_llm4poi_test_samples.jsonl", test_samples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


