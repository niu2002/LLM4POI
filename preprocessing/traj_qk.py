import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def simplify_poi_category(text):
    return re.sub(r"\[\{'url': '[^']+', 'name': '([^']+)'}\]", r"\1", str(text))


def prepare_dataframe(path):
    data = pd.read_csv(path)
    required = ["UserId", "pseudo_session_trajectory_id", "UTCTimeOffset", "PoiId", "PoiCategoryName"]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing required column {col!r} in {path}")
    data["UTCTimeOffset"] = pd.to_datetime(data["UTCTimeOffset"])
    if "UTCTimeOffsetEpoch" not in data.columns:
        data["UTCTimeOffsetEpoch"] = data["UTCTimeOffset"].astype("int64") // 1_000_000_000
    data["PoiCategoryName"] = data["PoiCategoryName"].apply(simplify_poi_category)
    if "PoiCategoryId" not in data.columns:
        data["PoiCategoryId"] = pd.factorize(data["PoiCategoryName"].astype(str))[0]
    return data


def row_to_sentence(row, user):
    return (
        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} "
        f"which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
    )


def generate_kq_pairs(main_data, history_tail=5):
    # This follows the original paper code structure: query is the full trajectory,
    # key is the prefix of the current trajectory, or recent same-user history for
    # single-check-in trajectories.
    main_data = main_data.sort_values(by=["UserId", "pseudo_session_trajectory_id", "UTCTimeOffsetEpoch"])
    key_query_pairs = []

    for user in tqdm(main_data["UserId"].unique(), desc="Generating K/Q pairs", unit="user"):
        user_data = main_data[main_data["UserId"] == user].sort_values("UTCTimeOffsetEpoch")
        for traj_id in user_data["pseudo_session_trajectory_id"].unique():
            user_trajectory_data = user_data[user_data["pseudo_session_trajectory_id"] == traj_id]
            start_time = user_trajectory_data["UTCTimeOffsetEpoch"].min()
            end_time = user_trajectory_data["UTCTimeOffsetEpoch"].max()

            query = [f"The following data is a trajectory of user {user}:"]
            for _, row in user_trajectory_data.iterrows():
                query.append(row_to_sentence(row, user))
            query = " ".join(query)

            if len(user_trajectory_data) == 1:
                prev_trajectories = user_data[user_data["UTCTimeOffsetEpoch"] < start_time]
                if prev_trajectories.empty:
                    continue
                key_rows = prev_trajectories.tail(history_tail)
            else:
                key_rows = user_trajectory_data.iloc[:-1]

            key = [f"The following data is a trajectory of user {user}:"]
            for _, row in key_rows.iterrows():
                key.append(row_to_sentence(row, user))
            key = " ".join(key)

            key_query_pairs.append((key, query, str(traj_id), str(start_time), str(end_time)))

    return key_query_pairs


def resolve_paths(args):
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path("..") / "datasets" / args.dataset_name / "preprocessed"
    train_csv = Path(args.train_csv) if args.train_csv else data_dir / "train_sample.csv"
    test_csv = Path(args.test_csv) if args.test_csv else data_dir / "test_sample_with_traj.csv"
    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    return train_csv, test_csv, out_dir


def main():
    parser = argparse.ArgumentParser(description="Generate original LLM4POI K/Q trajectory pairs.")
    parser.add_argument("-dataset_name", "--dataset_name", type=str, choices=["ca", "nyc", "tky"], required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--history_tail", type=int, default=5)
    args = parser.parse_args()

    train_csv, test_csv, out_dir = resolve_paths(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] train_csv={train_csv}")
    print(f"[info] test_csv={test_csv}")
    print(f"[info] out_dir={out_dir}")

    train_data = prepare_dataframe(train_csv)
    test_data = prepare_dataframe(test_csv)

    kq_pairs_train = generate_kq_pairs(train_data, history_tail=args.history_tail)
    qa_dict_train = [
        {
            "key": key,
            "query": query,
            "traj_id": traj_id,
            "user_id": int(str(traj_id).split("_", 1)[0]) if str(traj_id).split("_", 1)[0].isdigit() else None,
            "start_time": start,
            "end_time": end,
        }
        for key, query, traj_id, start, end in kq_pairs_train
    ]
    with (out_dir / "train_kq_pairs.json").open("w", encoding="utf-8") as json_file:
        json.dump(qa_dict_train, json_file, ensure_ascii=False)
    print(f"[info] wrote {len(qa_dict_train)} train K/Q pairs")

    kq_pairs_test = generate_kq_pairs(test_data, history_tail=args.history_tail)
    qa_dict_test = [
        {
            "key": key,
            "query": query,
            "traj_id": traj_id,
            "user_id": int(str(traj_id).split("_", 1)[0]) if str(traj_id).split("_", 1)[0].isdigit() else None,
            "start_time": start,
            "end_time": end,
        }
        for key, query, traj_id, start, end in kq_pairs_test
    ]
    with (out_dir / "test_kq_pairs.json").open("w", encoding="utf-8") as json_file:
        json.dump(qa_dict_test, json_file, ensure_ascii=False)
    print(f"[info] wrote {len(qa_dict_test)} test K/Q pairs")


if __name__ == "__main__":
    main()
