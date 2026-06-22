import argparse
import io
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def prepare_dataframe(path):
    data = pd.read_csv(path)
    data["UTCTimeOffset"] = pd.to_datetime(data["UTCTimeOffset"])
    if "UTCTimeOffsetEpoch" not in data.columns:
        data["UTCTimeOffsetEpoch"] = data["UTCTimeOffset"].astype("int64") // 1_000_000_000
    if "PoiCategoryId" not in data.columns:
        data["PoiCategoryId"] = pd.factorize(data["PoiCategoryName"].astype(str))[0]
    return data


def poi_hint(data):
    ids = data["PoiId"].astype(str).str.strip()
    if ids.str.fullmatch(r"[0-9a-fA-F]{24}").all():
        return "Note that POI id is a 24-character hexadecimal identifier. Return only the POI id."
    if ids.str.fullmatch(r"\d+").all():
        numeric = ids.astype(int)
        return f"Note that POI id is an integer in the range from {numeric.min()} to {numeric.max()}. Return only the POI id."
    return "Return only the POI id exactly as it appears in the data."


def generate_qa_pairs(main_data, kqt=None, historical_data=None, args=None):
    # This keeps the original KQT prompt construction, but fixes the lookup and
    # filtering bugs so top similar trajectories are actually used.
    main_data = main_data.sort_values(by=["UserId", "pseudo_session_trajectory_id", "UTCTimeOffsetEpoch"])
    qa_pairs = []
    kqt = kqt or {}
    hint = poi_hint(pd.concat([main_data, historical_data], ignore_index=True) if historical_data is not None else main_data)

    for user in tqdm(main_data["UserId"].unique(), desc="Generating KQT QA", unit="user"):
        user_data = main_data[main_data["UserId"] == user]
        for traj_id in user_data["pseudo_session_trajectory_id"].unique():
            user_trajectory_data = user_data[user_data["pseudo_session_trajectory_id"] == traj_id].copy()
            if len(user_trajectory_data) < 2:
                continue

            start_time = user_trajectory_data["UTCTimeOffsetEpoch"].min()
            top_ids = [str(item) for item in kqt.get(str(traj_id), [])]
            current_len = len(user_trajectory_data)

            if top_ids:
                source_data = historical_data if historical_data is not None else main_data
                user_historical_data = source_data[
                    source_data["pseudo_session_trajectory_id"].astype(str).isin(top_ids)
                    & (source_data["UTCTimeOffsetEpoch"] < start_time)
                ].tail(max(0, 200 - current_len))
            elif historical_data is not None:
                user_historical_data = historical_data[
                    (historical_data["UserId"] == user) & (historical_data["UTCTimeOffsetEpoch"] < start_time)
                ].tail(max(0, 600 - current_len))
            else:
                user_historical_data = user_data[user_data["UTCTimeOffsetEpoch"] < start_time].tail(max(0, 600 - current_len))

            user_trajectory_data.reset_index(drop=True, inplace=True)
            question_parts = [f"<question>: The following data is a trajectory of user {user}:"]
            for _, row in user_trajectory_data.iloc[:-1].iterrows():
                question_parts.append(
                    f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} "
                    f"which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                )

            if not user_historical_data.empty:
                question_parts.append("There is also historical data:")
                for _, row in user_historical_data.iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} "
                        f"which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                    )

            target = user_trajectory_data.iloc[-1]
            question = " ".join(question_parts)
            question += f" Given the data, At {target['UTCTimeOffset']}, Which POI id will user {user} visit? {hint}"
            answer = f"<answer>: At {target['UTCTimeOffset']}, user {user} will visit POI id {target['PoiId']}."
            qa_pairs.append((question, answer))

    return qa_pairs


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
    parser = argparse.ArgumentParser(description="Generate original KQT next-POI QA files.")
    parser.add_argument("-dataset_name", "--dataset_name", type=str, choices=["ca", "nyc", "tky"], required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    train_csv, test_csv, out_dir = resolve_paths(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_data = prepare_dataframe(train_csv)
    test_data = prepare_dataframe(test_csv)
    kqt_train = jload(out_dir / "train_key_top200.json")
    kqt_test = jload(out_dir / "test_key_top200.json")

    qa_pairs_train = generate_qa_pairs(train_data, kqt=kqt_train, historical_data=train_data, args=args)
    qa_pairs_test = generate_qa_pairs(test_data, kqt=kqt_test, historical_data=train_data, args=args)

    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
    with (out_dir / "train_qa_pairs_kqt.json").open("w", encoding="utf-8") as json_file:
        json.dump(qa_dict_train, json_file, ensure_ascii=False)
    print(f"[info] wrote {len(qa_dict_train)} train QA pairs")

    qa_dict_test = [{"question": q, "answer": a} for q, a in qa_pairs_test]
    with (out_dir / "test_qa_pairs_kqt.json").open("w", encoding="utf-8") as json_file:
        json.dump(qa_dict_test, json_file, ensure_ascii=False)
    with (out_dir / "test_qa_pairs_kqt.txt").open("w", encoding="utf-8") as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + "\n")
    print(f"[info] wrote {len(qa_dict_test)} test QA pairs")


if __name__ == "__main__":
    main()
