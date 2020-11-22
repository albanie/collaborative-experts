"""Simple aggregation script for experiments

ipy misc/find_latest_checkpoints.py -- --dataset youcook2long
"""
import argparse
from pathlib import Path
from datetime import datetime


def formatted_summary(dataset, exp_root, fname):
    try:
        summaries = list(Path(exp_root).glob(f"**/*{fname}"))
        summaries = [x for x in summaries if dataset in str(x)]
    except FileNotFoundError:
        fname = "summary-seed-1_seed-2_seed-3.json"
        summaries = list(Path(exp_root).glob(f"**/*{fname}"))
        summaries = [x for x in summaries if dataset in str(x)]
    print(f"Found {len(summaries)}")
    latest = {}
    time_format = "%Y-%m-%d_%H-%M-%S"
    for summary in summaries:
        rel_path = summary.relative_to(exp_root)
        key, group, timestamp = rel_path.parts[0], rel_path.parts[1], rel_path.parts[3]
        val = {"timestamp": timestamp, "group": group}
        if key in latest:
            prev_ts = datetime.strptime(latest[key]["timestamp"], time_format)
            curr_ts = datetime.strptime(timestamp, time_format)
            if curr_ts > prev_ts:
                latest[key] = val
        else:
            latest[key] = val
    for key, val in sorted(latest.items()):
        ts, group = val["timestamp"], val["group"]
        print(f'"{key}": ["{group}", "{ts}"],')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lsmdc")
    parser.add_argument("--exp_root", default="data/saved/log")
    parser.add_argument("--fname", default="summary-seed-0_seed-1_seed-2.json")
    args = parser.parse_args()

    formatted_summary(
        fname=args.fname,
        dataset=args.dataset,
        exp_root=args.exp_root,
    )


if __name__ == "__main__":
    main()
