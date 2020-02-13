"""A utility for generating experiment config files.
"""
import json
import copy
import argparse
import itertools
from pathlib import Path
from datetime import datetime
from collections import OrderedDict


def generate_configs(base_config, grid):
    job_queue = []
    timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    hparam_vals = [x for x in grid.values()]
    grid_vals = list(itertools.product(*hparam_vals))
    hparams = list(grid.keys())

    for cfg_vals in grid_vals:
        custom_tokens = [f"{hparam}@{val}" for hparam, val in zip(hparams, cfg_vals)]
        custom_args = "+".join(custom_tokens)
        job = f"--config {base_config} --custom_args {custom_args}"
        job_queue.append(job)

    job_queue_path = f"data/job-queues/latest.txt"
    Path(job_queue_path).parent.mkdir(exist_ok=True, parents=True)
    with open(str(job_queue_path), "w") as f:
        f.write("\n".join(job_queue))
    print(f"Wrote {len(job_queue)} jobs to queue at {job_queue_path}")
    job_queue_path = f"data/job-queues/{Path(base_config).stem}-{timestamp}.txt"
    with open(str(job_queue_path), "w") as f:
        f.write("\n".join(job_queue))
    print(f"Wrote backup {len(job_queue)} jobs to queue at {job_queue_path}")


def parse_grid(key_val_strs):
    print(f"parsing grid str: {key_val_strs}")
    key_val_pairs = key_val_strs.split("+")
    parsed = OrderedDict()
    for pair in key_val_pairs:
        key, val_str = pair.split("@")
        vals = []
        opts = [x for x in val_str.split(":")]
        for token in opts:
            if "," in token:
                val = [x for x in token.split(",") if x]
            else:
                val = token
            vals.append(val)
        parsed[key] = vals
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', default="")
    parser.add_argument('--config', default="configs/msrvtt/only-i3d.json")
    args = parser.parse_args()

    grid = parse_grid(args.grid)
    generate_configs(
        grid=grid,
        base_config=args.config,
    )


if __name__ == "__main__":
    main()
