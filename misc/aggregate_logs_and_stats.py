"""Aggregate logs across multiple seeded runs and summarise their statistics.

ipy misc/aggregate_logs_and_stats.py -- --group_id 3b737e0d
"""
import argparse
import logging
from pathlib import Path
from collections import OrderedDict
from utils.util import read_json
from glob import glob
from logger.log_parser import log_summary


def summarise(group_id, log_dir="data/saved/log", model_dir="data/saved/models"):
    seeded_runs = sorted(list(Path(log_dir).glob(f"**/{group_id}/seed-*")))
    print(f"Found a total of {len(seeded_runs)} seed runs in {group_id}")
    msg = f"Found no seeded runs for group_id: {group_id} in {log_dir}"
    assert len(seeded_runs) > 0, msg

    info_logs = OrderedDict()
    for seeded_run in seeded_runs:
        info_log_matches = list(Path(seeded_run).glob("**/info.log"))
        msg = f"expected to find a single info.log file, found {len(info_log_matches)}"
        assert len(info_log_matches) == 1, msg
        info_logs[seeded_run.stem] = info_log_matches[0]

    summary_log = []
    for seeded_run, info_log_path in info_logs.items():
        with open(info_log_path, "r") as f:
            log = f.read().splitlines()
        summary_log.extend(log)
    first_info_log = list(info_logs.values())[0]
    summary_log_name = f"summary-{'_'.join(list(info_logs.keys()))}.json"
    summary_log_path = first_info_log.parent / summary_log_name
    with open(summary_log_path, "w") as f:
        f.write("\n".join(summary_log))
    print(f"Wrote concatenated logs to {summary_log_path}")

    # retrieve the config from the first run
    rel_path = first_info_log.relative_to(log_dir).parent
    config_path = Path(model_dir) / rel_path / "config.json"
    assert config_path.exists(), f"Could not find config at {config_path}"
    config = read_json(config_path)

    logger = logging.getLogger("summary")

    # some care is required with logging to avoid sending all experiment logs
    # to the same file.  We avoid this by essentially resetting the logging utility

    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=summary_log_path, level=logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    log_summary(
        logger=logger,
        log_path=summary_log_path,
        eval_mode=config["eval_mode"],
        fixed_num_epochs=config["trainer"]["epochs"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_id", default="ed53d01d")
    args = parser.parse_args()
    summarise(group_id=args.group_id)


if __name__ == '__main__':
    main()
