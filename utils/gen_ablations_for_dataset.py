"""Generate a set of ablations for each dataset, using the config structure of the
MSRVTT experiments.

ipy utils/gen_ablations_for_dataset.py -- --refresh --dest_dataset didemo \
    --update_ablation_list 1

"""
import json
import argparse
from pathlib import Path


def handle_moee_config(config):
    """For the official ablations on MSRVTT, we provide MoEE with the same hyperparam
    budget as CE and run a search to find the best hyperparams.  For the unofficial
    ablations, we use the same padding/VLAD settings as CE.
    """
    config = {
        "inherit_from": config["inherit_from"],
        "arch": {"type": "CENet", "args": {"use_ce": ""}},
    }
    return config


def remove_audio_streams(config, dest_path):
    """Prune audio-based features from the config and dest_path name (necessary for
    datasets like MSVD which do not possess sound.)  If the audio feature was the control
    variable in the experiment, we return False for the dest_path, such that the ablation
    is removed altogether.
    """
    audio_tags = ["audio", "speech"]
    for audio_tag in audio_tags:
        if f"-{audio_tag}." in dest_path:
            return config, False

        dest_path = dest_path.replace(f"-{audio_tag}", "")
        if "experts" in config and "modalities" in config["experts"]:
            if audio_tag in config["experts"]["modalities"]:
                config["experts"]["modalities"].remove(audio_tag)
    return config, dest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action="store_true")
    parser.add_argument('--update_ablation_list', type=int, default=1)
    parser.add_argument('--src_dataset', default="msrvtt")
    parser.add_argument('--dest_dataset', default="lsmdc")
    parser.add_argument('--exp_list', default="slurm/msrvtt-ablations.txt")
    args = parser.parse_args()

    with open(args.exp_list, "r") as f:
        exps = [x for x in f.read().splitlines() if x]

    print(f"Found {len(exps)} experiments in {args.exp_list}")
    dest_exp_path = Path(args.exp_list.replace("msrvtt", args.dest_dataset))
    if dest_exp_path.exists() and not args.refresh:
        print(f"experiment list found at {dest_exp_path}, skipping...")
        return

    output_rows = []
    exclude = ["miech", "jsfusion"]
    for row in exps:
        flag, config_path, seed_flag, seed_opts = row.split()
        if any([x in config_path for x in exclude]):
            continue
        with open(config_path, "r") as f:
            config = json.load(f)
        if Path(config_path).stem == "train-full-moee":
            config = handle_moee_config(config)
        dest_path = config_path.replace(args.src_dataset, args.dest_dataset)
        config["inherit_from"] = config["inherit_from"].replace(args.src_dataset,
                                                                args.dest_dataset)
        if args.dest_dataset == "msvd":
            config, dest_path = remove_audio_streams(config, dest_path)
            if not dest_path:
                continue

        print(f"writing config to {dest_path}")
        with open(dest_path, "w") as f:
            json.dump(config, f, indent=4, sort_keys=False)
        output_rows.append([flag, dest_path, seed_flag, seed_opts])

    if args.update_ablation_list:
        print(f"Writing new experiment list to {dest_exp_path}")
        output_rows = [" ".join(x) for x in output_rows]
        with open(dest_exp_path, "w") as f:
            for row in sorted(list(set(output_rows))):
                f.write(f"{row}\n")



if __name__ == "__main__":
    main()
