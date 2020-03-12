"""A small utility for filling in the README paths to experiment artifacts.

The template contains tags of the form {{filetype.experiment_name}}, which are then
replaced with the urls for each resource.

Note: print a summary of the most recent results for a given dataset via:
python find_latest_checkpoints.py --dataset lsmdc
"""
import re
import json
import argparse
import importlib
import subprocess
from pathlib import Path
from itertools import zip_longest
from collections import OrderedDict
from typing import Dict

import tqdm
import numpy as np
from millify import millify
from zsvision.zs_beartype import beartype


@beartype
def generate_url(root_url: str, target: str, exp_name: str, experiments: Dict) -> str:
    path_store = {
        "log": {"parent": "log", "fname": "summary-seed-0_seed-1_seed-2.json"},
        "config": {"parent": "models", "fname": "config.json"},
        "model": {"parent": "models", "fname": "trained_model.pth"}
    }
    paths = path_store[target]
    group_id, timestamp = experiments[exp_name]
    rel_path = Path(group_id) / "seed-0" / timestamp / paths["fname"]
    return str(Path(root_url) / paths["parent"] / exp_name / rel_path)


def small_font_str(tokens):
    tokens = [f"<sub><sup>{x}</sup></sub>" for x in tokens]
    return " | ".join(tokens)


def sync_files(experiments, save_dir, webserver, web_dir):
    filetypes = {
        "log": ["summary-seed-0_seed-1_seed-2.json"],
        "models": ["trained_model.pth", "config.json"]
    }
    for key, (group_id, timestamp) in experiments.items():
        # copy experiment artifacts
        for filetype, fnames in filetypes.items():
            for fname in fnames:
                if timestamp.startswith("TODO"):
                    continue
                rel_path = Path(group_id) / "seed-0" / timestamp / fname
                local_path = Path(save_dir) / filetype / key / rel_path
                server_path = Path(web_dir).expanduser() / filetype / key / rel_path
                if not local_path.exists() and "/log/" in str(local_path):
                    # try historical logs
                    old, new = "/log/", "/log-includes-some-final-exps/"
                    local_path = Path(str(local_path).replace(old, new))
                    msg = f"neither original log nor historical data exist ({local_path})"
                    assert local_path.exists(), msg
                dest = f"{webserver}:{str(server_path)}"
                print(f"{key} -> {webserver} [{local_path} -> {server_path}]")
                subprocess.call(["ssh", webserver, "mkdir -p", str(server_path.parent)])
                rsync_args = ["rsync", "-hvrPt", str(local_path), dest]
                print(f"running command {' '.join(rsync_args)}")
                subprocess.call(rsync_args)


def model_specs2path(feat_aggregation, keep, tag=None):
    feat_paths = []
    for model_spec, aggs in feat_aggregation.items():
        if model_spec not in keep:
            continue

        feat_type, model_name, _ = model_spec.split(".")
        base = f"aggregated_{feat_type.replace('-', '_')}"
        required = ("fps", "pixel_dim", "stride")
        fps, pixel_dim, stride = [aggs.get(x, None) for x in required]
        if feat_type in {"facecrops", "faceboxes"}:
            base = f"{base}_{fps}fps_{pixel_dim}px_stride{stride}"
        elif feat_type not in {"ocr", "speech", "audio"}:
            base = f"{base}_{fps}fps_{pixel_dim}px_stride{stride}"

        for option in "offset", "inner_stride":
            if aggs.get(option, None) is not None:
                base += f"_{option}{aggs[option]}"

        for agg in aggs["temporal"].split("-"):
            fname = f"{model_name}-{agg}"
            if aggs["type"] == "logits":
                fname = f"{fname}-logits"
            if tag is not None:
                fname += f"-{tag}"
            feat_paths.append(Path(base) / f"{fname}.pickle")
    return feat_paths


def dataset_paths(dataset, split_name, text_feat):
    name_map = {
        "msvd": "MSVD",
        "lsmdc": "LSMDC",
        "msrvtt": "MSRVTT",
        "didemo": "DiDeMo",
        "activity-net": "ActivityNet",
    }
    class_name = name_map[dataset]
    mod = importlib.import_module(f"data_loader.{class_name}_dataset")
    get_dataset_paths = getattr(getattr(mod, class_name), "dataset_paths")
    if dataset == "activity-net":
        data_dir = dataset
    else:
        data_dir = name_map[dataset]
    root_feat = Path(f"data/{data_dir}/structured-symlinks")
    paths = get_dataset_paths(
        split_name=split_name,
        text_feat=text_feat,
    )
    return root_feat, paths


def generate_tar_lists(save_dir, experiments):
    all_feat_paths = {}
    for exp_name, (group_id, timestamp) in tqdm.tqdm(experiments.items()):
        rel_path = Path(group_id) / "seed-0" / timestamp / "config.json"
        config_path = Path(save_dir) / "models" / exp_name / rel_path
        with open(config_path, "r") as f:
            config = json.load(f)
        feat_aggregation = config["data_loader"]["args"]["feat_aggregation"]
        dataset_name = exp_name.split("-train")[0]
        if dataset_name not in all_feat_paths:
            all_feat_paths[dataset_name] = set()
        split_names = [config["data_loader"]["args"]["split_name"]]
        if "eval_settings" in config and config["eval_settings"]:
            test_split = config["eval_settings"]["data_loader"]["args"]["split_name"]
            split_names.append(test_split)
        keep = config["experts"]["modalities"]
        text_feat = config["experts"]["text_feat"]
        for split_name in split_names:
            split_paths = set()
            root_feat, paths = dataset_paths(dataset_name, split_name, text_feat)
            modern_feat_agg = {key: val for key, val in feat_aggregation.items()
                               if key in paths["feature_names"]}
            feat_paths = model_specs2path(modern_feat_agg, keep)
            split_paths.update({root_feat / x for x in feat_paths})
            for key, feat_list in paths["custom_paths"].items():
                for feat_path in feat_list:
                    split_paths.add(root_feat / feat_path)
            split_paths.update(set(root_feat / x for x in
                               paths["subset_list_paths"].values()))
            if "text_feat_path" in paths:
                split_paths.add(root_feat / paths["text_feat_path"])
            else:
                text_paths = [root_feat / x for x in paths["text_feat_paths"].values()]
                split_paths.update(set(text_paths))
            split_paths.add(root_feat / paths["raw_captions_path"])
            if "dict_youtube_mapping_path" in paths:
                split_paths.add(root_feat / paths["dict_youtube_mapping_path"])
            all_feat_paths[dataset_name].update(split_paths)

    for dataset_name, paths in all_feat_paths.items():
        tar_include_list = Path("misc") / "datasets" / dataset_name / "tar_include.txt"
        tar_include_list.parent.mkdir(exist_ok=True, parents=True)
        with open(tar_include_list, "w") as f:
            for path in sorted(paths):
                print(f"Writing {path} to {tar_include_list}")
                f.write(f"{path}\n")


def parse_log(log_path):
    with open(log_path, "r") as f:
        log = f.read().splitlines()
    results = {}
    for group in {"t2v", "v2t"}:
        tag = f"[{group}] loaded log file"
        results[group] = OrderedDict()
        presence = [tag in row for row in log]
        assert sum(presence) == 1, "expected single occurence of log tag"
        metrics = ["R1", "R5", "R10", "R50", "MedR", "MeanR"]
        pos = np.where(presence)[0].item()
        if "fixed training length" in log[pos + 2]:
            pos += 3
        else:
            pos += 2
        rows = log[pos: pos + len(metrics)]
        for row, metric in zip(rows, metrics):
            row = row.replace("INFO:summary:", "")
            tokens = row.split(" ")
            if tokens[-3] != f"{metric}:":
                import ipdb; ipdb.set_trace()
            assert tokens[-3] == f"{metric}:", f"unexpected row format {row}"
            mean, std = float(tokens[-2].split(",")[0]), float(tokens[-1])
            results[group][metric] = (mean, std)
    for row in log:
        if "Trainable parameters" in row:
            results["params"] = int(row.split(" ")[-1])
    return results


def parse_results(experiments, save_dir, backup_save_dirs):
    log_results = {}
    for exp_name, meta in experiments.items():
        group_id, timestamp = meta
        if timestamp.startswith("TODO"):
            log_results[exp_name] = {"timestamp": "TODO", "results": {}}
            continue
        rel_fname = Path(timestamp) / "summary-seed-0_seed-1_seed-2.json"
        found = False
        for log_dir in ["log", "log-includes-some-final-exps"]:
            rel_path = Path(exp_name) / group_id / "seed-0" / rel_fname
            log_path = Path(save_dir) / log_dir / rel_path
            if log_path.exists():
                found = True
                break
        assert found, f"Could not find {log_path}"
        results = parse_log(log_path)
        log_results[exp_name] = {"timestamp": timestamp, "results": results}
    return log_results


def generate_results_string(target, exp_name, results, latexify, drop=None):
    stats = results[exp_name]["results"][target]
    print(f"Filling template values for {exp_name}")
    tokens = []
    prepad = False
    for metric, values in stats.items():
        mean, std = values
        if drop and metric in drop:
            continue
        print(f"{metric}: {mean} ({std})")
        if latexify:
            str_tokens = ["&$",  f"{mean}_{{\\pm{std}}}$"]
            if prepad:
                str_tokens.insert(1, "\prepad")
            tokens.append(" ".join(str_tokens))
        else:
            tokens += [f"{mean}<sub>({std})</sub>"]
    return small_font_str(tokens)


def generate_readme(experiments, readme_templates, root_url, readme_dests, results_path,
                    save_dir, backup_save_dirs, latexify, keep_mnr):

    results = parse_results(experiments, save_dir, backup_save_dirs)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, sort_keys=False)

    for readme_template, readme_dest in zip(readme_templates, readme_dests):
        with open(readme_template, "r") as f:
            readme = f.read().splitlines()

        # insert sub-templates
        full_readme = []
        for row in readme:
            regex = r"\<\<(.*?)\>\>"
            matched = False
            for match in re.finditer(regex, row):
                matched = True
                groups = match.groups()
                assert len(groups) == 1, "expected single group"
                subtemplate_path, src_name, dest_name = groups[0].split(":")
                with open(subtemplate_path, "r") as f:
                    subtemplate = f.read().splitlines()
                subrows = []
                for subrow in subtemplate:
                    drop_subrow = False
                    subrow = subrow.replace(src_name, dest_name)
                    subrow = subrow.replace(src_name.upper(), dest_name.upper())
                    # Handle the missing audio modalities of MSVD
                    if dest_name == "msvd":
                        for tag in ("speech", "audio"):
                            # drop experiments for which the audio/speech features form
                            # the control variable
                            if f"-{tag}." in subrow:
                                print("skipping", subrow)
                                drop_subrow = True
                                break
                            # remove audio features from other experiments
                            subrow = subrow.replace(f"-{tag}", "")

                    if not drop_subrow:
                        subrows.append(subrow)
                full_readme.extend(subrows)
            if not matched:
                full_readme.append(row)

        generated = []
        for row in full_readme:
            edits = []
            regex = r"\{\{(.*?)\}\}"
            for match in re.finditer(regex, row):
                groups = match.groups()
                assert len(groups) == 1, "expected single group"
                # if "latex" in groups[0]:
                #     token = generate_latex_results_string(groups[0], results)
                # else:
                exp_name, target = groups[0].split(".")
                try:
                    x = results[exp_name]["timestamp"] == "TODO"
                except:
                    import ipdb; ipdb.set_trace()
                if results[exp_name]["timestamp"] == "TODO":
                    token = "TODO"
                elif target in {"config", "model", "log"}:
                    token = generate_url(root_url, target, exp_name,
                                         experiments=experiments)
                    # token = small_font_str([token])
                elif target in {"t2v", "v2t"}:
                    token = generate_results_string(target, exp_name, results,
                                                    latexify=latexify)
                elif target in {"short-t2v", "short-v2t"}:
                    if keep_mnr:
                        drop = {"R50"}
                    else:
                        drop = {"R50", "MeanR"}
                    target_ = target.split("-")[1]
                    token = generate_results_string(target_, exp_name, results,
                                                    drop=drop, latexify=latexify)
                elif target in {"params"}:
                    token = millify(results[exp_name]["results"]["params"], precision=2)
                edits.append((match.span(), token))
            if edits:
                # invert the spans
                spans = [(None, 0)] + [x[0] for x in edits] + [(len(row), None)]
                inverse_spans = [(x[1], y[0]) for x, y in zip(spans, spans[1:])]
                tokens = [row[start:stop] for start, stop in inverse_spans]
                urls = [str(x[1]) for x in edits]
                new_row = ""
                for token, url in zip_longest(tokens, urls, fillvalue=""):
                    new_row += token + url
                row = new_row

            generated.append(row)

        with open(readme_dest, "w") as f:
            f.write("\n".join(generated))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="data/saved")
    parser.add_argument("--backup_save_dirs",
                        default=[
                            "/scratch/shared/slow/yangl/code/aaai/collaborative-experts/data/saved",
                            "/scratch/shared/slow/yangl/code/aaai/collaborative-experts/data/curren_no/saved",
                        ])
    parser.add_argument("--webserver", default="login.robots.ox.ac.uk")
    parser.add_argument("--results_path", default="misc/results.json")
    parser.add_argument("--experiments_path", default="misc/experiments.json")
    parser.add_argument("--readme_template", default="misc/README-template.md")
    parser.add_argument("--latexify", action="store_true")
    parser.add_argument("--keep_mnr", action="store_true")
    parser.add_argument("--readme_dest", default="README.md")
    parser.add_argument("--ablation_readme_dest", default="misc/ablations.md")
    parser.add_argument("--ablation_readme_template",
                        default="misc/ablations-template.md")
    parser.add_argument("--task", default="generate_readme",
                        choices=["gen_tar_lists", "sync_files", "generate_readme"])
    parser.add_argument(
        "--web_dir",
        default="/projects/vgg/vgg/WWW/research/collaborative-experts/data",
    )
    parser.add_argument(
        "--root_url",
        default="http://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data",
    )
    args = parser.parse_args()

    with open(args.experiments_path, "r") as f:
        experiments = json.load(f)

    if args.task == "gen_tar_lists":
        generate_tar_lists(
            save_dir=args.save_dir,
            experiments=experiments,
        )
    elif args.task == "sync_files":
        sync_files(
            web_dir=args.web_dir,
            save_dir=args.save_dir,
            webserver=args.webserver,
            experiments=experiments,
        )
    elif args.task == "generate_readme":
        readme_dests = [args.readme_dest, args.ablation_readme_dest]
        readme_templates = [args.readme_template, args.ablation_readme_template]
        generate_readme(
            root_url=args.root_url,
            save_dir=args.save_dir,
            latexify=args.latexify,
            experiments=experiments,
            keep_mnr=args.keep_mnr,
            readme_dests=readme_dests,
            results_path=args.results_path,
            readme_templates=readme_templates,
            backup_save_dirs=args.backup_save_dirs,
        )


if __name__ == "__main__":
    main()
