"""A small utility for filling in the README paths to experiment artifacts.

The template contains tags of the form {{filetype.experiment_name}}, which are then
replaced with the urls for each resource.

Note: print a summary of the most recent results for a given dataset via:
python find_latest_checkpoints.py --dataset lsmdc
"""
import argparse
import glob
import importlib
import json
import multiprocessing
import os
import pickle
import re
import subprocess
import time
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pylatex
import scipy.stats
import tqdm
from millify import millify
from typeguard import typechecked

from aggregate_logs_and_stats import summarise


@typechecked
def gen_latex_version_of_table(
        latex_table_dir: Path,
        content: List[str],
        table_name: str,
        branch_name: str = "dev",
) -> Path:
    msg = "Expected latexify tag to be placed directly following a table"
    assert content[-1].startswith("|"), msg
    num_table_rows = [x.startswith("|") for x in reversed(content)].index(False)
    assert num_table_rows > 2, "expected at least three table rows (including header)"
    markdown_table = list(reversed(content[-1:-(num_table_rows + 1):-1]))
    col_names = [x.strip() for x in markdown_table[0].split("|")[1:-1]]

    # remove last column of links
    remove_links = col_names[-1].lower() == "links"
    if remove_links:
        col_names.pop()
    cols = "|".join(["c" for _ in range(len(col_names))])
    table = pylatex.Tabular(cols)
    table.add_hline()
    table.add_hline()
    table.add_row(tuple(col_names))
    table.add_hline()
    for row in markdown_table[2:]:
        tokens = [x.strip() for x in row.split("|")[1:-1]]
        if remove_links:
            tokens.pop()
        row_contents = []
        for token in tokens:
            mean_regexp = r"<sub><sup>([0-9]+[.][0-9]+)<sub>"
            # std_regexp = r"<sub>\(([0-9]+[.][0-9]+|[a-z]+)\)<\/sub>"
            std_regexp = r"<sub>\(([0-9]+[.][0-9]+e*-*[0-9]*|[a-z]+|)\)<\/sub>"
            mean_strs = re.findall(mean_regexp, token)
            if mean_strs:
                assert len(mean_strs) == 1, "expected a unique mean specifier"
                std_strs = re.findall(std_regexp, token)
                assert len(std_strs) == 1, "expected a unique std specifier"
                mean_str, std_str = mean_strs[0], std_strs[0]
                raw_str = "$" + mean_str + r"_{\pm" + std_str + r"}$"
                token = pylatex.NoEscape(raw_str)
            row_contents.append(token)
        table.add_row(tuple(row_contents))
        table.add_hline()
    latex_str = table.dumps()
    latex_table_dir.mkdir(exist_ok=True, parents=True)
    dest_path = latex_table_dir / f"{table_name}.txt"
    with open(dest_path, "w") as f:
        f.write(latex_str)
    github_project_root = f"/../../tree/{branch_name}/"
    markdown_link = Path(f"{github_project_root}{dest_path}")
    return markdown_link


@typechecked
def generate_url(root_url: str, target: str,
                 exp_name: str, experiments: Dict,
                 fnames: dict, seed_folders: dict) -> str:
    path_store = {
        "log": {"parent": "log", "fname": fnames[exp_name]},
        "log_TT": {"parent": "log", "fname": fnames[exp_name]},
        "config": {"parent": "models", "fname": "config.json"},
        "config_TT": {"parent": "models", "fname": "config.json"},
        "model": {"parent": "models", "fname": "trained_model.pth"},
        "model_TT": {"parent": "models", "fname": "trained_model.pth"}
    }
    paths = path_store[target]
    group_id, timestamp = experiments[exp_name]
    rel_path = Path(group_id) / seed_folders[exp_name] / timestamp / paths["fname"]
    return str(Path(root_url) / paths["parent"] / exp_name / rel_path)


def small_font_str(tokens):
    tokens = [f"<sub><sup>{x}</sup></sub>" for x in tokens]
    return " | ".join(tokens)


def sync_files(experiments, save_dir, webserver, web_dir):
    filetypes = {
        "log": ["summary-seed-1_seed-2_seed-3.json"],
        "log_TT": ["summary-seed-1_seed-2_seed-3.json"],
        "models": ["trained_model.pth", "config.json"],
        "models_TT": ["trained_model.pth", "config.json"]
    }
    for key, (group_id, timestamp) in experiments.items():
        # copy experiment artifacts
        for filetype, fnames in filetypes.items():
            for fname in fnames:
                if timestamp.startswith("TODO"):
                    continue
                rel_path = Path(group_id) / "seed-1" / timestamp / fname
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


@typechecked
def model_specs2path(feat_aggregation: Dict, keep: set, tag: str = None) -> List[Path]:
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

        for option in "offset", "inner_stride", "num_segments":
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


@typechecked
def dataset_paths(
        dataset: str
) -> Tuple[Path, Dict[str, Union[str, List[str], Dict, Path]]]:
    name_map = {
        "msvd": "MSVD",
        "lsmdc": "LSMDC",
        "msrvtt": "MSRVTT",
        "didemo": "DiDeMo",
        "activity-net": "ActivityNet",
        "youcook2": "YouCook2",
        "queryd": "QuerYD",
        "querydsegments": "QuerYDSegments"
    }
    if dataset in set(name_map.values()):
        class_name = dataset
    else:
        class_name = name_map[dataset]
    mod = importlib.import_module(f"data_loader.{class_name}_dataset")
    get_dataset_paths = getattr(getattr(mod, class_name), "dataset_paths")
    if dataset == "activity-net":
        data_dir = dataset
    else:
        data_dir = class_name
    root_feat = Path(f"data/{data_dir}/structured-symlinks")
    paths = get_dataset_paths()
    return root_feat, paths


def generate_tar_lists(save_dir, experiments):
    all_feat_paths = {}
    for exp_name, (group_id, timestamp) in tqdm.tqdm(experiments.items()):
        rel_path = Path(group_id) / "seed-1" / timestamp / "config.json"
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
        keep = set(config["experts"]["modalities"])
        text_feat = config["experts"]["text_feat"]
        root_feat, paths = dataset_paths(dataset_name)
        modern_feat_agg = {key: val for key, val in feat_aggregation.items()
                           if key in paths["feature_names"]}
        feat_paths = model_specs2path(modern_feat_agg, keep)
        all_feat_paths[dataset_name].update({root_feat / x for x in feat_paths})
        for key, feat_list in paths["custom_paths"].items():
            for feat_path in feat_list:
                all_feat_paths[dataset_name].add(root_feat / feat_path)
        text_paths = [root_feat / rel_path for rel_path in
                      paths["text_feat_paths"][text_feat].values()]
        all_feat_paths[dataset_name].update(set(text_paths))
        all_feat_paths[dataset_name].add(root_feat / paths["raw_captions_path"])
        if "dict_youtube_mapping_path" in paths:
            all_feat_paths[dataset_name].add(
                root_feat / paths["dict_youtube_mapping_path"])
        for split_name in split_names:
            split_paths = set(root_feat / x for x in
                              paths["subset_list_paths"][split_name].values())
            all_feat_paths[dataset_name].update(split_paths)

    for dataset_name, paths in all_feat_paths.items():
        tar_include_list = Path("misc") / "datasets" / dataset_name / "tar_include.txt"
        tar_include_list.parent.mkdir(exist_ok=True, parents=True)
        with open(tar_include_list, "w") as f:
            for path in sorted(paths):
                print(f"Writing {path} to {tar_include_list}")
                f.write(f"{path}\n")

@typechecked
def parse_geom_means_from_val_runs(log: List[str], group: str) -> List[float]:
    """TODO: Samuel - this is redundant due to log_summary() func in log_parser
    should refactor after deadline.
    """
    subset = "val"
    # sanity check, should not be used for experiments with test sets
    assert sum(["test_t2v" in x for x in log]) == 0, "should not parse test runs"
    scores = {
        "R1": defaultdict(list),
        "R5": defaultdict(list),
        "R10": defaultdict(list),
    }
    # Regex tag for finding the seed
    seed_tag = "Setting experiment random seed to"

    for row in log:
        if seed_tag in row:
            # Search for the log file entry describing the current random seed
            match = re.search(seed_tag + " (\d+)$", row)  # NOQA
            assert len(match.groups()) == 1, "expected a single regex match"
            current_seed = match.groups()[0]

        if f"{subset}_{group}_metrics" in row:
            tokens = row.split(" ")
            for key in scores:
                tag = f"{subset}_{group}_metrics_{key}:"
                if tag in tokens:
                    pos = tokens.index(tag) + 1
                    val = tokens[pos]
                    val = float(val)
                    assert current_seed is not None, "failed to determine the seed"
                    scores[key][current_seed].append(val)
    # keep last score
    agg_scores = {key: [] for key in scores}
    for metric, subdict in scores.items():
        for seed, values in subdict.items():
            agg_scores[metric].append(values[-1])
    geometric_means = []
    for r1, r5, r10 in zip(agg_scores["R1"], agg_scores["R5"], agg_scores["R10"]):
        geometric_means.append(scipy.stats.mstats.gmean([r1, r5, r10]))
    return geometric_means


def parse_log(log_path):
    with open(log_path, "r") as f:
        log = f.read().splitlines()
    results = {}
    for group in {"t2v", "v2t"}:
        tag = f"[{group}] loaded log file"
        results[group] = OrderedDict()
        presence = [tag in row for row in log]
        msg = f"expected single occurence of log tag, found {sum(presence)} in {log_path}"
        assert sum(presence) == 1, msg
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
                raise ValueError(f"Unexpteced log format [{row}]")
            assert tokens[-3] == f"{metric}:", f"unexpected row format {row}"
            mean, std = float(tokens[-2].split(",")[0]), float(tokens[-1])
            results[group][metric] = (mean, std)
        # geometric means are recomputed from summaries
        tag = f"test_{group}_metrics_geometric_mean"
        nan_tag = "INFO:summary:R1: nan"
        matches = [x for x in log if tag in x]
        if len(matches) in {1, 2, 3}:
            geoms = [float(x.split()[-1].replace("INFO:summary:", "")) for x in matches]
            if len(matches) < 3:
                print(f"WARNING: Getting stds from {len(matches)} runs for {log_path}!")
        elif sum([nan_tag in x for x in log]) > 0:
            geoms = [np.nan, np.nan, np.nan]
        else:
            valid_exceptions = ["miechfeats-moee", "miech-ce", "jsfusion"]
            msg = f"Did not expect fixed length training for {log_path}"
            assert any([x in str(log_path) for x in valid_exceptions]), msg
            geoms = parse_geom_means_from_val_runs(log, group=group)
        if len(geoms) == 1:
            std = np.nan
        else:
            std = np.std(geoms)
        results[group]["geom"] = (round(np.mean(geoms),1), round(std, 1))
    for row in log:
        if "Trainable parameters" in row:
            param_token = row.split(" ")[-1].replace("INFO:summary:", "")
            results["params"] = int(param_token)
    return results

@typechecked
def multiprocessing_parsing(exp_name: str, meta: list,
        save_dir: Path, refresh_summaries: bool, teachText: bool, pickle_files: str):
    if os.path.exists(Path(save_dir) / pickle_files / f'log_results_{exp_name}.pkl') is False:
        group_id, timestamp = meta
        _log_path = "log_old"
        if teachText:
            _log_path = "log"
        if timestamp.startswith("TODO"):
            log_results[exp_name] = {"timestamp": "TODO", "results": {}}
        else:
            seed_folder = sorted(os.listdir(Path(save_dir) / _log_path / Path(exp_name) / group_id))[0]
            files_in_seed_folder = os.listdir(Path(save_dir) / _log_path / Path(exp_name) / group_id / seed_folder /  Path(timestamp))
            for file in files_in_seed_folder:
                if ".json" in file and ".bak" not in file:
                    fname = file
                    break
            rel_fname = Path(timestamp) / fname
            rel_path = Path(exp_name) / group_id / seed_folder / rel_fname
            log_path = Path(save_dir) / _log_path / rel_path
            if refresh_summaries:
                summarise(group_id=group_id, log_dir=Path(save_dir) / _log_path)
            results = parse_log(log_path)
            log_results = {"timestamp": timestamp, "results": results}
        with open(Path(save_dir) / pickle_files / f'log_results_{exp_name}.pkl', 'wb') as f:
            pickle.dump([log_results, fname, seed_folder], f)
            print(f"Saved experiment {exp_name}")
    else:
        print(f"Experiment log_results_{exp_name}.pkl already saved")

@typechecked
def parse_results(
        experiments: Dict[str, List[str]],
        save_dir: Path,
        refresh_summaries: bool,
        teachText: bool,
) -> (Dict[str, Dict[str, Union[str, Dict]]],
      dict, dict):
    starttime = time.time()
    processes = []
    experiments_items = experiments.items()
    pickle_files = "pickle_files"
    if teachText:
        pickle_files = "pickle_files_teachText"
    if os.path.exists(Path(save_dir) / pickle_files) is False:
        os.mkdir(Path(save_dir) / pickle_files)
    for exp_name, meta in experiments_items:
        p = multiprocessing.Process(target=multiprocessing_parsing,
                                    args=(exp_name, meta,
                                          save_dir, refresh_summaries, teachText, pickle_files))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    print('That took {} seconds'.format(time.time() - starttime))
    log_results = {}
    fnames = {}
    seed_folders = {}
    for exp_name, _ in experiments_items:
        with open(Path(save_dir) / pickle_files / f'log_results_{exp_name}.pkl',
                  'rb') as f:
            log_results[exp_name],\
                fnames[exp_name],\
                    seed_folders[exp_name] = pickle.load(f)
        if not teachText:
            with open(Path(save_dir) / 'log_results2.pkl', 'wb') as f:
                pickle.dump([log_results, fnames, seed_folders], f)
        else:
            with open(Path(save_dir) / 'log_results_teachText.pkl', 'wb') as f:
                pickle.dump([log_results, fnames, seed_folders], f)
    


    return log_results, fnames, seed_folders


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
            str_tokens = ["&$", f"{mean}_{{\\pm{std}}}$"]
            if prepad:
                str_tokens.insert(1, r"\prepad")
            tokens.append(" ".join(str_tokens))
        else:
            tokens += [f"{mean}<sub>({std})</sub>"]
    return small_font_str(tokens)


def generate_readme(
    experiments: Dict[str, List[str]],
    root_url: str,
    readme_templates: List[Path],
    readme_dests: List[Path],
    results_path: Path,
    latex_table_dir: Path,
    save_dir: Path,
    latexify: bool,
    keep_mnr: bool,
    refresh_summaries: bool,
    results: Dict,
    fnames: Dict,
    seed_folders: Dict,
    append_to_existing_readme: bool,
):
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
                exp_name, target = groups[0].split(".")
                if target.startswith("latexify"):
                    latex_link = gen_latex_version_of_table(
                        content=generated[:],
                        table_name=exp_name,
                        latex_table_dir=latex_table_dir,
                    )
                    token = f"[latex]({latex_link}) | | | | | | | |"
                elif results[exp_name]["timestamp"] == "TODO":
                    token = "TODO"
                elif target in {"config", "model", "log"}:
                    token = generate_url(root_url, target, exp_name,
                                         experiments=experiments,
                                         fnames=fnames,
                                         seed_folders=seed_folders)
                elif target in {"config_TT", "model_TT", "log_TT"}:
                    token = generate_url("http://www.robots.ox.ac.uk/~vgg/research/teachtext/data-hq", target, exp_name,
                                         experiments=experiments,
                                         fnames=fnames,
                                         seed_folders=seed_folders)

                elif target in {"t2v", "v2t", "geomt2v", "geomv2t"}:
                    if not "geom" in target:
                        drop = {"geom"}
                    else:
                        drop = {}
                    target_ = target.split("geom")[-1]
                    token = generate_results_string(target_, exp_name, results,
                                                    drop=drop, latexify=latexify)
                elif target in {"short-t2v", "short-v2t"}:
                    if keep_mnr:
                        drop = {"R50", "geom"}
                    else:
                        drop = {"R50", "MeanR", "geom"}
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

        if not append_to_existing_readme:
            with open(readme_dest, "w") as f:
                f.write("\n".join(generated))
        else:
            with open(readme_dest, "a") as f:
                f.write("\n".join(generated))



def parse_generate_readme(
    experiments: Dict[str, List[str]],
    root_url: str,
    readme_templates: List[Path],
    readme_dests: List[Path],
    results_path: Path,
    latex_table_dir: Path,
    save_dir: Path,
    latexify: bool,
    keep_mnr: bool,
    refresh_summaries: bool,
    drop_experiments_hq: bool,
    results_path_teachText: Path,
    experiments_teachText: Dict[str, List[str]],
    teachText_template: Path,
):

    results, fnames, seed_folders = parse_results(experiments=experiments,
                                                  save_dir=save_dir,
                                                  refresh_summaries=refresh_summaries,
                                                  teachText=False,
                                                 )
   
    append_to_existing_readme=False
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, sort_keys=False)

    if not drop_experiments_hq:
        results_teachText, fnames_teachText, seed_folders_teachText = parse_results(experiments=experiments_teachText,
                                                  save_dir=save_dir,
                                                  refresh_summaries=refresh_summaries,
                                                  teachText=True,
                                                 )
        with open(results_path_teachText, "w") as f:
            json.dump(results, f, indent=4, sort_keys=False)

        generate_readme(experiments=experiments_teachText,
            root_url=root_url,
            readme_templates=[teachText_template],
            readme_dests=readme_dests,
            results_path=results_path_teachText,
            latex_table_dir=latex_table_dir,
            save_dir=save_dir,
            latexify=latexify,
            keep_mnr=keep_mnr,
            refresh_summaries=refresh_summaries,
            results=results_teachText,
            fnames=fnames_teachText,
            seed_folders=seed_folders_teachText,
            append_to_existing_readme=False,
            )

        append_to_existing_readme=True
    
    generate_readme(experiments=experiments,
            root_url=root_url,
            readme_templates=readme_templates,
            readme_dests=readme_dests,
            results_path=results_path,
            latex_table_dir=latex_table_dir,
            save_dir=save_dir,
            latexify=latexify,
            keep_mnr=keep_mnr,
            refresh_summaries=refresh_summaries,
            results=results,
            fnames=fnames,
            seed_folders=seed_folders,
            append_to_existing_readme=append_to_existing_readme,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="data/saved", type=Path)
    parser.add_argument("--webserver", default="login.robots.ox.ac.uk")
    parser.add_argument("--results_path", default="misc/results.json", type=Path)
    parser.add_argument("--results_path_teachText", default="misc/results_teachText.json", type=Path)
    parser.add_argument("--experiments_path", default="misc/experiments.json")
    parser.add_argument("--experiments_teachText", default="misc/experiments_teachText.json")
    parser.add_argument("--readme_template", default="misc/README-template.md")
    parser.add_argument("--teachText_template", default="misc/README-teachText.md")
    parser.add_argument("--latexify", action="store_true")
    parser.add_argument("--drop_experiments_hq", action="store_true")
    parser.add_argument("--keep_mnr", action="store_true")
    parser.add_argument("--refresh_summaries", action="store_true")
    parser.add_argument("--readme_dest", default="README.md")
    parser.add_argument("--latex_table_dir", default="latex-tables", type=Path)
    parser.add_argument("--ablation_readme_dest", default="misc/ablations.md")
    parser.add_argument("--challenge_readme_dest", default="misc/challenge.md")
    parser.add_argument("--ablation_readme_template",
                        default="misc/ablations-template.md")
    parser.add_argument("--challenge_readme_template",
                        default="misc/README-challenge-template.md")
    parser.add_argument("--task", default="generate_readme",
                        choices=["sync_files", "generate_readme"])
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

    with open(args.experiments_teachText, 'r') as f:
        experiments_teachText = json.load(f)

    if args.task == "sync_files":
        sync_files(
            web_dir=args.web_dir,
            save_dir=args.save_dir,
            webserver=args.webserver,
            experiments=experiments,
        )
    elif args.task == "generate_readme":
        readme_dests = [
            args.readme_dest,
            args.ablation_readme_dest,
            args.challenge_readme_dest,
        ]
        readme_templates = [
            args.readme_template,
            args.ablation_readme_template,
            args.challenge_readme_template,
        ]
        parse_generate_readme(
            root_url=args.root_url,
            save_dir=args.save_dir,
            latexify=args.latexify,
            experiments=experiments,
            latex_table_dir=args.latex_table_dir,
            keep_mnr=args.keep_mnr,
            readme_dests=readme_dests,
            results_path=args.results_path,
            readme_templates=readme_templates,
            refresh_summaries=args.refresh_summaries,
            drop_experiments_hq=args.drop_experiments_hq,
            results_path_teachText=args.results_path_teachText,
            experiments_teachText=experiments_teachText,
            teachText_template=args.teachText_template,
        )


if __name__ == "__main__":
    main()
