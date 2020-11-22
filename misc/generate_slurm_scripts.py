"""A small utility for generating slurm scripts.

The slurm template contains tags of the form {{variable-name}}, which are then
replaced with values for submission.

Examples:
EXP_LIST="seeded-exps.txt"
EXP_LIST="msrvtt-ablations.txt"
EXP_LIST="lsmdc-ablations.txt"
EXP_LIST="activity-net-ablations.txt"
EXP_LIST="didemo-ablations.txt"

EXP_LIST="msvd-ablations.txt"

EXP_LIST="msrvtt-text-study.txt"
python misc/generate_slurm_scripts.py --job_queue "slurm/${EXP_LIST}" \
                               && source data/slurm/scripts/slurm-dependencies.sh
"""
import re
import uuid
import copy
import argparse
import itertools
from pathlib import Path
from utils.util import parse_grid
from itertools import zip_longest
from collections import OrderedDict



def fill_template(template_path, rules):
    generated = []
    with open(template_path, "r") as f:
        template = f.read().splitlines()
    for row in template:
        edits = []
        regex = r"\{\{(.*?)\}\}"
        for match in re.finditer(regex, row):
            groups = match.groups()
            assert len(groups) == 1, "expected single group"
            key = groups[0]
            token = rules[key]
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
    return "\n".join(generated)


def parse_group_ids(parsed_cmds):
    group_ids = OrderedDict()
    for ii, cmd in enumerate(parsed_cmds):
        tokens = cmd.split(" ")
        group_id = tokens[tokens.index("--group_id") + 1]
        if group_id not in group_ids:
            group_ids[group_id] = []
        group_ids[group_id].append(ii + 1)  # slurm arrays are 1-indexed
    return group_ids


def generate_slurm_dependency_script(group_ids, dependency_template, aggregation_scripts,
                                     generated_script_paths):
    deps = []
    for group_id, aggregation_script in aggregation_scripts.items():
        array_id_list = group_ids[group_id]
        array_deps = ":".join([f"${{job_id}}_{x}" for x in array_id_list])
        dep = f"sbatch --dependency=afterok:{array_deps} {aggregation_script}"
        deps.append(dep)
    rules = {
        "dependencies": "\n".join(deps),
        "job_script_path": str(generated_script_paths["array-job"]),
    }
    return fill_template(template_path=dependency_template, rules=rules)


def jobn_name2agg_log_path(exp_dir, job_name):
    return Path(exp_dir) / "data/slurm" / job_name / "log.txt"


def generate_aggregation_script(exp_dir, group_id, aggregation_template,
                                aggregation_script_path):
    job_name = aggregation_script_path2job_name(aggregation_script_path)
    log_path = jobn_name2agg_log_path(exp_dir, job_name)
    log_path.parent.mkdir(exist_ok=True, parents=True)
    rules = {"job-name": job_name, "group_id": group_id, "log-path": log_path}
    return fill_template(template_path=aggregation_template, rules=rules)


def aggregation_script_path2job_name(aggregation_script_path):
    job_name = f"{aggregation_script_path.parent.stem}-{aggregation_script_path.stem}"
    return job_name

def generate_script(template_path, slurm_script_dir, job_queue, exp_dir,
                    monitor_script, constraints, dependency_template,
                    aggregation_template):

    with open(job_queue, "r") as f:
        custom_args = f.read().splitlines()
    # remove blank lines
    custom_args = [x for x in custom_args if x]
    parsed = []
    for line in custom_args:
        parsed.extend(parse_grid(line))
    num_array_workers = len(parsed)

    if Path(job_queue).stem != "latest":
        array_job_name = Path(job_queue).stem
    else:
        config = parsed[0].split(" ")[1]
        array_job_name = config.replace("/", "-")

    generated_script_paths = {
        "main": "slurm-dependencies.sh",
        "array-job": "slurm-job.sh",
        "backup": f"{array_job_name}.sh",
    }
    group_ids = parse_group_ids(parsed)
    generated_script_paths = {key: Path(slurm_script_dir) / val
                              for key, val in generated_script_paths.items()}

    aggregation_scripts = {}
    for group_id, arg_list in zip(group_ids, custom_args):
        arg_list = arg_list.replace("--", "").replace(" ", "_").replace(".json", "")
        arg_list = arg_list.replace("|", "_")
        fname = f"{array_job_name}-{arg_list}_agg_{group_id}.sh"
        path = Path(slurm_script_dir) / fname
        aggregation_scripts[group_id] = path
    generated_script_paths.update(aggregation_scripts)

    # worker logs
    array_log_path = Path(exp_dir) / "data/slurm" / array_job_name / "%4a-log.txt"
    array_log_path.parent.mkdir(exist_ok=True, parents=True)
    watched_logs = {"paths": [], "dividers": []}

    for idx in range(num_array_workers):
        slurm_id = idx + 1
        watched_log = Path(str(array_log_path).replace("%4a", f"{slurm_id:04d}"))
        msg = f">>  START OF NEW JOB [{idx}/{num_array_workers}] <<\n"
        watched_logs["paths"].append(watched_log)
        watched_logs["dividers"].append(msg)

    for aggregation_script_path in aggregation_scripts.values():
        job_name = aggregation_script_path2job_name(aggregation_script_path)
        watched_log = jobn_name2agg_log_path(exp_dir, job_name)
        watched_logs["paths"].append(watched_log)
        watched_logs["dividers"].append(f">>  STARTING AGGREGATION job [{job_name}] <<\n")

    for watched_log, divider in zip(watched_logs["paths"], watched_logs["dividers"]):
        watched_log.parent.mkdir(exist_ok=True, parents=True)
        if not watched_log.exists():
            print(f"Creating watch log: {watched_log} for the first time")
            watched_log.touch()
        else:
            with open(str(watched_log), "a") as f:
                f.write(divider)

    with open(monitor_script, "w") as f:
        cmd = f"watchlogs {','.join([str(x) for x in watched_logs['paths']])}"
        f.write(f"{cmd}\n")
    print(f"Watching logs: {','.join(watched_logs)}")
    for script_name, dest_path in generated_script_paths.items():
        dest_path.parent.mkdir(exist_ok=True, parents=True)
        if script_name in {"array-job", "backup"}:
            rules = {
                "job-name": array_job_name,
                "job_queue": " ".join([f'"{x}"' for x in parsed]),
                "constraints": constraints,
                "array-range": f"1-{num_array_workers}",
                "log-path": str(array_log_path),

            }
            script = fill_template(template_path, rules)
        elif script_name in aggregation_scripts:
            script = generate_aggregation_script(
                exp_dir=exp_dir,
                group_id=script_name,
                aggregation_script_path=dest_path,
                aggregation_template=aggregation_template,
            )
        elif script_name == "main":
            script = generate_slurm_dependency_script(
                group_ids=group_ids,
                generated_script_paths=generated_script_paths,
                aggregation_scripts=aggregation_scripts,
                dependency_template=dependency_template,
            )
        with open(str(dest_path), "w") as f:
            print(f"Writing slurm script ({script_name}) to {dest_path}")
            f.write(script)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_queue", default="data/job-queues/latest.txt")
    parser.add_argument("--slurm_script_dir", default="data/slurm/scripts")
    parser.add_argument("--slurm_template", default="misc/slurm/gpu-template_v2.sh")
    parser.add_argument("--dependency_template", default="misc/slurm/dependencies.sh")
    parser.add_argument("--aggregation_template",
                        default="misc/slurm/aggregate-logs-and-stats.sh")
    parser.add_argument("--constraints", default="")
    parser.add_argument("--exp_dir",
                        #default="/users/albanie/coding/libs/pt/collaborative-experts")
                        # default="/users/ioana/collaborative-experts-internal/collaborative-experts-internal")
                        default="/scratch/shared/beegfs/oncescu/shared-datasets/QuerYD/collaborative")
    args = parser.parse_args()

    monitor_script = f"slurm/monitor-jobs.sh"
    generate_script(
        exp_dir=args.exp_dir,
        job_queue=args.job_queue,
        monitor_script=monitor_script,
        template_path=args.slurm_template,
        slurm_script_dir=args.slurm_script_dir,
        dependency_template=args.dependency_template,
        aggregation_template=args.aggregation_template,
        constraints=args.constraints,
    )


if __name__ == "__main__":
    main()
