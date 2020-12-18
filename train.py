"""
%run -i train.py --config configs/data_loader_lsmdc.json --device 0
"""
import os
import copy
import time
import random
import socket
import argparse
import warnings
from test import evaluation
from pathlib import Path

import numpy as np
import swats
import torch
import torch.nn as nn
from mergedeep import Strategy, merge

import model.loss as module_loss
import model as module_arch
import model.metric as module_metric
import utils.visualizer as module_vis
import data_loader.data_loaders as module_data
from utils import radam, ranger, set_seeds, cos_restart
from trainer import Trainer
from utils.util import compute_dims, compute_trn_config, update_src_web_video_dir
from parse_config import ConfigParser
from logger.log_parser import log_summary
import math

def run_exp(config):
    warnings.filterwarnings('ignore')
    logger = config.get_logger('train')

    leaderboard_path = config._args.leaderboard
    Path(leaderboard_path).parent.mkdir(exist_ok=True, parents=True)
    with open(leaderboard_path, 'a') as f:
        txt_path = f"{config._log_dir}/preds.txt"
        print(txt_path, file=f, flush=True)

    expert_dims, raw_input_dims, text_dim = compute_dims(config, logger)
    trn_config = compute_trn_config(config)

    if config._args.group_seed:
        seeds = [int(config._args.group_seed)]
    else:
        seeds = [int(x) for x in config._args.seeds.split(",")]

    # set up local filesystem on the cluster
    if socket.gethostname().endswith("cluster"):
        os.system(str(Path.home() / "configure_tmp_data.sh"))

    for ii, seed in enumerate(seeds):
        tic = time.time()
        logger.info(f"{ii + 1}/{len(seeds)} Setting experiment random seed to {seed}")
        set_seeds(seed)
        config["seed"] = seed

        model = config.init(
            name='arch',
            module=module_arch,
            expert_dims=expert_dims,
            text_dim=text_dim,
            disable_nan_checks=config["disable_nan_checks"],
            spatial_feats=config["data_loader"]["args"].get("spatial_feats", False),
            task=config.get("task", "retrieval"),
            ce_shared_dim=config["experts"].get("ce_shared_dim", None),
            feat_aggregation=config["data_loader"]["args"]["feat_aggregation"],
            trn_config=trn_config,
            trn_cat=config["data_loader"]["args"].get("trn_cat", 0),
        )
        logger.info(model)

        data_loaders = config.init(
            name='data_loader',
            module=module_data,
            logger=logger,
            raw_input_dims=raw_input_dims,
            challenge_mode=config.get("challenge_mode", False),
            text_dim=text_dim,
            text_feat=config["experts"]["text_feat"],
            text_agg=config["experts"]["text_agg"],
            use_zeros_for_missing=config["experts"].get("use_zeros_for_missing", False),
            task=config.get("task", "retrieval"),
            eval_only=False,
            distil_params=config.get("distil_params", None),
            training_file=config.get("training_file", None),
            caption_masks=config.get("caption_masks", None),
            ce_shared_dim=config["experts"].get("ce_shared_dim", None),
        )

        if config.get("manual_linear_init", False):
            logger.info("manually setting init for linear layers")

            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform(m.weight)
                    m.bias.data.fill_(0.01)
            model.apply(init_weights)

        loss = config.init(name="loss", module=module_loss)
        metrics = [getattr(module_metric, met) for met in config['metrics']]
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

        if config["optimizer"]["type"] == "RAdam":
            optimizer = config.init('optimizer', radam, trainable_params)
        elif config["optimizer"]["type"] == "Ranger":
            optimizer = config.init('optimizer', ranger, trainable_params)
        elif config["optimizer"]["type"] == "SWATS":
            optimizer = config.init('optimizer', swats, trainable_params)
        else:
            optimizer = config.init('optimizer', torch.optim, trainable_params)

        if config["lr_scheduler"]["type"] == "StepLR":
            lr_scheduler = config.init('lr_scheduler', torch.optim.lr_scheduler,
                                       optimizer)
        else:
            lr_scheduler = config.init('lr_scheduler', cos_restart, optimizer)

        update_src_web_video_dir(config)
        visualizer = config.init(
            name='visualizer',
            module=module_vis,
            exp_name=config._exper_name,
            web_dir=config._web_log_dir,
        )

        trainer = Trainer(
            model,
            loss,
            metrics,
            optimizer,
            config=config,
            data_loaders=data_loaders,
            lr_scheduler=lr_scheduler,
            mini_train=config._args.mini_train,
            disable_nan_checks=config["disable_nan_checks"],
            visualizer=visualizer,
            val_freq=config["trainer"].get("val_freq", 1),
            distil_loss=config.get("distil_loss", False),
            distil_params=config.get("distil_params", None),
            force_cpu_val=config.get("force_cpu_val", False),
            skip_first_n_saves=config["trainer"].get("skip_first_n_saves", 0),
            include_optim_in_ckpts=config["trainer"].get("include_optim_in_ckpts", 1),
            cache_targets=set(config.get("cache_targets", [])),
        )
        trainer.train()
        best_ckpt_path = config.save_dir / "trained_model.pth"
        duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - tic))
        logger.info(f"Training took {duration}")

        if config._config.get("eval_settings", False):
            eval_config = copy.deepcopy(config)
            merge(eval_config._config, config["eval_settings"], strategy=Strategy.REPLACE)
            eval_config._args.resume = best_ckpt_path
            evaluation(eval_config, logger=logger, trainer=trainer)

    # If multiple runs were conducted, report relevant statistics
    if len(seeds) > 1:
        log_summary(
            logger=logger,
            log_path=config.log_path,
            eval_mode=config["eval_mode"],
            fixed_num_epochs=config["trainer"]["epochs"],
        )
    print(f"Log file stored at {config.log_path}")

    # Report the location of the "best" checkpoint of the final seeded run (here
    # "best" corresponds to the model with the highest geometric mean over the
    # R@1, R@5 and R@10 metrics when a validation set is used, or simply the final
    # epoch of training for fixed-length schedules).
    print(f"The best performing ckpt can be found at {str(best_ckpt_path)}")


def main():
    args = argparse.ArgumentParser(description='Main entry point for training')
    args.add_argument('--config', help='config file path')
    args.add_argument('--resume', help='path to latest checkpoint (default: None)')
    args.add_argument('--leaderboard', default="data/leaderboards/exp.txt",
                      help='path we want to draw on leadboard')
    args.add_argument('--device', help="indices of GPUs to enable")
    args.add_argument('--mini_train', action="store_true")
    args.add_argument('--group_id', help="if supplied, group these experiments")
    args.add_argument('--disable_workers', action="store_true")
    args.add_argument('--refresh_lru_cache', action="store_true")
    args.add_argument('--train_single_epoch', action="store_true")
    args.add_argument('--purge_exp_dir', action="store_true",
                      help="remove all previous experiments with the given config")
    args.add_argument("--dbg", default="ipdb.set_trace")
    args.add_argument("--custom_args", help="qualified key,val pairs")

    # Seeds can either be passed directly as a comma separated list at the command line,
    # or individually for separate experiments as a group (used for slurm experiments)
    seed_args = args.add_mutually_exclusive_group()
    seed_args.add_argument('--seeds', default="0", help="comma separated list of seeds")
    seed_args.add_argument('--group_seed', help="seed for group member")
    args = ConfigParser(args)
    os.environ["PYTHONBREAKPOINT"] = args._args.dbg
    args["data_loader"]["args"]["refresh_lru_cache"] = args._args.refresh_lru_cache
    msg = (f"Expected the number of training epochs ({args['trainer']['epochs']})"
           f"to exceed the save period ({args['trainer']['save_period']}), otherwise"
           " no checkpoints will be saved.")
    assert args["trainer"]["epochs"] >= args["trainer"]["save_period"], msg
    print("Launching experiment with config:")
    print(args)
    run_exp(config=args)


if __name__ == '__main__':
    main()
