import argparse
import torch
import os
import numpy as np
import random
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils.util import compute_dims
from logger.log_parser import log_summary
import utils.visualizer as module_vis
from parse_config import ConfigParser
from trainer import Trainer
from test import evaluation


def main(config):
    logger = config.get_logger('train')
    expert_modality_dim, raw_input_dims = compute_dims(config)
    seeds = [int(x) for x in config._args.seeds.split(",")]

    for seed in seeds:
        # Set the random initial seeds
        logger.info(f"Setting experiment random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        data_loaders = config.init(
            name='data_loader',
            module=module_data,
            raw_input_dims=raw_input_dims,
            text_feat=config["experts"]["text_feat"],
            text_dim=config["experts"]["text_dim"],
        )

        model = config.init(
            name='arch',
            module=module_arch,
            expert_modality_dim=expert_modality_dim,
            text_dim=config["experts"]["text_dim"],
            disable_nan_checks=config["disable_nan_checks"],
        )
        logger.info(model)

        loss = config.init(name="loss", module=module_loss)
        metrics = [getattr(module_metric, met) for met in config['metrics']]
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        visualizer = config.init(
            name='visualizer',
            module=module_vis,
            exp_name=config._exper_name,
            log_dir=config._web_log_dir,
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
        )
        trainer.train()
        best_ckpt_path = config.save_dir / "trained_model.pth"

        # If the dataset supports separate validation/test splits, the training config
        # json should specify an `eval_config` entry with the path to the test
        # configuration
        if config._config.get("eval_config", False):
            args = argparse.ArgumentParser()
            args.add_argument("--config", default=config["eval_config"])
            args.add_argument("--device", default=config._args.device)
            args.add_argument("--resume", default=best_ckpt_path)
            eval_config = ConfigParser(args, ignore_argv=True)
            evaluation(eval_config)

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


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('--device', type=str, help="indices of GPUs to enable")
    args.add_argument('--mini_train', action="store_true")
    args.add_argument('--seeds', default="0", help="comma separated list of seeds")
    args.add_argument("--dbg", default="ipdb.set_trace")
    args = ConfigParser(args)
    os.environ["PYTHONBREAKPOINT"] = args._args.dbg

    msg = (f"Expected the number of training epochs ({args['trainer']['epochs']})"
           f"to exceed the save period ({args['trainer']['save_period']}), otherwise"
           " no checkpoints will be saved.")
    assert args["trainer"]["epochs"] >= args["trainer"]["save_period"], msg

    print("Launching experiment with config:")
    print(args)
    main(config=args)
