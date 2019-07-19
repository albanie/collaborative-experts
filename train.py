import argparse
import torch
import os
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils.util import compute_dims
import utils.visualizer as module_vis
from parse_config import ConfigParser
from trainer import Trainer


def main(config):
    logger = config.get_logger('train')
    expert_modality_dim, raw_input_dims = compute_dims(config)

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
        mini_train=config["mini_train"],
        disable_nan_checks=config["disable_nan_checks"],
        visualizer=visualizer,
    )
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('--device', type=str, help="indices of GPUs to enable")
    args.add_argument('--mini_train', action="store_true")
    args.add_argument("--dbg", default="ipdb.set_trace")
    args = ConfigParser(args)
    os.environ["PYTHONBREAKPOINT"] = args._args.dbg
    main(config=args)
