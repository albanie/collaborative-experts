import argparse
import torch
import numpy as np
import copy
import random
import data_loader.data_loaders as module_data
import utils.visualizer as module_vis
import model.metric as module_metric
import model.model as module_arch
from trainer import ctxt_mgr, verbose
from parse_config import ConfigParser
from utils.util import compute_dims, compute_trn_config
from mergedeep import merge, Strategy


def evaluation(config, logger=None, trainer=None):

    if logger is None:
        logger = config.get_logger('test')

    if getattr(config._args, "eval_from_training_config", False):
        eval_conf = copy.deepcopy(config)
        merge(eval_conf._config, config["eval_settings"], strategy=Strategy.REPLACE)
        config = eval_conf

    logger.info("Running evaluation with configuration:")
    logger.info(config)

    expert_dims, raw_input_dims = compute_dims(config)
    trn_config = compute_trn_config(config)

    # Set the random initial seeds
    seed = config["seed"]
    logger.info(f"Setting experiment random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # We use cls defaults for backwards compatibility with the MMIT configs.  In the
    # long run this should be handled by the json configs themselves
    cls_defaults = ["train", "val", "tiny", "challenge"]

    data_loaders = config.init(
        name='data_loader',
        module=module_data,
        logger=logger,
        raw_input_dims=raw_input_dims,
        text_feat=config["experts"]["text_feat"],
        text_dim=config["experts"]["text_dim"],
        text_agg=config["experts"]["text_agg"],
        use_zeros_for_missing=config["experts"].get("use_zeros_for_missing", False),
        task=config.get("task", "retrieval"),
        cls_partitions=config.get("cls_partitions", cls_defaults),
    )

    model = config.init(
        name='arch',
        module=module_arch,
        trn_config=trn_config,
        expert_dims=expert_dims,
        text_dim=config["experts"]["text_dim"],
        disable_nan_checks=config["disable_nan_checks"],
        task=config.get("task", "retrieval"),
        ce_shared_dim=config["experts"].get("ce_shared_dim", None),
        feat_aggregation=config["data_loader"]["args"]["feat_aggregation"],
        trn_cat=config["data_loader"]["args"].get("trn_cat", 0),
    )
    logger.info(model)

    metrics = [getattr(module_metric, met) for met in config['metrics']]
    visualizer = config.init(
        name='visualizer',
        module=module_vis,
        exp_name=config._exper_name,
        web_dir=config._web_log_dir,
    )
    ckpt_path = config._args.resume
    logger.info(f"Loading checkpoint: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing.  Note that some datasets fail to fit the retrieval
    # set on the GPU, so we run them on the CPU
    if torch.cuda.is_available() and not config.get("disable_gpu", True):
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Running evaluation on {device}")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        samples, meta = data_loaders["retrieval"]

        # To use the nan-checks safely, we need make temporary copies of the data
        disable_nan_checks = config._config["disable_nan_checks"]
        with ctxt_mgr(samples, device, disable_nan_checks) as valid:
            output = model(**valid)

        sims = output["cross_view_conf_matrix"].data.cpu().float().numpy()
        dataset = data_loaders.dataset_name
        nested_metrics = {}
        for metric in metrics:
            metric_name = metric.__name__
            res = metric(sims, query_masks=meta["query_masks"])
            verbose(epoch=0, metrics=res, name=dataset, mode=metric_name)
            if trainer is not None:
                if not trainer.mini_train:
                    trainer.writer.set_step(step=0, mode="val")
                # avoid tensboard folding by prefixing
                metric_name_ = f"test_{metric_name}"
                trainer.log_metrics(res, metric_name=metric_name_, mode="val")
            nested_metrics[metric_name] = res

    if data_loaders.num_test_captions == 1:
        visualizer.visualize_ranking(
            sims=sims,
            meta=meta,
            epoch=0,
            nested_metrics=nested_metrics,
        )
    log = {}
    for subkey, subval in nested_metrics.items():
        for subsubkey, subsubval in subval.items():
            log[f"test_{subkey}_{subsubkey}"] = subsubval
    for key, value in log.items():
        logger.info(" {:15s}: {}".format(str(key), value))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', default=None, type=str, help="config file path")
    args.add_argument('--resume', default=None, help='path to checkpoint for evaluation')
    args.add_argument('--device', help='indices of GPUs to enable')
    args.add_argument('--eval_from_training_config', action="store_true",
                      help="if true, evaluate directly from a training config file.")
    args.add_argument("--custom_args", help="qualified key,val pairs")
    eval_config = ConfigParser(args)

    msg = "For evaluation, a model checkpoint must be specified via the --resume flag"
    assert eval_config._args.resume, msg
    evaluation(eval_config)
