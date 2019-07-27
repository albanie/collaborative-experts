import argparse
import torch
import numpy as np
import random
import data_loader.data_loaders as module_data
import utils.visualizer as module_vis
import model.metric as module_metric
import model.model as module_arch
from trainer import valid_samples, verbose
from parse_config import ConfigParser
from utils.util import compute_dims


def evaluation(config, logger=None):

    if logger is None:
        logger = config.get_logger('test')

    logger.info("Running evaluation with configuration:")
    logger.info(config)

    expert_dims, raw_input_dims = compute_dims(config)

    # Set the random initial seeds
    seed = config["seed"]
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
        expert_dims=expert_dims,
        text_dim=config["experts"]["text_dim"],
        disable_nan_checks=config["disable_nan_checks"],
    )
    logger.info(model)

    metrics = [getattr(module_metric, met) for met in config['metrics']]
    visualizer = config.init(
        name='visualizer',
        module=module_vis,
        exp_name=config._exper_name,
        log_dir=config._web_log_dir,
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
    if torch.cuda.is_available() and not config.get("disable_gpu", False):
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
        with valid_samples(samples, device, disable_nan_checks) as valid:
            output = model(**valid)

        sims = output["cross_view_conf_matrix"].data.cpu().float().numpy()
        dataset = data_loaders.dataset_name
        nested_metrics = {}
        for metric in metrics:
            metric_name = metric.__name__
            res = metric(sims, query_masks=meta["query_masks"])
            verbose(epoch=0, metrics=res, name=dataset, mode=metric_name)
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
        logger.info(' {:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', default=None, type=str, help="config file path")
    args.add_argument('--resume', default=None, help='path to checkpoint for evaluation')
    args.add_argument('--device', help='indices of GPUs to enable')
    eval_config = ConfigParser(args)

    msg = "For evaluation, a model checkpoint must be specified via the --resume flag"
    assert eval_config._args.resume, msg
    evaluation(eval_config)
