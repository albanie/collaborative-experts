import os
import time
import pprint
import logging
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, args, options='', timestamp=True, slave_mode=False):
        # slave_mode - when calling the config parser form an existing process, we
        # avoid reinitialising the logger and ignore sys.argv when argparsing.

        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)

        if slave_mode:
            args = args.parse_args(args=[])
        else:
            args = args.parse_args()

        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        if args.resume and not slave_mode:
            self.resume = Path(args.resume)
            # self.cfg_fname = self.resume.parent / 'config.json'
        else:
            msg_no_cfg = "Config file must be specified"
            assert args.config is not None, msg_no_cfg
            self.resume = None
        self.cfg_fname = Path(args.config)

        # load config file and apply custom cli options
        config = read_json(self.cfg_fname)
        self._config = _update_config(config, options, args)

        if self._config.get("eval_config", False):
            # validate path to evaluation file
            eval_cfg_path = self._config.get("eval_config")
            msg = f"eval_config was specified, but `{eval_cfg_path}` does not exist"
            assert Path(self._config.get("eval_config")).exists(), msg

        # set save_dir where trained model and log will be saved.
        if "trainer" in self.config:
            save_dir = Path(self.config['trainer']['save_dir'])
        else:
            save_dir = Path(self.config['tester']['save_dir'])
        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S") if timestamp else ""

        if slave_mode:
            timestamp = f"{timestamp}-eval-worker"

        # We assume that the config files are organised into directories such that
        # each directory has the name of the dataset.
        dataset_name = self.cfg_fname.parent.stem
        exper_name = f"{dataset_name}-{self.cfg_fname.stem}"
        self._save_dir = save_dir / 'models' / exper_name / timestamp
        self._log_dir = save_dir / 'log' / exper_name / timestamp
        self._web_log_dir = save_dir / 'web' / exper_name / timestamp
        self._exper_name = exper_name
        self._args = args

        # if set, remove all previous experiments with the current config
        if vars(args).get("purge_exp_dir", False):
            for dirpath in (self._save_dir, self._log_dir, self._web_log_dir):
                config_dir = dirpath.parent
                existing = list(config_dir.glob("*"))
                print(f"purging {len(existing)} directories from config_dir...")
                tic = time.time()
                os.system(f"rm -rf {config_dir}")
                print(f"Finished purge in {time.time() - tic:.3f}s")

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        if not slave_mode:
            self.log_path = setup_logging(self.log_dir)

        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def init(self, name, module, *args, **kwargs):
        """Finds a function handle with the name given as 'type' in config, and returns
        the instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        msg = 'Overwriting kwargs given in config file is not allowed'
        assert all([k not in module_args for k in kwargs]), msg
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get(self, name, default):
        return self.config.get(name, default)

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}."
        msg_verbosity = msg_verbosity.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    def __repr__(self):
        return pprint.PrettyPrinter().pformat(self.__dict__)


# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
