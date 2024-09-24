import argparse

from utils.utils import load_config


def get_args():
    parser = argparse.ArgumentParser(description="elania Training")
    parser.add_argument("--config_file", type=str, default=None, help="The config file")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    args = parser.parse_args()
    return args


def get_config_from_args(args):
    config_path = args.config_file
    return load_config(config_path)
