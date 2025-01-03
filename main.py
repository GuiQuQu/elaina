
from transformers import set_seed

from arguments import get_args,get_config_from_args
from trainer.build import build_trainer
from tester.build import build_tester

from utils.utils import check_environment
from utils.register import registry_pycls_by_path
from utils.register import Register

from logger import init_transformer_logger, logger

def initialize_by_config(config):
    registry_paths = config['registry_paths']
    for path in registry_paths:
        registry_pycls_by_path(path)

    if config['seed'] is not None:
        set_seed(config['seed'])
    else:
        logger.warning("rand seed is not set, the result is not reproducible.")

def main():
    init_transformer_logger()
    check_environment()
    args = get_args()
    config = get_config_from_args(args)

    initialize_by_config(config)

    if args.do_train:
        trainer = build_trainer(config)
        trainer.train()
    if args.do_test:
        tester = build_tester(config)
        tester.test()
    
if __name__ == "__main__":
    main()