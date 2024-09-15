import json

from logger import logger

def get_cls_or_func(path_str:str):
    """
    import a class or function from a string path
    """
    parts = path_str.split('.')
    module_path = '.'.join(parts[:-1])
    cls_name = parts[-1]
    logger.debug(f'Loading submodule {cls_name} from module {module_path}')
    module = __import__(module_path, fromlist=[cls_name])
    cls = getattr(module, cls_name)
    return cls


def load_config(config_file):
    """
    Load a json config file
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config