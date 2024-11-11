import os
from _collections_abc import dict_values
import importlib.util

class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __str__(self) -> str:
        return str(self._dict)

    def __call__(self, name=None):
        return self.registry(name)

    def registry(self,name):
        
        def add_to_dict(key, cls):
            if key in self._dict:
                print(f"Key {key} already exists")
            self._dict[key] = cls
        
        cls = name
        if callable(cls): # 传入cls,无参使用,直接完成注册返回cls
            add_to_dict(cls.__name__, cls)
            return cls
        else: # 传入name,有参使用,返回装饰器
            def decorator(cls_or_fn):
                key = name if name is not None else cls_or_fn.__name__
                add_to_dict(key, cls_or_fn)
                return cls_or_fn

            return decorator

    def keys(self):
        return self._dict.keys()

    def values(self) -> dict_values:
        return self._dict.values()

    def items(self):
        return self._dict.items()


register = Register()


def registry_pycls_by_path(path):

    if not os.path.exists(path):
        return
    
    if os.path.isfile(path) and not path.endswith(".py"):
        return
    
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file != '__pycache__':
                registry_pycls_by_path(os.path.join(path, file))
        return
    
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


