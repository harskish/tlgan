import sys
from pathlib import Path
from importlib import import_module

def try_import(root, name):
    try:
        sys.path += [str(ext_root / root)]
        return import_module(name, package=None)
    except ModuleNotFoundError as e:
        if e.name == name:
            raise RuntimeError(f'ERROR: Could not import {name} from {root}, please run "git submodule update --init --recursive"')
        raise e
    finally:
        sys.path = sys.path[:-1]

ext_root = Path(__file__).parent.resolve()

resize_right = try_import('ResizeRight', 'resize_right')
#loftr = try_import('LoFTR', 'src.loftr')
#loftr_plotting = try_import('LoFTR', 'src.utils.plotting')