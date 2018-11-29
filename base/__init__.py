from . import *
import importlib
import pkgutil


def import_subpackages(package_path):
    submodules = []
    for importer, modname, ispkg in pkgutil.iter_modules(package_path):
        submodule = importer.find_module(modname).load_module(modname)
    return submodules
