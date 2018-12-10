import importlib
import numpy as np
from .base_deployer import BaseDeployer


def find_deployer_using_name(deployer_name, task_name):
    # Given the option --deployer [deployername],
    # the file "deployers/deployername_deployer.py"
    # will be imported.
    task_module = importlib.import_module(task_name)
    deployer_filename = task_name + ".deploy." + deployer_name.lower() + "_deployer"
    deployerlib = importlib.import_module(deployer_filename, package=task_module)

    # In the file, the class called deployerNamedeployer() will
    # be instantiated. It has to be a subclass of Basedeployer,
    # and it is case-insensitive.
    deployer = None
    target_deployer_name = deployer_name.replace('_', '') + 'deployer'
    for name, cls in deployerlib.__dict__.items():
        if name.lower() == target_deployer_name.lower() \
                and next(iter(cls.__bases__)).__module__.endswith(BaseDeployer.__module__):
            deployer = cls

    if deployer is None:
        raise AttributeError("In %s.py, there should be a subclass of BaseDeployer with class name that matches %s in lowercase." % (deployer_filename, target_deployer_name))

    return deployer


def create_deployer(opt):
    deployer = find_deployer_using_name(opt.deployer_name, opt.task)
    instance = deployer(opt)
    print('dataset [{}] was created'.format(instance.name()))
    return instance
