import importlib
from typing import NamedTuple
from quant.utils import is_namedtuple


class BaseExperiment:
    parameters_names = ()
    parameters_defaults = ()
    parameters_descriptions = ()
    preprocess_parameters_names = ()
    preprocess_parameters_defaults = ()
    preprocess_parameters_descriptions = ()

    def __init__(self, args):
        self.args = args
        self.run_parameters = None

    @staticmethod
    def name():
        return "BaseExperiment"

    @classmethod
    def modify_commandline_options(cls, parser):
        r"""
        Adds parameter range options.
        Can be modified in subclasses to add experiment-specific options to parser
        """
        for name, default, description in zip(
                cls.preprocess_parameters_names,
                cls.preprocess_parameters_defaults,
                cls.preprocess_parameters_descriptions):
            parser.add_argument(f'--{name}', type=str, default=default, help=f"{description}. Values for preprocessing parameter {name}. Formats: number, range ('[start]:stop[:step]') or values (v1;v2;....)")
        for name, default, description in zip(
                cls.parameters_names,
                cls.parameters_defaults,
                cls.parameters_descriptions):
            parser.add_argument(f'--{name}', type=str, default=default, help=f"{description}. Values for preprocessing parameter {name}. Formats: number, range ('[start]:stop[:step]') or values (v1;v2;....)")
        return parser

    def __len__(self):  # number of parameters
        return len(self.parameters_names) + len(self.preprocess_parameters_names)

    def read_data(self):
        r"""
        Abstract method: reads data to perform experiment on.
        In case data is read sequentially when running the experiment, this function is called multiple times
        """
        pass

    def preprocess(self, parameters=None):
        r"""
        :parameter parameters: optional parameters for preprocessing
        Abstract method: transforms data before the experiment is run
        """

    def run(self, parameters=None):
        r"""
        Abstract method: implements the experimental procedure
        """
        pass

    def evaluate(self):
        r"""
        Abstract method: evaluates the experiments' results
        """
        pass

    def save(self):
        r"""
        Abstract method: saves result and evaluation
        """

    @staticmethod
    def format_parameters_key(parameters: NamedTuple):
        if not is_namedtuple(parameters):
            raise ValueError(f"Parameters must be a NamedTuple, not {type(parameters)}")
        parameters_key = ''
        for parameter, value in parameters._asdict().items():
            parameters_key += f'{parameter}:{value}'
        return parameters_key

    def save_results(self, results):
        r"""
        Save results from multiple runs of the experiment
        :param results:
        :return:
        """
        pass

    def select_best(self, results) -> NamedTuple:
        r"""
        Called at end of grid search. Chooses best run based on results
        :param results:
        :return:
        """
        pass

    def postprocess(self, best_parameters, best_result):
        r"""
        Heavier save method, serialising entire models, only called on the best method
        :return:
        """


def find_experiment_using_name(experiment_name, task_name):
    r"""Finds BaseExperiment subclass in any of directories inside quant folder"""
    try:
        task_module = importlib.import_module(task_name)
        experiment_filename = 'quant.' + task_name + '.' + experiment_name + "_experiment"
        experimentlib = importlib.import_module(experiment_filename, package=task_module)
    except ModuleNotFoundError as err1:
            try:
                # if module not found, attempt to load from base
                task_module = importlib.import_module(task_name)
                experiment_filename = 'quant.' + 'experiment.' + experiment_name + "_experiment"
                experimentlib = importlib.import_module(experiment_filename, package=task_module)
            except ModuleNotFoundError:
                if not err1.args:
                    err1.args = ('',)
                err1.args = err1.args + (f"{task_name}.experiments contains no file '{experiment_name}.py'",)
                raise err1
    except ImportError as importerr:
        if not importerr.args:
            importerr.args = ('',)
        importerr.args = importerr.args + (f"Module {task_name} not found.",)
        raise

    # In the file, the class called experimentNameexperiment() will
    # be instantiated. It has to be a subclass of Baseexperiment,
    # and it is case-insensitive.

    def is_subclass(subclass, superclass):
        return next(iter(subclass.__bases__)).__module__.endswith(superclass.__module__)

    experiment = None
    target_experiment_name = experiment_name.replace('_', '') + 'experiment'
    for name, cls in experimentlib.__dict__.items():
        if name.lower() == target_experiment_name.lower():
            if is_subclass(cls, BaseExperiment) or any(is_subclass(cls_b, BaseExperiment) for cls_b in cls.__bases__):
                experiment = cls

    if experiment is None:
        raise NotImplementedError("In {}.py, there should be a subclass of Baseexperiment with class name that matches {} in lowercase.".format(
              experiment_filename, target_experiment_name))

    return experiment


def create_experiment(args):
    experiment = find_experiment_using_name(args.experiment, args.task)
    instance = experiment(args)
    return instance