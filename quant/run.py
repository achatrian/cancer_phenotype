import argparse
from collections import namedtuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
from quant.experiment import BaseExperiment, find_experiment_using_name, create_experiment
from quant.utils import print_options, read_parameter_values, print_parameters
np.random.seed(42)  # ensure reproducibility for numpy and scikit-learn applications


r"""Script for running sequentially a grid search on an experiment over an arbitrary number of parameters"""


def grid_search(experiment, parameters=None, level=0):
    r"""Function initiating one level of a for loop, corresponding to one parameter in the grid search"""
    if parameters is None:
        Parameters = namedtuple('Parameters', experiment.parameters_names,
                                defaults=[None]*len(experiment.parameters_names))
        parameters = Parameters()
    try:
        # levels 0 to P - 1
        parameter = experiment.parameters_names[level]
        parameter_values = read_parameter_values(experiment.args, parameter)
        if not hasattr(experiment.args, f'{parameter}'):
            raise ValueError(f"No parameter values are defined for parameter {parameter} (level {level})")
        level_results = {}
        for parameter_value in tqdm(parameter_values, desc=f"{parameter} "):
            run_parameters = parameters._replace(**{parameter: parameter_value})  # returns updated copy of namedtuple
            results = grid_search(experiment, run_parameters, level + 1)
            level_results.update(results)
        return level_results
    except IndexError:
        # level P - where experiment is run
        assert not any(value is None for value in parameters._asdict().values()), \
            "At deepest level of search, all parameters have been assigned (are not None)"
        print_parameters(parameters, 'Run ', tqdm.write)
        experiment.run(parameters)
        result = {parameters: experiment.evaluate()}
        experiment.save()
        return result


# TODO merge this function and above and make preprocess search conditional on having any preprocessing parameters
def preprocess_grid_search(experiment, parameters=None, level=0):
    if parameters is None:
        Parameters = namedtuple('Parameters', experiment.preprocess_parameters_names + experiment.parameters_names,
                                defaults=[None] * (len(experiment.preprocess_parameters_names) + len(
                                             experiment.parameters_names)))
        parameters = Parameters()
        # levels 0 to PP - 1
    try:
        parameter = experiment.preprocess_parameters_names[level]
        parameter_values = read_parameter_values(experiment.args, parameter)
        level_results = {}
        for parameter_value in tqdm(parameter_values, desc=f"{parameter} "):
            run_parameters = parameters._replace(**{parameter: parameter_value})  # returns updated copy of namedtuple
            results = preprocess_grid_search(experiment, run_parameters, level + 1)
            level_results.update(results)
        return level_results
    except IndexError:
        # level PP - where grid search over parameters is started
        experiment.preprocess(parameters)
        return grid_search(experiment, parameters, 0)


if __name__ == '__main__':
    # set up experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help="Description of task")
    parser.add_argument('--experiment', type=str, help="Experiment type - used to load desired experiment class")
    parser.add_argument('--data_dir', type=Path, help="Directory where data is stored")
    parser.add_argument('--preprocess_search', action='store_true', help="whether to instantiate a search over preprocessing parameters")
    parser.add_argument('--exp_help', action='store_true')
    args, unparsed = parser.parse_known_args()
    experiment_type = find_experiment_using_name(args.experiment, args.task)
    parser = experiment_type.modify_commandline_options(parser)
    args = parser.parse_args()
    if args.exp_help:
        import sys
        parser.print_help()
        sys.exit(0)
    print_options(args, parser)
    experiment: BaseExperiment = create_experiment(args)
    print(f"Experiment {experiment.name()} was created")
    # read the data in
    experiment.read_data()
    if not len(experiment):
        experiment.preprocess()
        experiment.run()
        results = experiment.evaluate()
    if not args.preprocess_search:
        experiment.preprocess()
        results = grid_search(experiment)
    else:
        results = preprocess_grid_search(experiment)
    experiment.save_results(results)
    # choose best run
    best_parameters = experiment.select_best(results)
    print_parameters(best_parameters, modifier='Best ')
    # postprocessing
    experiment.postprocess(best_parameters, results[best_parameters])


