import argparse
from pathlib import Path
from collections import namedtuple
from quant.experiment import BaseExperiment, find_experiment_using_name, create_experiment
from quant.utils import print_options, read_parameter_values, print_parameters


r"""
Script for applying experiment results to new / different dataset
"""


if __name__ == '__main__':
    # set up experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help="Description of task")
    parser.add_argument('--experiment', type=str, help="Experiment type - used to load desired experiment class")
    parser.add_argument('--data_dir', type=Path, help="Directory where data is stored")
    parser.add_argument('--preprocess_search', action='store_true', help="whether to perform a search over preprocessing parameters")
    parser.add_argument('--exp_help', action='store_true')
    parser.add_argument('--debug', action='store_true', help="flag used in experiment to run tests for debugging")
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
    all_parameter_names = experiment.preprocess_parameters_names + experiment.parameters_names
    # parameters = namedtuple('Parameters', all_parameter_names,
    #                         defaults=[None] * len(all_parameter_names))()  # defaults do not work with python 3.6
    parameters = namedtuple('Parameters', all_parameter_names)(None, None, None, None, None)
    parameters_assignment = {parameter: read_parameter_values(args, parameter).pop()
                             for parameter in all_parameter_names}
    parameters = parameters._replace(**parameters_assignment)
    experiment.read_data()
    experiment.preprocess()
    experiment.apply(parameters)
