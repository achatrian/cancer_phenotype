import argparse
from collections import namedtuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from quant.experiment import find_experiment_using_name
from quant.utils import print_options, read_parameter_values, print_parameters


def grid_points(experiment_type, args, parameters=None, level=0):
    r"""Function initiating one level of a for loop, corresponding to one parameter in the grid search"""
    if parameters is None:
        Parameters = namedtuple('Parameters', experiment_type.parameters_names,
                                defaults=[None]*len(experiment_type.parameters_names))
        parameters = Parameters()
    try:
        # levels 0 to P - 1
        parameter = experiment_type.parameters_names[level]
        parameter_values = read_parameter_values(args, parameter)
        if not hasattr(args, f'{parameter}'):
            raise ValueError(f"No parameter values are defined for parameter {parameter} (level {level})")
        level_points = []
        for parameter_value in tqdm(parameter_values, desc=f"{parameter} ", position=level):
            run_parameters = parameters._replace(**{parameter: parameter_value})  # returns updated copy of namedtuple
            points = grid_points(experiment_type, args, run_parameters, level + 1)
            level_points.extend(points)
        return level_points
    except IndexError:
        # level P
        assert not any(value is None for value in parameters._asdict().values()), \
            "At deepest level of search, all parameters have been assigned (are not None)"
        return [parameters]


def preprocess_grid_points(experiment_type, args, parameters=None, level=0):
    if parameters is None:
        Parameters = namedtuple('Parameters', experiment_type.preprocess_parameters_names + experiment_type.parameters_names,
                                defaults=[None] * (len(experiment_type.preprocess_parameters_names) + len(
                                             experiment_type.parameters_names)))
        parameters = Parameters()
        # levels 0 to PP - 1
    try:
        parameter = experiment_type.preprocess_parameters_names[level]
        parameter_values = read_parameter_values(args, parameter)
        level_points = []
        for parameter_value in tqdm(parameter_values, desc=f"{parameter} ", position=level):
            run_parameters = parameters._replace(**{parameter: parameter_value})  # returns updated copy of namedtuple
            points = preprocess_grid_points(experiment_type, args, run_parameters, level + 1)
            level_points.extend(points)
        return level_points
    except IndexError:
        # level PP - where grid search over parameters is started
        return grid_points(experiment_type, args, parameters, 0)


if __name__ == '__main__':
    # set up experiment_type
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help="Description of task")
    parser.add_argument('--experiment', type=str, help="experiment type - used to load desired experiment class")
    parser.add_argument('--data_dir', type=Path, help="Directory where data is stored")
    parser.add_argument('--exp_help', action='store_true')
    args, unparsed = parser.parse_known_args()
    experiment_type = find_experiment_using_name(args.experiment, args.task)
    parser = experiment_type.modify_commandline_options(parser)
    args = parser.parse_args()
    if args.exp_help:
        import sys
        parser.print_help()
        sys.exit(0)
    parameter_points = preprocess_grid_points(experiment_type, args)
    experiment_name = experiment_type.name().lower().split('experiment')[0]
    scheduled_runs_dir = Path(args.data_dir, 'data', 'experiments', experiment_name, 'scheduled_runs')
    scheduled_runs_dir.mkdir(exist_ok=True, parents=True)
    # turn grid points into command-separated parameter values
    runs_path = Path(scheduled_runs_dir, f'runs_{str(datetime.now())[:17].replace(" ", "_")}.txt')
    print(f"Writing runs parameters into {str(runs_path)}")
    with open(runs_path, 'w') as runs_file:
        for parameter_point in parameter_points:
            options_str = ''
            for parameter, value in parameter_point._asdict().items():
                options_str += f'--{parameter}={value},'
            options_str += '\n'
            runs_file.write(options_str)

