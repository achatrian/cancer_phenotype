from typing import NamedTuple


def print_options(args, parser):
    r"""Print arguments to terminal, including their default values if this is overwritten"""
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    return message


def to_num(s):
    r"""Convert strings into integers if they have no decimal digits, otherwise convert to float"""
    try:
        return int(s)
    except ValueError:
        return float(s)


def read_range(args, parameter):
    r"""Read a string defining a range of the type start:stop:step, start:stop:step,"""
    try:  # 'start:stop:step'
        start, stop, step = tuple(int(d) for d in getattr(args, f'{parameter}').split(':'))
    except ValueError:
        try:  # 'start:stop'
            start, stop = tuple(int(d) for d in getattr(args, f'{parameter}').split(':'))
            step = 1
        except ValueError:  # ':stop'
            try:
                stop = int(getattr(args, f'{parameter}').split(':')[1])
                start, step, = 0, 1
            except (IndexError, ValueError):  # error at splitting or error at conversion
                raise ValueError(f"Range strings for parameters must be specified as start:stop:[step]")
    return range(start, stop, step)


def read_parameter_values(args, parameter):
    r"""Function to read different types of parameter values strings"""
    try:
        parameter_values = [to_num(getattr(args, f'{parameter}'))]
    except ValueError:
        try:
            parameter_values = read_range(args, parameter)  # range defined using :
        except ValueError:
            try:
                # or read ; - separated values
                parameter_values = tuple(to_num(s) for s in getattr(args, f'{parameter}').split(';'))
            except ValueError:
                raise ValueError(f"Unknown format for parameter values: '{getattr(args, f'{parameter}')}'")
    return parameter_values


def print_parameters(parameters: NamedTuple, modifier='', print_function=print):
    message = f'----------------- {modifier}Parameters ---------------\n'
    for parameter, value in sorted(parameters._asdict().items()):
        message += f'{parameter}: {value}; '
    message += '\n'
    message += '------------------------------------------------'
    print_function(message)


def is_namedtuple(obj):
    return isinstance(obj, tuple) and hasattr(obj, '_fields')
