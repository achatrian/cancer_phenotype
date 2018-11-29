import argparse


class TaskOptions:
    """
    Base class to be inherited from task instances when they want to add task-dependent options.
    E.g. segmentation options for images.
    The options from this object are added to the options in BaseOptions
    """
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                         add_help=False)  # needed to merge parsers
        parser.add_argument('--dummy', default='dummy')

        self.parser = parser
        self.is_train = None
        self.opt = None