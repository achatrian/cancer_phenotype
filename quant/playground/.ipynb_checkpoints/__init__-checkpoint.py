from pandas import DataFrame


class Experiment:

    def __init__(self):
        self.x = None
        self.y = None
        self.data_steps = {'original': None, 'result': None}

    def read_data(self):


    def run(self):
        r"""Abstract method where y is computed"""
        pass

