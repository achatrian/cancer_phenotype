import argparse
from base.options.task_options import TaskOptions


class PhenotypeOptions(TaskOptions):

    def __init__(self):
        super().__init__()
        # Resetting defaults by adding arguments in TaskOption instance
        self.parser.add_argument('--model', type=str, default="Inception", help="The network model that will be used")
        self.parser.add_argument('--dataset_name', type=str, default="tilepheno")
        self.parser.add_argument('--data_dir', type=str, default='')
        self.parser.add_argument('--num_class', type=int, default=2)  # addition / deletion of gene
