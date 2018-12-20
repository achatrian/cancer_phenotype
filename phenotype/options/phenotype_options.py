import argparse
from base.options.task_options import TaskOptions


class PhenotypeOptions(TaskOptions):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('--model', type=str, default="Inception", help="The network model that will be used")
        self.parser.add_argument('--dataset_name', type=str, default="tcga")
