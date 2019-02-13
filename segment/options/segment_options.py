import argparse
from base.options.task_options import TaskOptions


class SegmentOptions(TaskOptions):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('--model', type=str, default="UNet", help="The network model that will be used")
        self.parser.add_argument('--deployer_name', type=str, default='segment', help="The deployer used to batch apply the network to the data")
        self.parser.add_argument('--dataset_name', type=str, default='tileseg')
        pass
