import argparse
from segment.options.deploy_options import DeployOptions



# Overwrite Base option class
#
# class BaseOptions(DeployOptions):
#     def __init__(self):
#         super(DeployOptions, self).__init__()
#         # new arguments
#         self.parser.add_argument('--metadata_dir', default='/gpfs0/well/rittscher/projects/TCGA_prostate/data')
#
#         # instead of calling an error for two actions with the same name, call the second one
#         self.parser.conflict_handler = 'resolve'
#         for action_group in self.parser._action_groups:
#             action_group.conflict_handler = 'resolve'
#         self.parser.set_defaults(dataset_name='WSIDataset')
#         self.parser.set_defaults(data_dir='/gpfs0/well/rittscher/projects/TCGA_prostate')
#         self.parser.add_argument('--model', type=str, default='ResNet', choices=['ResNet'], help='The network model used for classification')
#
#
