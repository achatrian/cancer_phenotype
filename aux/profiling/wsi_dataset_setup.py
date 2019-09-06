import sys
import os
sys.path.extend(['/well/rittscher/users/achatrian/cancer_phenotype/base',
                 '/well/rittscher/users/achatrian/cancer_phenotype'])
from base.options.apply_options import ApplyOptions
from base.datasets import create_dataset


opt = ApplyOptions().parse()
opt.dataset_name = 'wsi'
dataset = create_dataset(opt)
dataset.setup()