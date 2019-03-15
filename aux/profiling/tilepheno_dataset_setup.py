import sys
import os
sys.path.extend(['/well/rittscher/users/achatrian/cancer_phenotype/base',
                 '/well/rittscher/users/achatrian/cancer_phenotype'])
from base.options.train_options import TrainOptions
from base.data import create_dataset

opt = TrainOptions().parse()
train_dataset = create_dataset(opt)
train_dataset.setup()
print("Done!")
