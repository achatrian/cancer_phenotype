import sys
import pytest
from base.models import create_model
from base.deploy import create_deployer


def test_segment_deploy(apply_options, wsi_file):

    sys.argv.extend(
        [
            '--data_dir=/well/rittscher/projects/TCGA_prostate/TCGA'
        ]
    )
    opt = apply_options.parse()
    model = create_model(opt)
    # model.setup()
    model.share_memory()

    deployer = create_deployer(opt)
    deployer.setup()
    workers = deployer.get_workers(model)

    for i, worker in enumerate(workers):
        worker.start()
        print("Worker {} was started ...".format(i))