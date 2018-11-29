import os
import imageio
from base.deploy.base_deployer import BaseDeployer


class TCGADeployer(BaseDeployer):

    def __init__(self, opt):
        super(TCGADeployer, self).__init__(opt)
        self.data_fields = (self.opt.sample_id_name, 'sample_id', 'case_id', 'sample_submitter_id', 'is_ffpe', 'sample_type',
                                 'state', 'oct_embedded')
        self.fields_datatype = {field_name: 'text' for field_name in self.meta_field_names}
        """
        slide_id:
        is_ffpe: whether the ffpe version of a slide has been released or not
        state: release state - can be 'released' | '--' 
        """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_default(sample_id_name='case_submitter_id')
        return parser

    def name(self):
        return "TCGADeployer"

    @staticmethod
    def run_deployment_worker(dataloader, model, opt, queue=None):
        assert dataloader.dataset.name() == "WSIDataset"
        if not opt.is_train:
            model.eval()
            for i, data in enumerate(dataloader):
                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()
                if opt.results_dir:
                    for path, image in visuals.items():
                        save_path = os.path.join(opt.save_dir, path)
                        imageio.imwrite(visuals, save_path)

        else:
            raise NotImplementedError("Training")



