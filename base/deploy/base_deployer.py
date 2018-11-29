from pathlib import Path
import torch.multiprocessing as mp
from base.data import create_dataset, create_dataloader

# Which philosophy should one embrace ? Build initial good document describing dataset, or deal with multiple documents in code?
# When is it convenient to do both?
# In TCGA, there are many fields for each one of the data modalities, and thus the data are split


class BaseDeployer:

    def __init__(self, opt):
        super(BaseDeployer, self).__init__()
        self.opt = opt
        self.data_fields = (self.opt.sample_id_name,)
        self.fields_datatype = ('text',)

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        ABSTRACT METHOD
        :param parser:
        :param is_train:
        :return:
        """
        return parser

    def name(self):
        return "BaseDeployer"

    def setup(self):

        # assign a split to each child process
        n_slides_per_child = len(self.sample_id_names) // self.opt.ndeploy_workers
        n_child0 = n_slides_per_child + len(self.sample_id_names) % self.opt.ndeploy_workers
        splits_bounds = (0, n_child0) + tuple(n_child0 + n_slides_per_child * i for i in range(self.opt.ndeploy_workers))
        splits_bounds = tuple((splits_bounds[i], splits_bounds[i+1]) for i in range(self.opt.ndeploy_workers))
        self.splits = tuple(
            {name: getattr(self, name)[slice(*b)] for name in [self.opt.sample_id_name] + meta_field_names}
            for b in splits_bounds)

    def make_workers(self, model):
        """
        Give different splits of the data to different workers
        :return:
        """
        workers = []
        for i in self.opt.ndeploy_workers:
            dataset = self.create_dataset(i)
            dataloader = create_dataloader(dataset)
            worker = mp.Process(target=self.run_deployment_worker,
                       args=(dataset, model, self.opt))
            workers.append(worker)
        return workers

    def create_dataset(self, split, validation_phase=False):
        if self.opt.is_train and split > 1:
            raise ValueError("During training, only split = 0 for training and = 1 for validation is supported")
        dataset = create_dataset(self.opt, validation_phase)
        try:
            dataset.assign_data(files=self.splits[split][self.opt.sample_id_name],
                                metadata=self.splits[split])
        except AttributeError as exc:
            raise ValueError("Dataset {} cannot be used for deployment to database".format(dataset.name())) from exc
        return dataset

    @staticmethod
    def run_deployment_worker(dataloader, model, opt, queue=None):
        """
        ABSTRACT METHOD: implemented in subclasses to run multiprocessing on workers that use/update the model
        :param dataloader:
        :param model:
        :param opt:
        :param queue:
        :return:
        """
        pass








