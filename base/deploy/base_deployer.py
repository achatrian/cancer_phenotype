from itertools import cycle
import copy
from contextlib import contextmanager
import torch.multiprocessing as mp


class BaseDeployer:

    def __init__(self, opt):
        super(BaseDeployer, self).__init__()
        self.opt = opt

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

    @contextmanager
    def start_data_loading(self):
        # __enter__
        queue = mp.JoinableQueue(2 * self.opt.ndeploy_workers)
        yield queue
        # __exit__
        for i in range(len(self.opt.ndeploy_workers)):
            queue.put(None)  # workers terminate as they read the None
        queue.join()

    def get_workers(self, model, queue):
        """
        Give different splits of the data to different workers
        :return:
        """
        # set GPU allocation
        self.worker_gpu_ids = []
        for gpu_id, worker_id in zip(cycle(self.opt.gpu_ids), range(self.opt.ndeploy_workers)):
            # cycle over gpu assignment - each worker has 1 gpu only
            self.worker_gpu_ids.append([gpu_id])

        workers = []
        for i in range(self.opt.ndeploy_workers):
            opt = copy.copy(self.opt)
            opt.gpu_ids = self.worker_gpu_ids[i]
            args = (i, opt, model, queue)
            worker = mp.Process(target=self.run_worker,
                                args=args)
            workers.append(worker)
        return workers

    @staticmethod
    def run_worker(process_id, opt, model, queue):
        """
        ABSTRACT METHOD: implemented in subclasses to run multiprocessing on workers that use/update the model
        Rules:
        workers must terminate if they get a None from the queue
        """
        pass








