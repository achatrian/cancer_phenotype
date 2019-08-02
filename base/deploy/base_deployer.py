from itertools import cycle
import copy
from contextlib import contextmanager
import torch.multiprocessing as mp


class BaseDeployer(mp.Process):

    def __init__(self, opt):
        super(BaseDeployer, self).__init__()
        self.opt = opt
        self.worker_gpu_ids = None
        mp.set_sharing_strategy('file_system')  # https://github.com/pytorch/pytorch/issues/973#issuecomment-426559250
        self.worker_name = ''  # name given to workers

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
    def queue_env(self, sentinel=True):
        # __enter__
        queue = mp.JoinableQueue(2 * min(self.opt.ndeploy_workers, 1))
        yield queue
        # __exit__
        if sentinel:
            for i in range(self.opt.ndeploy_workers):
                queue.put(None)  # sentinel for workers to terminate
        queue.join()

    def get_workers(self, model, input_queue, output_queue=None, sync=()):
        """
        Give different splits of the data to different workers
        :return:
        """
        # set GPU allocation
        if self.opt.gpu_ids:
            self.worker_gpu_ids = []
            for gpu_id, worker_id in zip(cycle(self.opt.gpu_ids), range(self.opt.ndeploy_workers)):
                # cycle over gpu assignment - each worker has 1 gpu only
                self.worker_gpu_ids.append([gpu_id])
        else:
            self.worker_gpu_ids = ['cpu'] * self.opt.ndeploy_workers

        workers = []
        for i in range(self.opt.ndeploy_workers):
            opt = copy.copy(self.opt)
            opt.gpu_ids = self.worker_gpu_ids[i]
            args = (i, opt, model, input_queue, output_queue)
            worker = mp.Process(target=self.run_worker, args=args, name=self.worker_name + 'Worker', daemon=True)
            workers.append(worker)

        # optional gather-worker to process outputs of workers
        if self.opt.gatherer:
            gatherer = mp.Process(target=self.gather, args=(self, output_queue, sync), name=self.worker_name + 'Gatherer', daemon=True)
            workers.append(gatherer)
        return workers

    @staticmethod
    def run_worker(process_id, opt, model, input_queue, output_queue=None):
        """
        ABSTRACT METHOD: implemented in subclasses to run multiprocessing on workers that use/update the model
        Rules:
        workers must terminate if they get a None from the queue
        """
        pass

    @staticmethod
    def gather(deployer, output_queue, sync=()):
        """
        Abstract method to gather and process outputs of workers (separate process that runs with workers)
        :param sync
        :param deployer
        :param output_queue:
        """
        pass

    def cleanup(self, output=None):
        """
        Abstract method, called to process all data at end (main process)
        :return:
        """








