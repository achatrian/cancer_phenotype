import os
from contextlib import contextmanager
from itertools import chain
import torch
from base.utils import utils
from . import networks

# Benefits of having one skeleton, e.g. for train - is that you can keep all the incremental changes in
# one single code, making it your streamlined and updated script -- no need to keep separate logs on how
# to implement stuff


class BaseModel:
    """
    Philosophy: a model is different from a pytorch module, as a model may contain
    multiple networks that have a forward method that is not sequential
    (thing about VAE-GAN)

    call method is forward() -- as in Module

    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.experiment_name)
        self.loss_names = []
        self.metric_names = []
        self.module_names = []  # changed from 'model_names'
        self.visual_names = []
        self.visual_types = []  # added to specify how to plot (e.g. in case output is a segmentation map)
        self.visual_paths = []
        self.optimizers = []
        self.schedulers = []
        self.meters = dict()
        self.nets = None
        self.input = None
        self.target = None
        self.output = None
        self.is_val = False  # switches behaviour of getters and updates between validation and training
        self.model_tag = 'latest_net'  # keeps track of last saved / loaded model, thus identifying current weights
        self.setup_methods = []  # contains extra functions to be run at setup

    def name(self):
        return 'BaseModel'

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

    def set_input(self, data):
        self.input = data['input']  # 1
        self.visual_paths = {'input': data['input_path'],
                             'output': [''] * len(data['input_path'])}  # and 3 must be returned by dataset
        if 'target' in data:
            self.target = data['target']  # 2
        if 'target_path' in data:
            self.visual_paths['target'] = data['target_path']  # 4 is optional, only for when available
        if self.opt.gpu_ids and (self.input.device.type == 'cpu' or self.target.device.type == 'cpu'):
            self.input = self.input.cuda(device=self.device)
            if 'target' in data:
                self.target = self.target.cuda(device=self.device)

    def forward(self):
        pass

    # load and print networks; create schedulers;
    def setup(self, dataset=None):
        """
        This method shouldn't be overwritten.
        1. Initialises networks and pushes nets and losses to cuda,
        2. Sets up schedulers
        3. Loads and prints networks;
        :param dataset: used
        :return:
        """
        if self.gpu_ids:  # push networks and losses modules to gpus if needed
            for module_name in self.module_names:
                net = getattr(self, "net" + module_name)
                net.train()
                setattr(self, "net" + module_name, networks.init_net(net, self.opt.init_type, self.opt.init_gain,
                                         self.opt.gpu_ids))  # takes care of pushing net to cuda
            assert torch.cuda.is_available(), f"Cuda must be available for gpu option: {str(self.gpu_ids)}"
            for loss_name in self.loss_names:
                loss = getattr(self, loss_name).cuda(device=self.device)
                setattr(self, loss_name, loss)
        if self.is_train:  # make schedulers
            self.schedulers = [networks.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
        if not self.is_train or self.opt.continue_train:
            load_suffix = 'iter_%d' % self.opt.load_iter if self.opt.load_iter > 0 else self.opt.load_epoch
            self.load_networks(load_suffix)
        self.print_networks(self.opt.verbose)
        # run setup functions if any
        for method_name in self.setup_methods:
            getattr(self, method_name)(dataset)
        # add meters to compute average of performance measures
        for name in chain(self.loss_names, self.metric_names):
            self.meters[name] = utils.AverageMeter()
        print("network setup complete")

    # make models eval mode during test time
    def eval(self):
        for name in self.module_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def optimize_parameters(self):
        r"""Abstract method: call forward and backward + other optimization steps
            Use self.update_measure_values to store losses"""
        pass

    def evaluate_parameters(self):
        r"""
        Abstract method that I added -- pix2pix code did not compute evaluation metrics,
        but for many tasks they can be quite useful
        Updates metrics values (metric must start with 'metric_')
        Must use self.update_measure_values
        """
        pass

    def update_measure_value(self, name, value, n_samples=None):
        r"""
        Updates loss or metric. If model is in validation, validation values are updated
        """
        n_samples = n_samples or self.output.shape[0]
        if name in self.loss_names:
            prefix = 'loss_'
        elif name in self.metric_names:
            prefix = 'metric_'
        else:
            raise ValueError(f"Attempted update of unknown measure {name} - add to loss_names or metric_names")
        if self.is_val:
            setattr(self, prefix + name + '_val', value.item() if isinstance(value, torch.Tensor) else value)
        else:
            setattr(self, prefix + name, value)
            self.meters[name].update(value.item() if isinstance(value, torch.Tensor) else value, n_samples)

    def u_(self, name, value, n_samples=None):
        r"""Shortcut to update measures"""
        self.update_measure_value(name, value, n_samples=n_samples)

    @contextmanager
    def start_validation(self):
        """
        Context manager for setting up meter that average the validation metrics over validation data-set,
        and then set the val attributes of the model.
        Use the yielded function 'update_validation_meters' to compute running average of validation metrics
        """
        # __enter__ #
        self.is_val = True  # get functions now get validation measures and batch average is not taken
        loss_meters = {loss_name: utils.AverageMeter() for loss_name in self.loss_names}
        metric_meters = {metric_name: utils.AverageMeter() for metric_name in self.metric_names}
        model = self

        def update_validation_meters():
            # update meters (which remain hidden from main)
            for loss_name in model.loss_names:
                loss = getattr(model, 'loss_' + loss_name + '_val')
                loss_meters[loss_name].update(loss, model.opt.batch_size)
            for metric_name in model.metric_names:
                metric = getattr(model, 'metric_' + metric_name + '_val')
                metric_meters[metric_name].update(metric, self.opt.batch_size)

        # as #
        yield update_validation_meters
        # __exit__ #
        # Copy values to validation fields
        for loss_name in self.loss_names:
            loss_val_name = 'loss_' + loss_name + '_val'
            loss = loss_meters[loss_name].avg
            setattr(self, loss_val_name, loss)
        for metric_name in self.metric_names:
            metric_val_name = 'metric_' + metric_name + '_val'
            metric = metric_meters[metric_name].avg
            setattr(self, metric_val_name, metric)
        for visual_name in self.visual_names:
            visual_val = getattr(self, visual_name)
            setattr(self, visual_name + "_val", visual_val)
        self.is_val = False  # begin working with training measures again

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_losses(self):
        errors_ret = dict()  # before python 3.6, dictionaries are not ordered and this malfunctions (use OrderedDict)
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                if self.is_val:
                    name = name + '_val'
                    errors_ret[name] = float(getattr(self, 'loss_' + name))
                else:
                    errors_ret[name] = float(self.meters[name].avg)
        return errors_ret

    def get_current_metrics(self):
        metric_ret = dict()
        for name in self.metric_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                if self.is_val:
                    name = name + '_val'
                    metric_ret[name] = float(getattr(self, 'metric_' + name))
                else:
                    metric_ret[name] = float(self.meters[name].avg)
        return metric_ret

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = dict()
        for name, kind in zip(self.visual_names, self.visual_types):
            if isinstance(name, str):
                visual_ret[name + "_" + kind] = getattr(self, name)
        return visual_ret

    # get images paths
    def get_visual_paths(self):
        return self.visual_paths

    # save models to the disk
    def save_networks(self, epoch):
        for name in self.module_names:
            if isinstance(name, str):
                self.model_tag = f'{epoch}_net'
                save_filename = f'{self.model_tag}_{name}.pth'
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, epoch):
        for name in self.module_names:
            if isinstance(name, str):
                self.model_tag = f'{epoch}_net'
                load_filename = f'{self.model_tag}_{name}.pth'
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print(f'loading the model from {load_path}')
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=torch.device('cpu'))
                #state_dict = dict((key, value.cuda(device=self.device)) for key,value in state_dict.items())
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                # if not self.opt.gpu_ids:
                #    state_dict = {key[6:]: value for key, value in
                #                    state_dict.items()}  # remove data_parallel's "module."
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.module_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    utils.summary(net, (3, self.opt.fine_size, self.opt.fine_size), self.device.type)
                    # print(net)
                print(f'[Network {name}] Total number of parameters : {num_params/1e6:.3f} M')
        print('-----------------------------------------------')

    # set requies_grad=False to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # shares modules memory, so that models can be used in multiprocessing
    def share_memory(self):
        for module_name in self.module_names:
            net = getattr(self, 'net' + module_name)
            if net is not None:
                net.share_memory()

