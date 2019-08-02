import importlib
from torch.utils.data import DataLoader
from .base_dataset import BaseDataset
from . import table_reader


def find_dataset_using_name(dataset_name, task_name):
    # Given the option --dataset_name [datasetname],
    # the file "data/datasetname_dataset.py"
    # will be imported.
    try:
        task_module = importlib.import_module(task_name)
        dataset_filename = task_name + ".data." + dataset_name + "_dataset"
        datasetlib = importlib.import_module(dataset_filename, package=task_module)
    except ModuleNotFoundError as err1:
            try:
                # if module not found, attempt to load from base
                task_name = 'base'
                task_module = importlib.import_module(task_name)
                dataset_filename = task_name + ".data." + dataset_name + "_dataset"
                datasetlib = importlib.import_module(dataset_filename, package=task_module)
            except ModuleNotFoundError:
                if not err1.args:
                    err1.args = ('',)
                err1.args = err1.args + (f"{task_name}.data contains no file '{dataset_name}.py'",)
                raise err1
    except ImportError as importerr:
        if not importerr.args:
            importerr.args = ('',)
        importerr.args = importerr.args + (f"Module {task_name} not found.",)
        raise

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.

    def is_subclass(subclass, superclass):
        return next(iter(subclass.__bases__)).__module__.endswith(superclass.__module__)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            if is_subclass(cls, BaseDataset) or any(is_subclass(cls_b, BaseDataset) for cls_b in cls.__bases__):
                dataset = cls

    if dataset is None:
        raise NotImplementedError("In {}.py, there should be a subclass of BaseDataset with class name that matches {} in lowercase.".format(
              dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name, task_name):
    dataset_class = find_dataset_using_name(dataset_name, task_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, validation_phase=False):
    dataset = find_dataset_using_name(opt.dataset_name, opt.task)
    if validation_phase:
        opt.phase = "val"
    instance = dataset(opt)
    print('dataset [{}] was created {}'.format(instance.name(), "(val)" if validation_phase else ''))
    return instance


def create_dataloader(dataset):
    if len(dataset) == 0:
        raise ValueError(f"Dataset {dataset.name()} is empty")
    try:
        opt = dataset.opt
    except AttributeError:
        opt = dataset.dataset.opt  # for Subset instances
    sampler = dataset.get_sampler()
    is_val = (opt.phase == "val")
    return DataLoader(dataset,
                      batch_size=opt.batch_size if not is_val else opt.val_batch_size,
                      shuffle=not is_val and sampler is None,  # if a sampler is specified, shuffle must be false
                      num_workers=opt.workers, sampler=sampler)


# utilities to organise patches
class Node:
    __slots__ = ['left', 'right', 'value', 'idx']  # to occupy less memory, as tree will have many nodes

    def __init__(self, value, idx=None):
        self.left = None
        self.right = None
        self.value = value
        self.idx = idx

    def __repr__(self):
        return f"Node({self.value})"


class Tree:
    def __init__(self, criterion=lambda value, node_value: value < node_value):
        self.root = None
        self.nodes = []
        self.criterion = criterion  # criterion to assign to the left
        # speed up leaf computations, by starting computation from last computed leaves
        # this works assuming no nodes are removed from tree
        self._computed_leaves = []

    def get_root(self):
        return self.root

    def add(self, value):
        if self.root is None:
            new_node = Node(value, idx=len(self.nodes))
            self.root = new_node
            self.nodes.append(new_node)
        else:
            self.add_to_node(value, self.root)

    def add_to_node(self, value, node, left=None):
        if (left is not None and left) or self.criterion(value, node.value):
            if node.left is not None:
                self.add_to_node(value, node.left)
            else:
                new_node = Node(value, idx=len(self.nodes))
                node.left = new_node
                self.nodes.append(new_node)
        else:
            if node.right is not None:
                self.add_to_node(value, node.right)
            else:
                new_node = Node(value, idx=len(self.nodes))
                node.right = new_node
                self.nodes.append(new_node)

    def find(self, value):
        if self.root is not None:
            return self._find(value, self.root)
        else:
            return None

    def _find(self, value, node):
        if value == node.value:
            return node
        elif value < node.value and node.leaves is not None:
            self._find(value, node.leaves)
        elif value > node.value and node.right is not None:
            self._find(value, node.right)

    def delete_tree(self):
        # garbage collector will do this for us.
        self.root = None

    def get_leaves(self):
        leaves = []
        current_nodes = [self.root] if not self._computed_leaves else self._computed_leaves
        while len(current_nodes) > 0:
            next_nodes = []
            for node in current_nodes:
                if node.left is None and node.right is None:
                    leaves.append(node)
                    continue
                if node.left is not None:
                    next_nodes.append(node.left)
                if node.right is not None:
                    next_nodes.append(node.right)
            current_nodes = next_nodes
        self._computed_leaves = leaves
        return leaves

    def print_tree(self):
        if self.root is not None:
            self._print_tree(self.root)

    def _print_tree(self, node):
        if node is not None:
            self._print_tree(node.left)
            print(str(node.value) + ' ')
            self._print_tree(node.right)

    def __iter__(self):
        r"""Breadth first traversal"""
        current_nodes = [self.root]
        while len(current_nodes) > 0:
            next_nodes = []  # nodes in this list are labelled as 'discovered'
            for node in current_nodes:
                yield node.value
                if node.left is not None:
                    next_nodes.append(node.left)
                if node.right is not None:
                    next_nodes.append(node.right)
            current_nodes = next_nodes

    # could update leaves whenever new node is added to save time, rather than traversing tree each time
    # or copy algorithm from https://github.com/joowani/binarytree/blob/master/binarytree/__init__.py
