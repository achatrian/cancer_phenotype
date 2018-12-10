import os
from base.data.base_dataset import BaseDataset
from base.data import find_dataset_using_name


class TCGADataset(BaseDataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # create datasets for whole slide images and table data
        wsi_dataset = find_dataset_using_name('wsi', 'base')
        self.wsi_dataset = wsi_dataset(opt)
        table_dataset = find_dataset_using_name('table', 'base')
        self.table_dataset = table_dataset(opt)
        # check that data in tables and wsi's match and change table ids order to match wsi file order
        assert len(wsi_dataset.files) == len(table_dataset)
        wsi_dataset.match_table_order(self.table_dataset.data_id)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        wsi_dataset = find_dataset_using_name('wsi', 'base')
        parser = wsi_dataset.modify_commandline_options(parser, is_train)
        table_dataset = find_dataset_using_name('table', 'base')
        parser = table_dataset.modify_commandline_options(parser, is_train)
        return parser

    def __getitem__(self, item):
        wsi = self.wsi_dataset[item]
        table_entry = self.table_dataset[item]
        return wsi, table_entry



# TODO option for using slides only, but then don't know what data I am running on











