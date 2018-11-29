from pathlib import Path
from .base_dataset import BaseDataset
from .table_reader import TableReader


class TableDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.table = TableReader(self.opt)

        data_dirpath = Path(self.opt.metadata_dir)
        files_paths = [file for file in data_dirpath.iterdir() if file.is_file()]
        self.table.read_table_files(files_paths)
        ids = self.opt.ids if type(self.opt.ids) is list else self.opt.ids.split(',')
        self.table.merge_by_pivots(ids)
        self.data = self.table.get_pivot_data(ids[0])
        self.data_ids = list(self.data.keys())

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return "TableDataset"

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, item):
        return self.data[self.data_ids[item]]
