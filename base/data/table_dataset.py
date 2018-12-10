from pathlib import Path
from .base_dataset import BaseDataset
from .table_reader import TableReader


class TableDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.table = TableReader(self.opt)

        # Assign wsi data
        self.table.read_singleentry_data([self.opt.wsi_tablefile], name='wsi_metadata')
        wsi_metadata = self.table.pivot_data(name='wsi_metadata', pivot_id=self.opt.sample_id_pivot)
        wsi_metadata.query("is_ffpe == 'released'", inplace=True)  # remove all slides that are not FFPE
        setattr(self.table, 'wsi_metadata', wsi_metadata)  # overwrite after trimming out non-ffpe
        # Assign CNA data
        self.table.read_matrix_data([self.opt.cna_tablefile], yfield='Hugo_Symbol', xfield=0, name='cna_data')
        self.table.pivot_data(name='cna_data')
        self.table.get_merge_tables('wsi_metadata', merge_se_id=self.opt.sample_id_pivot)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Adds arguments that determine organization of the table
        """
        parser.add_argument('--wsi_tablefile', type=str, defaut='', help='file with wsi metadata')
        parser.add_argument('--cna_tablefile', type=str, defaut='', help='file with cna data')
        data_fields = ('case_submitter_id', 'sample_id', 'case_id', 'sample_submitter_id', 'is_ffpe', 'sample_type',
        'state', 'oct_embedded')
        fields_datatype = tuple('text' for field_name in data_fields)
        data_fields = ','.join(data_fields)
        ids = ','.join(data_fields[0:3])
        fields_datatype = ','.join(fields_datatype)
        parser.add_argument('--ids', type=str, default=ids, help='ids used to organize the data')
        parser.add_argument('--data_fields', type=str, default=data_fields, help='information to store from table')
        parser.add_argument('--field_datatypes', type=str, default=fields_datatype, help='type of stored information')
        parser.add_argument('--sample_id_pivot', type=str, default='case_submitter_id', help='Slide specific identifier that is used to organize the metadata entries - must be 1 per slide')
        return parser

    def name(self):
        return "TableDataset"

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, item):
        return self.data[self.data_ids[item]]

