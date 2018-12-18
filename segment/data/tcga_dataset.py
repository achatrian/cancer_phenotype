import os
import re
from base.data.base_dataset import BaseDataset
from base.data import find_dataset_using_name
from base.data.table_reader import TableReader


class TCGADataset(BaseDataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # create datasets for whole slide images
        wsi_dataset = find_dataset_using_name('wsi', 'base')
        self.wsi_dataset = wsi_dataset(opt)

        # read metadata
        field_names = self.opt.data_fields.split(',')
        datatypes = self.opt.field_datatypes.split(',')
        self.sample = TableReader(field_names, datatypes)
        self.cna = TableReader(field_names, datatypes)
        self.data = None
        wsi_replacements = {
            'FALSE': False,
            'TRUE': True,
            'released': True
        }
        self.sample.read_singleentry_data(self.opt.wsi_tablefile, replace_dict=wsi_replacements)
        self.sample.index_data(index=self.opt.sample_index)
        self.cna.read_matrix_data(self.opt.cna_tablefile, yfield='Hugo_Symbol', xfield=(0, 2))
        self.cna.index_data(index='y')
        self.sample.data.query("is_ffpe == True", inplace=True)  # remove all slides that are not FFPE

        # discard files with no ffpe
        self.wsi_dataset.setup(good_files=self.sample.data.index.values)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # whole slide images
        wsi_dataset = find_dataset_using_name('wsi', 'base')
        parser = wsi_dataset.modify_commandline_options(parser, is_train)
        # metadata
        parser.add_argument('--wsi_tablefile', type=str, default='', help='file with wsi metadata')
        parser.add_argument('--cna_tablefile', type=str, default='', help='file with cna data')
        data_fields = ('case_submitter_id', 'sample_id', 'case_id', 'sample_submitter_id', 'is_ffpe', 'sample_type',
                       'state', 'oct_embedded')
        fields_datatype = tuple('text' for field_name in data_fields)
        data_fields = ','.join(data_fields)
        fields_datatype = ','.join(fields_datatype)
        parser.add_argument('--data_fields', type=str, default=data_fields, help='information to store from table')
        parser.add_argument('--field_datatypes', type=str, default=fields_datatype, help='type of stored information')
        parser.add_argument('--sample_index', type=str, default='sample_submitter_id',
                            help='Slide specific identifier that is used to organize the metadata entries - must be 1 per slide')
        return parser

    def __getitem__(self, item):
        image_data = self.wsi_dataset[item]
        sample_id = os.path.basename(image_data['file_name'])
        sample_id = ''.join(sample_id.split('-')[0:4])
        cna_data = self.cna.data[sample_id]
        label = question(cna_data)
        return {'input': image_data['tile'], 'target': label}


def question(data):
    label = data['PTEN']
    return label











