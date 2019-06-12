import os
import csv
import math
import pandas as pd
import numpy as np
from numbers import Integral
from collections import OrderedDict, defaultdict
from itertools import product


# FIXME kinda useless


class TableReader:

    def __init__(self, field_names=tuple(), field_datatypes=tuple()):
        assert len(field_names) == len(field_datatypes)
        self.read_table_files = []
        self.field_names = field_names
        self.field_types = {name: dtype for name, dtype in zip(self.field_names, field_datatypes)}
        self.table_type = ''
        self.indexed = False
        self.data = None

    def read_singleentry_data(self, filenames, replace_dict=None):
        """
        Rows are data-points, entries are separated by tabs (.tsv)
        For reading many fields - single entry per field - table
        NB: fields might have duplicates across different datapoints (rows)
        """
        if type(filenames) is not list:
            filenames = [filenames]
        self.read_table_files.extend(filenames)
        datamat = []
        for filename in filenames:
            with open(filename, 'r') as metadata_file:
                reader = csv.DictReader(metadata_file, delimiter='\t')  # , delimiter='\t')  first row must contain text describing the fields
                for row in reader:
                    datamat.append([])
                    for field_name in self.field_names:
                        field_type = self.field_types[field_name]
                        try:
                            value = row[field_name]
                            if field_type == 'text':
                                pass
                            elif field_type == 'number':
                                value = value if value.isdigit() else math.nan  # using nan to denote missing fields https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html
                            else:
                                raise NotImplementedError("Field type '{}' is not supported".format(field_type))
                            if replace_dict and value in replace_dict.keys():
                                value = replace_dict[value]
                        except KeyError:
                            value = '' if field_type == 'text' else math.nan
                        datamat[-1].append(value)  # compose final row
        datamat = np.array(datamat, dtype=object)  # allows storage of bools strings and numbers in one array
        data = pd.DataFrame(data=datamat, columns=self.field_names)
        self.data = data
        self.table_type = 'single_entry'

    def read_matrix_data(self, filenames, yfield, xfield, data_type=int):
        """
        Reading data in matrix format, i.e. there is one entry per x field per y field |x|*|y|
        xfield or yfield can be tuples of integers - in which case they are considered as location for the field id values -
        or strings  - in which case they are used to detect the starting row / column by the field name
        Data is assumed to start from (y + 1, x + 1). where y <- max(y_xfield, y_yfield), x <- max(x_xfield, x_yfield)
        """
        if type(filenames) is not list:
            filenames = [filenames]
        datamat = []
        for filename in filenames:
            with open(filename, 'r') as metadata_file:
                reader = csv.reader(metadata_file, delimiter='\t')  # , delimiter='\t')  first row must contain text describing the fields
                coords_xfield, coords_yfield, i = None, None, 0
                while not coords_xfield or not coords_yfield:
                    if i > 10:
                        raise ValueError('Unable to find table entry ids within first 10 rows')
                    # find row and column where data ids are
                    row = next(reader)
                    try:
                        if isinstance(yfield, str):
                            coords_yfield = (i + 1, next(j for j, field in enumerate(row) if field == yfield))
                        elif isinstance(yfield, tuple):
                            coords_yfield = yfield  # tuple given for unlabelled ids
                        else:
                            raise ValueError(f"Invalid yfield object type: {type(yfield)}")
                        if isinstance(xfield, str):
                            coords_xfield = (i + 1, next(j for j, field in enumerate(row) if field == xfield))
                        elif isinstance(xfield, tuple):
                            coords_xfield = xfield  # tuple given for unlabelled ids
                        else:
                            raise ValueError(f"Invalid xfield object type: {type(xfield)}")
                    except StopIteration:
                        pass
                    i += 1
                # once structure of matrix is understood, read data in
                ids_yfield = []
                metadata_file.seek(0, os.SEEK_SET)  # move file pointer back to the beginning of the file (SEEK_SET + 0)
                for i, row in enumerate(reader):
                    if i == coords_xfield[0]:
                        start = coords_xfield[1]
                        ids_xfield = row[start:]  # store ids for x field
                    if i >= coords_yfield[0]:
                        ids_yfield.append(row[coords_yfield[1]])  # store ids for y field
                    # coords are for ids, data is assumed to start from next entry
                    y_start = max(coords_xfield[0], coords_yfield[0])
                    if i >= y_start:
                        x_start = max(coords_xfield[1], coords_yfield[1])
                        datamat.append([data_type(entry) for entry in row[x_start:]])
            datamat = pd.DataFrame(data=np.array(datamat), index=ids_yfield, columns=ids_xfield)
            self.data = datamat
            self.table_type = 'matrix'
            self.indexed = True  # matrix tables are already indexed by x or y

    def index_data(self, index=None):
        """
        Choose id for indexing data.
        For single entry tables: the field 'index' is used as index for the data
        For matrix data: matrix is stored as is, or is transposed beforehand.
        :param name:
        :param index: column name that is used for indexing the data / direction of indexing
        :return: data indexed
        """
        index = index or self.opt.sample_id_index
        data = self.data
        if self.table_type == 'single_entry':
            try:
                index_values = data[index]
            except KeyError as err:
                raise ValueError(f"'{index}' is not a recorded data field") from err
            fields_minus_index = list(data.columns.values)
            index_loc = fields_minus_index.index(index)
            fields_minus_index.remove(index)
            datamat = data.values[:, list(range(index_loc)) + list(range(index_loc+1, data.values.shape[1]))]
            indexed_data = pd.DataFrame(index=index_values, data=datamat, columns=fields_minus_index)
        elif self.table_type == 'matrix':
            assert index in ('x', 'y')
            indexed_data = data.transpose() if index == 'y' else data
        else:
            raise ValueError(f"Invalid table type '{self.table_type}'")
        self.data = indexed_data
        self.indexed = True
        return indexed_data

    def join(self, table, criterion='all'):
        """
        Assuming table 'name0' and table 'name1' are indexed according to the same values,
        the DataFrames are merged into one.
        """
        if not table.indexed:
            raise ValueError("Unable to match index with unindexed table")
        table0 = self.data
        table1 = table.data
        # Check overlap of index values
        common_indices = set(table0.index.values).intersection(table1.index.values)
        if criterion == 'all':
            merged_data = table0.join(table1)
        elif criterion == 'minimal':
            merged_data = pd.merge(table0.loc[list(common_indices)],
                                   table1.loc[list(common_indices)],
                                   right_index=True, left_index=True)
        elif criterion == '0':
            merged_data = table0.join(table1.loc[list(common_indices)])
        elif criterion == '1':
            merged_data = table1.join(table0.loc[list(common_indices)])
        else:
            raise ValueError(f"Invalid merge criterion '{criterion}'")
        self.data = merged_data
        return merged_data


class MultipleIdMap:

    def __init__(self, id_types):
        self.id_types = id_types
        id_store = lambda: OrderedDict((id_type, '') for id_type in id_types)  # useless for python > 3.7
        # id_store = lambda: {id_type: '' for id_type in id_types}  # dictionaries are ordered since python 3.6
        self.id_data = {id_type: defaultdict(id_store) for id_type in id_types}
        #id type 1 < dict key >
        #---------- id 1 value < dict value > < dict key >
        #----------------------- id type 2 < dict value > < namedtuple key >
        #------------------------------------ id 2 value  < namedtuple value >
        self.data = dict()
        # ids -> data : different entry ids for same sample can be present, code ensures that they map to same entry

    def link_id_pair(self, id_type0, id0_value, id_type1, id1_value):
        assert id_type0 in self.id_types and id_type1 in self.id_types; "indexs not among specified ones"
        id_store0 = self.id_data[id_type0][id0_value]
        id_store0[id_type0] = id0_value
        id_store0[id_type1] = id1_value
        id_store1 = self.id_data[id_type1][id1_value]
        id_store1[id_type0] = id0_value
        id_store1[id_type1] = id1_value
        full0 = sum(1 for i in id_store0 if i)
        full1 = sum(1 for i in id_store0 if i)
        new_id_store = id_store0 if full0 >= full1 else id_store1
        self.id_data[id_type0][id0_value] = new_id_store
        self.id_data[id_type1][id1_value] = new_id_store

    def link_id_group(self, **ids):
        for ((id_type0, id0_value), (id_type1, id1_value)) in product(ids, ids):
            self.link_id_pair(id_type0, id0_value, id_type1, id1_value)

    def __setitem__(self, id_value, data):
        self.data[id_value] = data

        for id_type0 in self.id_types:
            id_store = self.id_data[id_type0][id_value]  # returned even if it doesn't exist
            for id_value1 in id_store.values():
                self.data[id_value1] = data

    def __getitem__(self, id_value):
        return self.data[id_value]
