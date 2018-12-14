import os
import re
import csv
import math
import pandas as pd
import numpy as np
from numbers import Integral
from collections import OrderedDict, defaultdict
from itertools import product

__author__ = "andreachatrian"


class TableReader:

    def __init__(self, opt):
        self.opt = opt
        self.read_table_files = []
        self.field_names = self.opt.data_fields.split(',')
        self.field_types = {name: dtype for name, dtype in zip(self.field_names, self.opt.field_datatypes.split(','))}
        ids = self.opt.ids if type(self.opt.ids) is list else self.opt.ids.split(',')
        self.ids = ids
        self.table_names = []
        self.table_types = dict()
        self.pivoted = dict()

    def read_singleentry_data(self, filenames, name='data', replace_dict=None):
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
                            value = re.sub(r'\W+', '', value)  # strip punctuation
                            if replace_dict and value in replace_dict.keys():
                                value = replace_dict[value]
                            elif field_type == 'text':
                                value = value
                            elif field_type == 'number':
                                value = value if value.isdigit() else math.nan  # using nan to denote missing fields https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html
                            else:
                                raise NotImplementedError("Field type '{}' is not supported".format(field_type))
                        except KeyError:
                            value = '' if field_type == 'text' else math.nan
                        datamat[-1].append(value)  # compose final row
        datamat = np.array(datamat, dtype=object)  # allows storage of bools strings and numbers in one array
        data = pd.DataFrame(data=datamat, columns=self.field_names)
        setattr(self, name, data)
        self.table_names.append(name)
        self.table_types[name] = 'single_entry'

    def read_matrix_data(self, filenames, yfield, xfield, data_type=int, name='datamat'):
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
            setattr(self, name, datamat)
            self.table_names.append(name)
            self.table_types[name] = 'matrix'
            self.pivoted[name] = True  # matrix tables are already pivoted by x or y

    def pivot_data(self, name, pivot_id=None):
        pivot_id = pivot_id or self.opt.sample_id_pivot
        data = getattr(self, name)
        if self.table_types[name] == 'single_entry':
            try:
                pivot_values = data[pivot_id]
            except KeyError as err:
                raise ValueError(f"'{pivot_id}' is not a recorded data field") from err
            fields_minus_pivot = list(data.columns.values)
            pivot_loc = fields_minus_pivot.index(pivot_id)
            fields_minus_pivot.remove(pivot_id)
            datamat = data.values[:, list(range(pivot_loc)) + list(range(pivot_loc+1, data.values.shape[1]))]
            pivoted_data = pd.DataFrame(index=pivot_values, data=datamat, columns=fields_minus_pivot)
        elif self.table_types[name] == 'matrix':
            pivoted_data = data  # no need to pivot for matrix table
        else:
            raise ValueError(f"Invalid table type '{self.table_types[name]}'")
        setattr(self, name, pivoted_data)
        self.pivoted[name] = True
        return pivoted_data

    def get_merge_tables(self, name0, name1, merge_se_id=None, merge_mat_dir=('y', 'y')):
        assert all(mdir == 'x' or mdir == 'y' for mdir in merge_mat_dir)
        if not self.pivoted[name0] and self.table_types[name0] == 'single_entry':
            self.pivot_data(name0, merge_se_id)
        if not self.pivoted[name1] and self.table_types[name1] == 'single_entry':
            self.pivot_data(name0, merge_se_id)
        table0 = getattr(self, name0)
        table1 = getattr(self, name1)
        if merge_mat_dir[0] == 'x' and self.table_types[name0] == 'matrix':
                table0 = table0.transposed()
        if merge_mat_dir[1] == 'x' and self.table_types[name1] == 'matrix':
                table1 = table1.transposed()

        def merge_se_tables(se, se0):
            return se.join(se0)  # joins two DataFrames with same indices

        def merge_se_mat_tables(se, mat):
            return se.join(mat)

        def merge_mat_tables(mat, mat0):
            return mat.join(mat0)

        table_type_err = ValueError("Invalid table type")
        if self.table_types[name0] == 'single_entry':
            if self.table_types[name1] == 'single_entry':
                merged_data = merge_se_tables(table0, table1)
            elif self.table_types[name1] == 'matrix':
                merged_data = merge_se_mat_tables(table0, table1)
            else:
                raise table_type_err
        elif self.table_types[name0] == 'matrix':
            if self.table_types[name1] == 'single_entry':
                merged_data = merge_se_mat_tables(table1, table0)
            elif self.table_types[name1] == 'matrix':
                merged_data = merge_mat_tables(table0, table1)
            else:
                raise table_type_err
        else:
            raise table_type_err

        self.merged_data = merged_data
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
        assert id_type0 in self.id_types and id_type1 in self.id_types; "pivots not among specified ones"
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
