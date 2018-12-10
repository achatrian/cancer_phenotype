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

    def read_singleentry_data(self, filenames, nsniff=30, name='data'):
        """
        Rows are datapoints
        For reading many fields - single entry per field - table
        NB: fields might have duplicates across different datapoints (rows)
        """
        if type(filenames) is not list:
            filenames = [filenames]
        self.read_table_files.extend(filenames)
        datamat = []
        for filename in filenames:
            with open(filename, 'r') as metadata_file:
                dialect = csv.Sniffer().sniff(metadata_file.read(nsniff))  # detects delimiter and others - TODO: this is fiddly, change with delimeter specification ?
                reader = csv.DictReader(metadata_file, dialect=dialect)  # , delimiter='\t')  first row must contain text describing the fields
                for row in reader:
                    datamat.append([])
                    for field_name in self.field_names:
                        field_type = self.field_types[field_name]
                        try:
                            value = row[field_name]
                            value = re.sub(r'\W+', '', value)  # strip punctuation
                            if field_type == 'text':
                                value = value
                            elif field_type == 'number':
                                value = value if value.isdigit() else math.nan  # using nan to denote missing fields https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html
                            else:
                                raise NotImplementedError("Field type '{}' is not supported".format(field_type))
                        except KeyError:
                            value = '' if field_type == 'text' else math.nan
                        datamat[-1].append(value)  # compose final row
        datamat = np.array(datamat)
        data = pd.DataFrame(data=datamat, columns=self.field_names)
        setattr(self, name, data)
        self.table_names.append(name)
        self.table_types[name] = 'single_entry'

    def read_matrix_data(self, filenames, yfield, xfield, name='datamat', nsniff=30):
        """
        Reading data in matrix format, i.e. there is one entry per x field per y field |x|*|y|
        """
        if type(filenames) is not list:
            filenames = [filenames]
        for filename in filenames:
            with open(filename, 'r') as metadata_file:
                dialect = csv.Sniffer().sniff(metadata_file.read(nsniff))  # detects delimiter and others - TODO: this is fiddly, change with delimeter specification ? could use csv.excel_tab
                reader = csv.reader(metadata_file, dialect=dialect)  # , delimiter='\t')  first row must contain text describing the fields
                reader_iter = iter(reader)
                ycoord_xfield, xcoord_yfield, i = math.nan, math.nan, 0
                while math.isnan(ycoord_xfield) or math.isnan(xcoord_yfield):
                    if i > 4:
                        raise ValueError('Unable to find table entry ids within first 4 rows')
                    # find row and column where data ids are
                    row = next(reader_iter)
                    if isinstance(yfield, str):
                        xcoord_yfield = next(i for i, field in enumerate(first_row) if field == yfield)
                    elif isinstance(yfield, Integral):
                        xcoord_yfield = yfield
                    else:
                        raise ValueError(f"Invalid yfield object type: {type(yfield)}")
                    if isinstance(xfield, str):
                        ycoord_xfield = next(i for i, field in enumerate(first_row) if field == xfield)
                    elif isinstance(yfield, Integral):
                        ycoord_xfield = xfield
                    else:
                        raise ValueError(f"Invalid xfield object type: {type(xfield)}")
                    i += 1

                # once structure of matrix is understood, read data in
                reader_iter = iter(reader)
                first_row = next(reader_iter)
                yfield_entries = []
                datamat = []
                for j, row in enumerate(reader_iter):
                    if row[xcoord_yfield]:
                        # append indices
                        yfield_entries.append(row[xcoord_yfield])
                    if j < ycoord_xfield:
                        continue
                    if j == xcoord_yfield:
                        xfield_entries = row
                    datamat.append(row)
            datamat = pd.DataFrame(data=datamat, index=yfield_entries, columns=xfield_entries)
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
            fields_minus_pivot = data.columns.values
            pivot_loc = fields_minus_pivot.index(pivot_id)
            fields_minus_pivot.remove(pivot_id)
            datamat = data.values[:, list(range(pivot_loc)) + list(range(pivot_loc+1, data.values.shape[1]))]
            pivoted_data = pd.DataFrame(index=pivot_values, data=datamat, columns=fields_minus_pivot)
        elif self.table_types[name] == 'single_entry':
            pivoted_data = data  # no need to pivot for matrix table
        else:
            raise ValueError(f"Invalid table type {self.table_types[name]}")
        setattr(self, name, pivoted_data)
        self.pivoted[name] = True
        return pivoted_data

    # def __getitem__(self, item):
    #     return self.data[item]

    def get_merge_tables(self, name0, name1, merge_se_id=None, merge_mat_dir='y'):
        assert merge_mat_dir == 'x' or merge_mat_dir == 'y'
        if not self.pivoted[name0] and self.table_types[name0] == 'single_entry':
            self.pivot_data(name0, merge_se_id)
        if not self.pivoted[name1] and self.table_types[name1] == 'single_entry':
            self.pivot_data(name0, merge_se_id)
        table0 = getattr(self, name0)
        table1 = getattr(self, name1)
        if merge_mat_dir == 'x':
            if self.table_types[name0] == 'matrix':
                table0 = table0.transposed()
            if self.table_types[name1] == 'matrix':
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
