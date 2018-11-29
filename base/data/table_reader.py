import re
import csv
import math
from collections import OrderedDict, defaultdict
from itertools import product

__author__ = "andreachatrian"


class TableReader:

    def __init__(self, opt):
        self.opt = opt
        self.read_table_files = []
        self.field_names = self.opt.data_fields
        self.field_types = self.opt.field_datatypes
        ids = self.opt.ids if type(self.opt.ids) is list else self.opt.ids.split(',')
        self.data = MultipleIdMap(ids)
        self.pivots = []
        self.merged = False

    def read_data(self, filenames):
        self.read_table_files.extend(filenames)
        for filename in filenames:
            with open(filename, 'r') as metadata_file:
                dialect = csv.Sniffer().sniff(metadata_file.read(30))  # detects delimiter and others - TODO: this is fiddly, change with delimeter specification ?
                reader = csv.DictReader(metadata_file, dialect=dialect)  # , delimiter='\t')  first row must contain text describing the fields
                for row in reader:
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
                                raise NotImplementedError("Field '{}' is not supported".format(field_type))
                        except KeyError:
                            value = '' if field_type == 'text' else math.nan

                        if not hasattr(self, field_name ):
                            setattr(self, field_name , [value])
                        else:
                            field_values = getattr(self, field_name )
                            field_values.append(value)
                            setattr(self, field_name, field_values)
        self.merged = False

    def merge_by_pivots(self, pivots):
        # merge entries using specific pivot fields
        if len(self.read_table_files) > 1:
            loop_through_read_data = zip(getattr(self, field_name) for field_name in self.field_names)
            for values in loop_through_read_data:
                good_ids = {key: value for key, value in zip(self.field_names, values) if key in pivots and value}
                self.data.link_id_group(**good_ids)
                pivot = self.opt.sample_id_name
                try:
                    for i, field_name in self.field_names:
                        self.data[good_ids[pivot]][i] = self.data[good_ids[pivot]][i] or values[i]  # update if empty
                except KeyError:
                    self.data[good_ids[pivot]] = values
        self.merged = True

    def get_pivot_data(self, pivot_id):
        pivot_id_values = list(self.data.id_data[pivot_id].keys())
        return {self.data.data[id0] for id0 in pivot_id_values}



    def reset(self):
        self.merged = False

    def __getitem__(self, item):
        return self.data[item]


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
