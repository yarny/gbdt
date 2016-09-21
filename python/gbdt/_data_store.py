from libgbdt import DataStore as _DataStore
from libgbdt import BucketizedFloatColumn, StringColumn, RawFloatColumn
from operator import itemgetter

class DataStore:
    def __init__(self, data_store):
        self._data_store = data_store

    def __len__(self, ):
        return len(self._data_store)

    def __getitem__(self, k):
        if k not in self._data_store:
            raise ValueError("Column '{}' cannot be found.".format(k))
        try:
            return self._data_store.get_bucketized_float_col(k)
        except:
            try:
                return self._data_store.get_string_col(k)
            except:
                return self._data_store.get_raw_float_col(k)

    def __contains__(self, name):
        return name in self._data_store

    def cols(self):
        return self._data_store.cols()

    def bucketized_float_cols(self):
        return self._data_store.bucketized_float_cols()

    def raw_float_cols(self):
        return self._data_store.raw_float_cols()

    def string_cols(self):
        return self._data_store.string_cols()

    def add_bucketized_float_col(self, name, values):
        """Adds a bucketized float column into data store.
           Input:
             - name: the column name.
             - values: a list of floats
        """
        self._data_store.add_bucketized_float_col(name, values)

    def add_raw_float_col(self, name, values):
        """Adds a raw float column into data store.
           Input:
             - name: the column name.
             - values: a list of floats
        """
        self._data_store.add_raw_float_col(name, values)

    def add_string_col(self, name, values):
        """Adds a string float column into data store.
           Input:
             - name: the column name.
             - values: a list of strings
        """
        self._data_store.add_string_col(name, values)

    def erase(self, name):
        """Erases column from data store."""
        self._data_store.remove_col(name)

    def iteritems(self):
        for k in self.cols():
            yield (k, self[k])

    def copy(self):
        d = _DataStore()
        for key, value in self.iteritems():
            column_type = str(type(value))
            if type(value) is StringColumn:
                d.add_string_col(key, list(value))
            elif type(value) is BucketizedFloatColumn:
                d.add_bucketized_float_col(key, map(itemgetter(0), value))
            elif type(value) is RawFloatColumn:
                d.add_raw_float_col(key, list(value))

        return DataStore(d)

class DataLoader:
    @staticmethod
    def from_tsvs(tsvs, bucketized_float_cols=[], string_cols=[], raw_float_cols=[]):
        """Loads data from tsvs.
           Inputs:
             tsvs: Blocks of tsvs, among which only the first contains header.
             bucketized_float_cols: Float columns that will be bucketized. All features will be bucketized.
             string_cols: String cols.
             raw_float_cols: Float columns that are loaded raw. Target columns are usually not bucketized.
        """
        d = _DataStore()
        d.load_tsv(tsvs,
                   bucketized_float_cols=bucketized_float_cols,
                   string_cols=string_cols,
                   raw_float_cols=raw_float_cols)
        return DataStore(d)

    @staticmethod
    def from_dict(bucketized_float_cols={}, string_cols={}, raw_float_cols={}):
        """Loads data from dict of columns.
             bucketized_float_cols: Float columns that will be bucketized. All features will be bucketized.
             string_cols: String cols.
             raw_float_cols: Float columns that are loaded raw. Target columns are usually not bucketized.
        """
        d = _DataStore()
        for key, value in bucketized_float_cols.iteritems():
            d.add_bucketized_float_col(key, value)
        for key, value in string_cols.iteritems():
            d.add_string_col(key, value)
        for key, value in raw_float_cols.iteritems():
            d.add_raw_float_col(key, value)

        return DataStore(d)
