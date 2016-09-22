from libgbdt import DataStore as _DataStore
from libgbdt import BucketizedFloatColumn, StringColumn, RawFloatColumn

class DataStore:
    def __init__(self, data_store):
        self._data_store = data_store

    def __len__(self):
        return len(self._data_store)

    def _get_col(self, k):
        if k not in self._data_store:
            raise ValueError("Column '{}' cannot be found.".format(k))
        type = self._data_store.get_column_type(k)

        if type == 'bucketized_float':
            return self._data_store.get_bucketized_float_col(k)
        elif type == 'string':
            return self._data_store.get_string_col(k)
        else:
            return self._data_store.get_raw_float_col(k)

    def __getitem__(self, k):
        if type(k) is int or type(k) is long:
            return self.slice([k])
        if type(k) is tuple and len(k) == 2:
            return self.slice(xrange(k[0], k[1]))
        elif type(k) is list:
            return self.slice(k)
        return self._get_col(k)

    def __contains__(self, name):
        return name in self._data_store

    def cols(self):
        return self._data_store.cols()

    def keys(self):
        return self.cols()

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
        self._data_store.erase(name)

    def iteritems(self):
        for k in self.cols():
            yield (k, self[k])

    def slice(self, index):
        d = _DataStore()
        for key, value in self.iteritems():
            if type(value) is StringColumn:
                d.add_string_col(key, [value[i] for i in index])
            elif type(value) is BucketizedFloatColumn:
                d.add_bucketized_float_col(key, [value[i] for i in index])
            elif type(value) is RawFloatColumn:
                d.add_raw_float_col(key, [value[i] for i in index])
        return DataStore(d)

    def copy(self):
        return self.slice(range(len(self._data_store)))

    def to_df(self):
        try:
            import pandas
        except ImportError:
            raise ImportError('Please install pandas.')

        return pandas.DataFrame(dict([(k, list(v)) for k, v in self.iteritems()]))

    def __repr__(self):
        return self._data_store.__repr__()

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

    @staticmethod
    def from_df(df, type_overrides={}):
        """Loads data store from pandas DataFrame.
             - df: By default, we will load all numeric type as BucketizedFloatColumn
                   and all other type as string_cols unless instructed by type_overrides.
             - type_overrides: A dict of col->type which allows us to override the types of
                   each column. Valid override types are string, bucketized_float and float (raw_float).
        """
        try:
            import numpy
            import pandas
        except ImportError:
            raise ImportError('Please install numpy and pandas.')

        d = _DataStore()
        raw_float_cols = set([k for k, v in type_overrides.iteritems()
                              if v == 'float' or v == 'raw_float'])
        string_cols = set([k for k, v in type_overrides.iteritems()
                           if v == 'string'])

        for col in df.select_dtypes(include=[numpy.number]).keys():
            if col in raw_float_cols:
                d.add_raw_float_col(col, list(df[col]))
            elif col not in string_cols:
                d.add_bucketized_float_col(col, list(df[col]))

        for col in df.select_dtypes(exclude=[numpy.number]).keys():
            d.add_string_col(col, list(df[col]))

        return DataStore(d)
