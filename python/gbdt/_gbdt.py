from libgbdt import DataStore
from libgbdt import Forest
from libgbdt import train as __train_internal__
from libgbdt import init_logging

def train(data_store,
          config,
          y=[],
          float_features=[],
          cat_features=[],
          w=[],
          base_forest=None,
          random_seed=1234567,
          num_threads=16):
    """
    train gbdt model.
    Inputs: data_store: The Data Store.
            config: the config params. See https://github.com/yarny/gbdt/blob/master/src/proto/config.proto for all params.
            y: The target
            float_features: Continuous features.
            cat_features: Categorical features.
            w: The sampling weight.
            base_forest: The base forest based on which the training will continue.
            num_threads: The number of threads to run training with.
    Outputs: The forest.
    """
    try:
        import json
    except ImportError:
        raise ImportError('Please install json.')

    config['float_feature'] = float_features
    config['categorical_feature'] = cat_features
    if 'example_sampling_rate' not in config:
        config['example_sampling_rate'] = 1
    if 'feature_sampling_rate' not in config:
        config['feature_sampling_rate'] = 1

    return __train_internal__(data_store,
                              y=y,
                              w=w,
                              config=json.dumps(config),
                              base_forest=base_forest,
                              random_seed=random_seed,
                              num_threads=num_threads)

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
        d = DataStore()
        d.load_tsv(tsvs,
                   bucketized_float_cols=bucketized_float_cols,
                   string_cols=string_cols,
                   raw_float_cols=raw_float_cols)
        return d

    @staticmethod
    def from_dict(bucketized_float_cols={}, string_cols={}, raw_float_cols={}):
        """Loads data from dict of columns.
             bucketized_float_cols: Float columns that will be bucketized. All features will be bucketized.
             string_cols: String cols.
             raw_float_cols: Float columns that are loaded raw. Target columns are usually not bucketized.
        """
        d = DataStore()
        for key, value in bucketized_float_cols.iteritems():
            d.add_bucketized_float_col(key, value)
        for key, value in string_cols.iteritems():
            d.add_string_col(key, value)
        for key, value in raw_float_cols.iteritems():
            d.add_raw_float_col(key, value)
        return d
