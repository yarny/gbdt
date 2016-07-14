import sys
sys.path.append('../../bazel-bin/src/python')

from libgbdt import DataStore
from libgbdt import Forest
from libgbdt import train as __train_internal__
import json

def train_gbdt(data_store, y, config,
               float_features=[],
               cat_features=[], w=[],
               base_forest=None,
               random_seed=123012,
               num_threads=16):
    """
    train gbdt model.
    Inputs: data_store: The Data Store.
            float_features: Continuous features
            cat_features: Categorical features
            y: The target
            config: the config params. See https://github.com/yarny/gbdt/blob/master/src/proto/config.proto for all params.
            w: the sampling weight
    Outputs: The forest.
    """
    assert data_store is not None
    assert data_store.num_rows() == len(y), '{0} vs {1}'.format(data_store.num_rows(), len(y))
    assert len(w) == 0 or data_store.num_rows() == len(w), '{0} vs {1}'.format(data_store.num_rows(), len(w))
    assert len(float_features) + len(cat_features) > 0

    config['float_feature'] = float_features
    config['categorical_feature'] = cat_features

    return __train_internal__(data_store, y,
                              w=w,
                              config=json.dumps(config),
                              base_forest=base_forest,
                              random_seed=random_seed,
                              num_threads=num_threads)
