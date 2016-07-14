import sys
import os
abspath = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{0}/../../bazel-bin/src/python'.format(abspath))

from libgbdt import DataStore
from libgbdt import Forest
from libgbdt import train as __train_internal__
import json

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
            float_features: Continuous features
            cat_features: Categorical features
            y: The target
            config: the config params. See https://github.com/yarny/gbdt/blob/master/src/proto/config.proto for all params.
            w: the sampling weight
    Outputs: The forest.
    """
    config['float_feature'] = float_features
    config['categorical_feature'] = cat_features

    return __train_internal__(data_store,
                              y=y,
                              w=w,
                              config=json.dumps(config),
                              base_forest=base_forest,
                              random_seed=random_seed,
                              num_threads=num_threads)
