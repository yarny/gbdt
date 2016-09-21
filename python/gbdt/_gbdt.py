from libgbdt import Forest
from libgbdt import train as _train
from libgbdt import init_logging
from ._forest import Forest

def train(data_store,
          config,
          y=[],
          features=[],
          w=[],
          base_forest=None,
          random_seed=1234567,
          num_threads=16):
    """
    Trains a gbdt model.
    Inputs: data_store: The Data Store.
            config: the config params. See https://github.com/yarny/gbdt/blob/master/src/proto/config.proto for all params.
            y: The target
            features: the list of features.
            w: The sampling weight.
            base_forest: The base forest upon which the training will continue.
            random_seed: the random seed.
            num_threads: The number of threads to run training with.
    Outputs: The forest.
    """
    try:
        import json
    except ImportError:
        raise ImportError('Please install json.')

    feature_set = set(features)
    float_features = [col for col in data_store.bucketized_float_cols() if str(col) in feature_set]
    cat_features = [col for col in data_store.string_cols() if col in feature_set]
    unfound_features = feature_set - (set(float_features) | set(cat_features))
    if len(unfound_features) > 0:
        raise ValueError('Failed to find feature {} in data store.'.format(unfound_features))

    config['float_feature'] = float_features
    config['categorical_feature'] = cat_features
    if 'example_sampling_rate' not in config:
        config['example_sampling_rate'] = 1
    if 'feature_sampling_rate' not in config:
        config['feature_sampling_rate'] = 1

    return Forest(_train(data_store._data_store,
                         y=y,
                         w=w,
                         config=json.dumps(config),
                         base_forest=base_forest if base_forest is None else base_forest._forest,
                         random_seed=random_seed,
                         num_threads=num_threads))
