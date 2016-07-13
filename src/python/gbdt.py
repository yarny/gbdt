import sys
sys.path.append('/home/ubuntu/repos/gbdt/bazel-bin/src/python')

from libgbdt import DataStore
from libgbdt import Forest
from libgbdt import train as train_internal
import json

def construct_config_from_params(params):
    config = {}
    config['tree_config'] = {}
    config['tree_config']['num_iterations'] = params['num_iterations']
    config['tree_config']['num_leaves'] = params['num_leaves']
    config['loss_func_config']['loss_func'] = params['loss']

def train_gbdt(data_store, float_features, cat_features, y, params, w=[]):
    """
    train gbdt model.
    Inputs: data_store
            float_features: continuous features
            cat_features: categorical features
            y: the target
            params: the parameters
            w: the sampling weight
    """
    assert(data_store, 'DataStore shouldn\'t be null.')
    assert(data_store->num_rows() == len(y),
        'The lengths of data_store and y don\'t match (%d vs. %d)' %
        (data_store->num_rows(), len(y)))
    assert(len(w) == 0 or data_store->num_rows() == len(w),
           'The lengths of data_store and w don\'t match (%d vs. %d)' %
           (data_store->num_rows(), len(w)))
    assert(len(float_features) + len(cat_features) > 0,
           'Feature set is empty.')

    params['float_features'] = float_features
    params['categoriccal_features'] = cat_features


if __name__ == __main__:
    d = DataStore()
    d.load_tsv(tsvs=["/home/ubuntu/repos/gbdt/examples/benchm-ml/train-0.1m.tsv"],
               binned_float_cols=["DepTime", "Distance"],
               raw_float_cols=["dep_delayed_15min_float"],
               string_cols=["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"])
    print d
    f = Forest(open("/home/ubuntu/repos/gbdt/examples/benchm-ml/models/forest.logloss.json").read())
