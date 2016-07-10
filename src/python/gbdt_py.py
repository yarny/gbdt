import sys
sys.path.append('/home/ubuntu/repos/gbdt/bazel-bin/src/python')

from gbdt import DataStore
from gbdt import Forest

d = DataStore()

d.load_tsv(tsvs=["/home/ubuntu/repos/gbdt/examples/benchm-ml/train-0.1m.tsv"],
           binned_float_cols=["DepTime", "Distance"],
           raw_float_cols=["dep_delayed_15min_float"],
           string_cols=["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"])
print d

f = Forest(open("/home/ubuntu/repos/gbdt/examples/benchm-ml/models/forest.logloss.json").read())
