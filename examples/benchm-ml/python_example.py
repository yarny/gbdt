import sys
import csv
from sklearn import metrics

import gbdt

def ComputeAUC(forest, data, targets):
    predictions = forest.predict(data)
    fpr, tpr, _ = metrics.roc_curve(targets, predictions, pos_label=1)
    return metrics.auc(fpr, tpr)

def main():
    loss_func = sys.argv[1]
    float_features = ["DepTime", "Distance"]
    cat_features = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
    config = {'loss_func': loss_func,
              'num_trees': 20,
              'num_leaves': 16,
              'example_sampling_rate': 0.5,
              'feature_sampling_rate': 0.8,
              'pair_sampling_rate': 0.00001,
              'shrinkage' : 0.1}

    target_column = 'dep_delayed_15min'

    training_data = gbdt.DataLoader.from_tsvs(tsvs=["train-0.1m.tsv"],
                                              bucketized_float_cols=float_features,
                                              string_cols=cat_features + [target_column])
    training_targets = [1 if l == 'Y' else -1 for l in training_data.get_string_col(target_column)]
    forest = gbdt.train(training_data,
                        y=training_targets,
                        float_features=float_features,
                        cat_features=cat_features,
                        config=config)

    testing_data = gbdt.DataLoader.from_tsvs(tsvs=["test.tsv"],
                                             bucketized_float_cols=float_features,
                                             string_cols=cat_features + [target_column])
    testing_targets = [1 if l == 'Y' else -1 for l in testing_data.get_string_col(target_column)]

    print "\nFeature Importance:"
    print '\n'.join(['{0}\t{1}'.format(feature, imp) for feature,imp in forest.feature_importance()])
    print

    print "Training AUC =", ComputeAUC(forest, training_data, training_targets)
    print "Testing AUC =", ComputeAUC(forest, testing_data, testing_targets)

if __name__ == '__main__':
    main()
