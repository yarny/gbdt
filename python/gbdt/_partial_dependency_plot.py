def plot_partial_dependency(forest, data, feature, x, x0=None, log_scale=False, color='blue'):
    """Plots partial dependency graph.
       Plot forest(instance|f=x) - forest(instance|f=x0) vs x, where instance is
       a feature vector and instance|f=x represents the resulting feature vector after
       setting f to x.

       Inputs:
         forest: the forest model.
         data: the sample of data to plot the graph with.
         feature: the feature.
         x: the values to perturb the feature with.
         x0: the base feature value to compare with.
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import math
    except ImportError:
        raise ImportError('Please install matplotlib and numpy.')

    from ._data_store import DataLoader
    from libgbdt import StringColumn, BucketizedFloatColumn

    def replace_float_feature(data, feature, v):
        n = len(data)
        data.erase(feature)
        data.add_bucketized_float_col(feature, [v] * n)

    def replace_string_feature(data, feature, v):
        n = len(data)
        data.erase(feature)
        data.add_string_col(feature, [v] * n)

    def compute_score_stats_float(forest, data, feature, v, base_scores):
        replace_float_feature(data, feature, v)

        scores = np.array(forest.predict(data))
        score_diffs = scores - base_scores
        return (np.mean(score_diffs), np.std(score_diffs))

    def compute_score_stats_categorical(forest, data, feature, v, base_scores):
        replace_string_feature(data, feature, v)

        scores = np.array(forest.predict(data))
        score_diffs = scores - base_scores
        return (np.mean(score_diffs), np.std(score_diffs))

    def plot_float_feature(forest, data, feature, x, x0, log_scale):
        x0 = float('nan') if x0 is None else x0

        replace_float_feature(data, feature, x0)
        base_scores = np.array(forest.predict(data))

        x = sorted(x)
        score_stats = [compute_score_stats_float(forest, data, feature, v, base_scores)
                       for v in x]
        # Use 1.96 for 95% confidence interval.
        lower_bounds =  [p[0] - 1.96 * p[1] for p in score_stats]
        upper_bounds =  [p[0] + 1.96 * p[1] for p in score_stats]

        _, ax = plt.subplots()
        ax.set_xlabel(feature)
        ax.set_ylabel('forest score delta')
        ax.set_title('Partial dependency plot')
        if log_scale:
            x = np.log(x)
            ax.set_xlabel('log({})'.format(feature))

        plt.plot(x, [p[0] for p in score_stats], color=color)
        plt.fill_between(x, lower_bounds, upper_bounds, alpha=.3, color=color)
        plt.plot(x, [0] * len(x), color='k', linestyle='--')
        plt.show()

    def plot_categorical_features(forest, data, feautre, x, x0):
        BAR_WIDTH = 0.5

        x0 = x[0] if x0 is None else x0
        replace_string_feature(data, feature, x0)
        base_scores = np.array(forest.predict(data))

        score_stats = [compute_score_stats_categorical(forest, data, feature, v, base_scores)
                       for v in x]
        means = np.array([a[0] for a in score_stats])
        stds = np.array([a[1] for a in score_stats])
        order = np.argsort(means)

        _, ax = plt.subplots()
        ax.set_xlabel(feature)
        ax.set_ylabel('forest score delta')
        ax.set_title('Partial dependency plot')
        ind = np.arange(len(x))
        ax.set_xticks(ind + BAR_WIDTH / 2)
        ax.set_xticklabels([x[i] for i in order])

        ax.bar(ind, means[order], BAR_WIDTH, color=color, yerr=stds[order])
        plt.plot(ind, [0] * len(x), color='k', linestyle='-')
        plt.show()

    data = data.copy()
    if feature not in data:
        raise ValueError("Unknown feature '{}'".format(feature))

    if type(data[feature]) is BucketizedFloatColumn:
        plot_float_feature(forest, data, feature, x, x0, log_scale)
    elif type(data[feature]) is StringColumn:
        plot_categorical_features(forest, data, feature, x, x0)
    else:
        raise ValueError("Unsupported feature type {}.".format(type(data[feature])))
