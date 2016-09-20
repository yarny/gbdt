def plot_partial_dependency(forest, data, float_feature, perturbations, color='blue'):
    """Plot partial dependency graph.
       Inputs:
         forest: the forest model.
         data: the data sample.
         float_feature: the float feature .
         perturbations: the values to perturb the feature with.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from ._gbdt import DataLoader

    def replace_feature_value(data, feature, v):
        n = len(data)
        data.remove_col(feature)
        data.add_bucketized_float_col(feature, [v] * n)

    def compute_score_stats(forest, data, feature, v, base_scores):
        n = len(data)
        replace_feature_value(data, feature, v)

        scores = forest.predict(data)
        score_diffs = np.array([scores[i] - base_scores[i] for i in xrange(n)])
        return (np.mean(score_diffs), np.std(score_diffs))

    def plot_missing_values(perturbations, mean, std):
        n = len(perturbations)
        plt.plot(perturbations, [mean] * n, color='red', linestyle='--')

    def copy_data(data):
        bucketized_float_cols = dict([(str(col), [v[0] for v in col]) for col in data.get_bucketized_float_cols()])
        string_cols = dict([(str(col), [v for v in col]) for col in data.get_string_cols()])
        return DataLoader.from_dict(
            bucketized_float_cols=bucketized_float_cols,
            string_cols=string_cols)

    feature = float_feature
    data = copy_data(data)
    replace_feature_value(data, feature, float('nan'))
    # Base scores are fixed to scores when the feature are missing.
    base_scores = forest.predict(data)

    perturbations = sorted(perturbations)
    score_stats = [compute_score_stats(forest, data, feature, v, base_scores)
                   for v in perturbations]
    # Use 1.96 for 95% confidence interval.
    lower_bounds =  [p[0] - 1.96 * p[1] for p in score_stats]
    upper_bounds =  [p[0] + 1.96 * p[1] for p in score_stats]

    plt.plot(perturbations, [p[0] for p in score_stats], color=color)
    plt.fill_between(perturbations, lower_bounds, upper_bounds, alpha=.3, color=color)
