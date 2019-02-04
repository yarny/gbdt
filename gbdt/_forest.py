import six

from ._libgbdt import libgbdt
from ._forest_visualizer import ForestVisualizer

class Forest:
    def __init__(self, forest):
        if isinstance(forest, six.text_type):
            self._forest = libgbdt.Forest(forest)
        elif isinstance(forest, libgbdt.Forest):
            self._forest = forest
        else:
            raise TypeError('Unsupported forest type: {0}'.format(type(forest)))
        self._visualizer = ForestVisualizer(str(self))

    def predict(self, data_store):
        """Computes prediction scores for data_store."""
        return self._forest.predict(data_store._data_store)

    def predict_at_checkpoints(self, data_store, checkpoints):
        """Computes prediction scores for data_store at different checkpoints. At each checkpoint n,
           We compute prediction scores for the sub forest from the first tree to nth tree.
        """
        import tempfile

        output_dir = tempfile.gettempdir()

        self._forest.predict_and_output(data_store._data_store, checkpoints, output_dir)
        for p in checkpoints:
            yield p, [float(line) for line in open(output_dir + '/forest.{}.score'.format(p))
                      if line.strip()]

    def feature_importance(self):
        """Outputs list of feature importances in descending order."""
        return self._forest.feature_importance()

    def features(self):
        """Outputs list of features."""
        return [f for (f, imp) in self._forest.feature_importance()]

    def feature_importance_bar_chart(self, color=None):
        try:
            from matplotlib import pyplot as plt
            import numpy
        except ImportError:
            raise ImportError('Please install matplotlib and numpy.')

        fimps = self.feature_importance()
        importances = [v for _, v in fimps]
        features = [f for f,_ in fimps]
        ind = -numpy.arange(len(fimps))

        _, ax = plt.subplots()
        plt.barh(ind, importances, align='center', color=color)
        ax.set_yticks(ind)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature importance')
        plt.show()

    def __str__(self):
        return self._forest.as_json()

    def visualize_tree(self, i):
        return self._visualizer.visualize_tree(i)