from libgbdt import Forest as _Forest

class Forest:
    def __init__(self, forest):
        if type(forest) is str or type(forest) is unicode:
            self._forest = _Forest(forest)
        elif type(forest) is _Forest:
            self._forest = forest
        else:
            raise TypeError, 'Unsupported forest type: {0}'.format(type(forest))

    def predict(self, data_store):
        """Computes prediction scores for data_store."""
        return self._forest.predict(data_store._data_store)

    def feature_importance(self):
        """Outputs list of feature importances in descending order."""
        return self._forest.feature_importance()

    def feature_importance_bar_chart(self, color='blue'):
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

    def __str__(self):
        return self._forest.as_json()
