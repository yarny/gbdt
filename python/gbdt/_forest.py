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

    def __str__(self):
        return self._forest.as_json()
