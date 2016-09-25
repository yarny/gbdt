"""
Visualizes forest model.
"""

import json
from ._forest import Forest

class ForestVisualizer:
    def __init__(self, forest):
        if type(forest) is dict:
            self._forestJson = forest
        else:
            self._forestJson = json.loads(str(forest))

    def num_trees(self):
        return len(self._forestJson['tree'])

    def visualize_tree(self, i):
        """Visualizes tree i."""
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError('Please install graphviz.')

        tree = self._forestJson['tree'][i]
        digraph = Digraph(format='png')
        self.__visualizeTreeInternal__(digraph, tree, 'tree{0}'.format(i))
        return digraph

    def __floatSplit__(self, split):
        missingToRightChild = 'missingToRightChild' in split['floatSplit'] and split['floatSplit']['missingToRightChild']
        return '{feature} {less_than} {threshold:.3f}'.format(
            less_than='<=*' if missingToRightChild else '*<=',
            feature=split['feature'],
            threshold=split['floatSplit']['threshold'])

    def __catSplit__(self, split):
        return '{feature} in {categories}'.format(
            feature=split['feature'],
            categories='{' + ', '.join(split['catSplit']['category']) + '}')

    def __branchNodeLabel__(self, node):
        split = node['split']

        splitDescription = self.__floatSplit__(split) if 'floatSplit' in split else self.__catSplit__(split)
        return splitDescription + '\\n{0: .6f}'.format(node['score'])

    def __leafNodeLabel__(self, node):
        return '{0:.6f}'.format(node['score'])

    def __nodeLabel__(self, node):
        if 'leftChild' in node:
            return self.__branchNodeLabel__(node)
        else:
            return self.__leafNodeLabel__(node);

    def __visualizeTreeInternal__(self, digraph, node, nodeName):
        digraph.node(nodeName, label=self.__nodeLabel__(node), color='blue')
        if 'leftChild' in node:
            leftNodeName = nodeName + 'l'
            rightNodeName = nodeName + 'r'
            self.__visualizeTreeInternal__(digraph, node['leftChild'], nodeName + 'l')
            self.__visualizeTreeInternal__(digraph, node['rightChild'], nodeName + 'r')
            digraph.edge(nodeName, leftNodeName)
            digraph.edge(nodeName, rightNodeName)

def main():
    import sys
    model = json.loads(open(sys.argv[1]).read())
    visualizer = ForestVisualizer(model)
    digraph = visualizer.visualize_tree(1)
    digraph.render(filename='tree')

if __name__ == '__main__':
    main()
