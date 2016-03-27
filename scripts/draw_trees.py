#!/usr/bin/python

"""
The code uses Graphviz dot. Please install GraphViz before calling the script.

Usage:
    python draw_trees.py input.model output_dir
"""

import json
import os
import re
import subprocess
import sys

def branchNodeToString(node):
    split = node['split']
    if 'floatSplit' in split:
        return '{feature} {less_than} {threshold:.3f} \\n{score:.6f}'.format(
            less_than='<=*' if 'missingToRightChild' in split['floatSplit'] else '*<=',
            feature=split['feature'],
            threshold=split['floatSplit']['threshold'],
            score=node['score'])
    elif 'catSplit' in split:
        return '{feature} in {categories}'.format(
            feature=split['feature'],
            categories=str(split['catSplit']['category']))
    else:
        raise

def leafNodeToString(node):
    return '{0:.6f}'.format(node['score'])

def traverse(tree, node_name):
    buffer = ''
    color = 'blue'
    if "leftChild" in tree:
        buffer += '{node_name} [label="{label}" color={color}];\n'.format(
            node_name=node_name, label=branchNodeToString(tree), color=color)
        left_node_name = node_name + 'l'
        right_node_name = node_name + 'r'
        buffer += '{parent} -> {child};\n'.format(parent=node_name, child=left_node_name)
        buffer += '{parent} -> {child};\n'.format(parent=node_name, child=right_node_name)
        buffer += '{left} -> {right}[style=invis constraint=false];\n'.format(
            left=left_node_name, right=right_node_name)
        buffer += traverse(tree['leftChild'], left_node_name)
        buffer += traverse(tree['rightChild'], right_node_name)
    else:
        # leaf node
        label = leafNodeToString(tree)
        buffer += '{node_name} [label="{label}" color={color}];\n'.format(
            node_name=node_name, label=leafNodeToString(tree), color=color)
    return buffer

def main():
    if len(sys.argv) != 3:
        raise Exception('Usage python tree_visualizer.py input.model output_dir')

    output_dir = sys.argv[2]
    model_json = json.loads(open(sys.argv[1]).read())

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    trees = model_json["tree"]
    dot_file = '{output_dir}/forest.dot'.format(output_dir=output_dir)
    print >>open(dot_file, 'w'), '\n'.join(
        ['digraph tree {\n' + traverse(tree, 'node{0}'.format(i)) + '\n}'
         for i, tree in enumerate(trees)])
    subprocess.check_call('dot -Tpng -O {0}'.format(dot_file), shell=True)
    subprocess.check_call('dot -Tps2 {dot_file} -o {output_dir}/forest.ps'.format(
        dot_file=dot_file, output_dir=output_dir), shell=True)

if __name__ == '__main__':
    main()
