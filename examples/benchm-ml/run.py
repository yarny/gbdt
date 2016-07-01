#!/usr/bin/python

import sys
import json
import subprocess
import numpy as np
from sklearn import metrics
import csv
import matplotlib.pyplot as plt

def configName(model):
    return 'benchm-ml.{0}.config'.format(model)

def forestName(model):
    return 'forest.{0}'.format(model)

def executeCommand(command):
    print >>sys.stderr, command
    subprocess.check_output(command, shell=True)

def runTraining(model):
    command = '~/repos/gbdt/bazel-bin/src/gbdt --tsvs=train-0.1m.tsv --config_file={0} --num_threads=16 --output_dir=models --output_model_name={1} --logtostderr'.format(configName(model), forestName(model))
    executeCommand(command)

def runTesting(model):
    command = 'mkdir -p scores; ~/repos/gbdt/bazel-bin/src/gbdt --tsvs=test.tsv --config_file={0} --num_threads=16 --testing_model_file=models/{1}.json --logtostderr --output_dir=scores/{1} --mode=test'.format(configName(model), forestName(model))
    executeCommand(command)

def runAUC(model):
    reader = csv.reader(open('test.tsv'), delimiter='\t')
    reader.next()
    targets = np.array([float(row[-1]) for row in reader])
    scoreDir = 'scores/{}'.format(forestName(model))
    auc = []
    for i in range(1, 200, 20):
        predictions = np.array([float(line) for line in open(scoreDir + '/forest.{}.score'.format(i))])
        assert(len(targets) == len(predictions))
        fpr, tpr, _ = metrics.roc_curve(targets, predictions, pos_label=1)
        auc.append(metrics.auc(fpr, tpr))
    print >> open('scores/{}.auc'.format(forestName(model)), 'w'), '\n'.join([str(a) for a in auc])

def main():
    model = sys.argv[1]

    runTraining(model)
    runTesting(model)
    runAUC(model)


if __name__ == '__main__':
    main()
