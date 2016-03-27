#!/usr/bin/python

import json
from optparse import OptionParser
from optparse import make_option
import os
import re
from shutil import copytree
from shutil import rmtree
import subprocess
import sys

option_list = [
    make_option("--num_bags",
                action="store", type="int", dest="num_bags", default=10),
    make_option("--training_flatfiles_dir",
                action="store", type="string", dest="training_flatfiles_dir"),
    make_option("--config_file",
                action="store", type="string", dest="config_file"),
    make_option("--output_dir",
                action="store", type="string", dest="output_dir"),
    make_option("--num_threads",
                action="store", type=int, dest="num_threads", default=10),
]

def executeCommand(command):
    print command
    subprocess.check_call(command, shell=True)

class Bagging:
    def __init__(self, opts):
        self._opts = opts
        flatfiles = os.listdir(self._opts.training_flatfiles_dir)
        flatfile = os.path.join(self._opts.training_flatfiles_dir, flatfiles[0])
        self._opts.num_rows = len(
            [line for line in open(flatfile).readlines() if not line.startswith('#') and line.strip()])

    def bagDir(self, k):
        return '{0}/bag_{1}'.format(self._opts.output_dir, k)

    def inBag(self, row_number, k):
        return hash(str(row_number)) % self._opts.num_bags != k

    def outOfBag(self, row_number, k):
        return not self.inBag(row_number, k)

    def trainAndEvalBags(self):
        for k in xrange(self._opts.num_bags):
            self.trainAndEvalBag(k)

        testPoints = self.getTestPoints()

        score_dir = '{0}/scores'.format(self._opts.output_dir)
        try:
            rmtree(score_dir)
        except:
            pass
        try:
            os.makedirs(score_dir)
        except:
            pass

        for t in testPoints:
            bagged_scores = self.baggedScores(t)
            scores = [self.inBagScore(bagged_scores, i) for i in xrange(self._opts.num_rows)]
            in_bag_file = open('{0}/in_bag.{1}.score'.format(score_dir, t), 'w')
            print >>in_bag_file, '\n'.join([str(score) for score in scores])
            scores = [self.outOfBagScore(bagged_scores, i) for i in xrange(self._opts.num_rows)]
            out_bag_file = open('{0}/out_of_bag.{1}.score'.format(score_dir, t), 'w')
            print >>out_bag_file, '\n'.join([str(score) for score in scores])

    def prepareBaggingDir(self, k):
        bagging_dir = self.bagDir(k)
        flatfiles_dir = os.path.join(bagging_dir, 'flatfiles')
        try:
            os.makedirs(flatfiles_dir)
        except:
            pass

        self._opts.config['data_config']['sample_weight_column'] = 'bagging_weights'
        print >>open(os.path.join(bagging_dir, 'bagging.config'), 'w'), json.dumps(self._opts.config)
        # First prepair a weight flatfile for the bag.
        weights = ['# dtype=raw_floats'] + [str(int(self.inBag(i, k))) for i in xrange(self._opts.num_rows)]
        # Make a temporary directory and copying all the files.
        print >> open(os.path.join(flatfiles_dir, 'bagging_weights'), 'w'), '\n'.join(weights)

    def trainAndEvalBag(self, k):
        working_dir = self.bagDir(k)
        self.prepareBaggingDir(k)
        flatfiles_dir = os.path.join(working_dir, 'flatfiles')
        model_dir = os.path.join(working_dir, 'model')

        # Train the model.
        training_command = """{binary} --training_flatfiles_dirs={flatfiles_dir},{training_flatfiles_dir} \
        --num_threads={num_threads} \
        --logtostderr \
        --output_model_name=forest \
        --config_file={config_file} --output_dir={model_dir}""".format(
            binary=self._opts.binary,
            flatfiles_dir=flatfiles_dir,
            num_threads=self._opts.num_threads,
            training_flatfiles_dir=self._opts.training_flatfiles_dir,
            config_file=os.path.join(working_dir, 'bagging.config'),
            model_dir=model_dir)
        testing_command = """{binary} --testing_flatfiles_dirs={training_flatfiles_dir} \
        --mode=test --config_file={config_file} --testing_model_file={model_dir}/forest.json \
        --logtostderr \
        --num_threads={num_threads} \
        --output_dir={model_dir}""".format(
            binary=self._opts.binary,
            training_flatfiles_dir=self._opts.training_flatfiles_dir,
            config_file=self._opts.config_file,
            num_threads=self._opts.num_threads,
            model_dir=model_dir)
        executeCommand(training_command)
        executeCommand(testing_command)

    def getTestPoints(self):
        return [score_file.split('.')[-2]
                for score_file in os.listdir('{0}/model/'.format(self.bagDir(0)))
                if re.match('forest\..*\.score', score_file)]

    def baggedScores(self, t):
        return  [ [float(line)
                    for line in open('{0}/model/forest.{1}.score'.format(self.bagDir(k), t)).readlines()]
                   for k in xrange(self._opts.num_bags)]

    def inBagScore(self, bagged_scores, i):
        scores = [bagged_scores[k][i] for k in xrange(self._opts.num_bags)
                  if self.inBag(i, k)]
        return sum(scores) / len(scores)

    def outOfBagScore(self, bagged_scores, i):
        scores = [bagged_scores[k][i] for k in xrange(self._opts.num_bags)
                  if self.outOfBag(i, k)]
        return sum(scores) / len(scores)

def main():
    parser = OptionParser(option_list=option_list)
    (opts, args) = parser.parse_args(sys.argv)
    opts.binary = os.path.join(os.path.dirname(args[0]), '../bazel-bin/src/gbdt')
    opts.config = json.loads(open(opts.config_file).read())

    bagging = Bagging(opts)
    bagging.trainAndEvalBags()


if __name__ == '__main__':
    main()
