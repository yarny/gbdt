#!/usr/bin/python

import collections
import math
import re
import sys
import operator
from optparse import make_option
from optparse import OptionParser

option_list = [
    make_option('--group_file', action='store', type='string', dest='group_file',
                help='[REQUIRED] The group flatfile.'),
    make_option('--score_file', action='store', type='string', dest='score_file',
                help='[REQUIRED] The score file.'),
    make_option('--target_file', action='store', type='string', dest='target_file',
                help='[REQUIRED] The target file.')
]

def checkRequiredArguments(opts, parser):
    missing_options = []
    for option in parser.option_list:
        if re.match(r'^\[REQUIRED\]', option.help) and eval('opts.' + option.dest) == None:
            missing_options.extend(option._long_opts)
    if len(missing_options) > 0:
        parser.error('Missing REQUIRED parameters: ' + str(missing_options))

def ReadFlatFile(f):
    for line in open(f):
        if line.startswith('#'):
            continue
        yield line

def ReadByGroup(opts):
    group_file = ReadFlatFile(opts.group_file)
    score_file = ReadFlatFile(opts.score_file)
    target_file = ReadFlatFile(opts.target_file)

    prev_gid = None
    group = []
    for gid in group_file:
        score = float(score_file.next())
        target = float(target_file.next())
        if prev_gid != None and gid != prev_gid:
            yield group
            group = []

        prev_gid = gid
        group.append((score, target))

    if group:
        yield group


def discount(pos):
    return math.log(2.4)/math.log(pos + 2.4)

def mergeMetrics(sub_metric, summary_metrics):
    for k,v in sub_metric.iteritems():
        summary_metrics[k] += sub_metric[k]

def computeMetrics(ranked_targets):
    metrics = collections.defaultdict(lambda: 0)
    for i, target in enumerate(ranked_targets):
        utility = discount(i)  * target
        metrics['dcu_{0}'.format(target)] += utility
        metrics['dcu_total'] += utility
        metrics['count_{0}'.format(target)] += 1
        metrics['count_total'] += 1

    return metrics

def main():
    parser = OptionParser(option_list=option_list)
    (opts, args) = parser.parse_args(sys.argv)
    checkRequiredArguments(opts, parser)
    metrics = collections.defaultdict(lambda: 0)

    for numGroups, group in enumerate(ReadByGroup(opts)):
        group.sort(key=operator.itemgetter(0), reverse=True)
        mergeMetrics(computeMetrics(map(operator.itemgetter(1), group)), metrics)

    keys = sorted(metrics.keys())
    for k in keys:
        print '%s=%.4f' % (k, metrics[k] / (numGroups + 1))

if __name__ == '__main__':
    main()
