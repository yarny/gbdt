#!/usr/bin/python

import math
import sys

def ComputeMse(scores, responses):
    return math.sqrt(sum([(s-r)*(s-r) for s, r in zip(scores, responses)]) / len(scores))

def main():
    scores = [float(line) for line in open(sys.argv[1]).readlines() if not line.startswith('#')]
    responses = [float(line) for line in open(sys.argv[2]).readlines() if not line.startswith('#')]
    print ComputeMse(scores, responses)
    assert(len(scores) == len(responses))

if __name__ == '__main__':
    main()
