#!/bin/bash

../../bazel-bin/src/gbdt --config_file=hepatitis.logloss.config --tsvs=hepatitis.tsv --output_dir=model --logtostderr
