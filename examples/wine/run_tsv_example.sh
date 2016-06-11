#!/bin/bash

../../bazel-bin/src/gbdt --config_file=wine.tsv.config --tsvs=wine.tsv --output_dir=model --logtostderr

../../bazel-bin/src/gbdt --config_file=wine.tsv.config --tsvs=wine.tsv --output_dir=model --testing_model_file=model/forest.json --mode=test --logtostderr

../../bazel-bin/src/gbdt --config_file=wine.tsv.config --tsvs=wine.header,wine.headerless.tsv --output_dir=model --logtostderr
