#!/bin/bash

../../bazel-bin/src/gbdt --config_file=wine.tsv.config --tsvs=wine.header,wine.tsv --output_dir=model

../../bazel-bin/src/gbdt --config_file=wine.tsv.config --tsvs=wine.header,wine.tsv --output_dir=model --testing_model_file=model/forest.json --mode=test
