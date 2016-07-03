TUTORIALS
==================

The data in this directory comes from https://github.com/szilard/benchm-ml/tree/master/z-other-tools.

* **Run training:**  `../../bazel-bin/src/gbdt --config_file=benchm-ml.logloss.config --tsvs=train-0.1m.tsv --output_dir=model --logtostderr --num_threads=16`
* **Run testing:**  `../../bazel-bin/src/gbdt --config_file=benchm-ml.logloss.config --tsvs=test.tsv --output_dir=scores --testing_model_file=model/forest.json --logtostderr --num_threads=16`

We also include a convenient script `run.py` to run training, testing and auc computation.
* `./run.py logloss` or `./run.py mse`.

# Writing Configuration file
The package uses json formatted config file with schema defined by `src/proto/config.proto`. A
configuration file includes the following sections:
* tree_config
* sampling_config
* loss_func_config
* data_config
* eval_config.

Examples with can be found in `examples/benchm-ml/benchm-ml.*.config`.

# Large TSVs
When the tsvs are large, you can break it into blocks as long as the first block contains header or contains simply the header. The package will load blocks in parallel.