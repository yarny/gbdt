TUTORIALS
==================

The data in this directory comes from https://github.com/szilard/benchm-ml/tree/master/z-other-tools.

Python Example
-------------------
To run python example, run `python python_example.py logloss`.

C++ Binary Example
------------------
Build C++ binary and run the following commands.

* **Run training:**  `../../bazel-bin/src/gbdt --config_file=benchm-ml.logloss.config --tsvs=train-0.1m.tsv --output_dir=model --logtostderr --num_threads=16`
* **Run testing:**  `../../bazel-bin/src/gbdt --config_file=benchm-ml.logloss.config --tsvs=test.tsv --output_dir=scores --testing_model_file=model/forest.json --logtostderr --num_threads=16`

We also include a convenient script `run.py` to run training, testing and auc computation.
* `./run.py logloss` or `./run.py mse`.

# Writing Configuration file
The package uses json formatted config file with schema defined by [`src/proto/config.proto`](https://github.com/yarny/gbdt/blob/master/src/proto/config.proto). Example configs can be found in `examples/benchm-ml/benchm-ml.*.config`.

# Large TSVs
When the tsvs are large, you can break it into blocks as long as the first block contains header or contains simply the header. The package will load blocks in parallel.