# Examples
Toy examples can be found in `examples` directory. `examples/wine/run_tsv_example.sh` contains
example commands for training and testing:
* `../../bazel-bin/src/gbdt --config_file=wine.tsv.config --training_tsvs=wine.header,wine.tsv --output_dir=model`
* `../../bazel-bin/src/gbdt --config_file=wine.tsv.config --testing_tsvs=wine.header,wine.tsv --output_dir=model --testing_model_file=model/forest.json --mode=test`

# Configuration file
The package uses json formatted config file with schema defined by `src/proto/config.proto`. A
configuration file includes the following sections:
* tree_config
* sampling_config
* loss_func_config
* data_config
* eval_config.
An example with can be found in `examples/wine/wine.tsv.config`.

# Input Data Format
The package supports two formats
1. TSV Blocks.
2. Flatfiles.

TSV Blocks
-------------
TSV block format is essentially a TSV file divided into blocks for parallel loading. The header
file is seperated and supplied as the first file to flags `--training_tsvs` or `--testing_tsvs`.
See an example in `examples/wine/run_tsv_examples.sh`.

Flatfiles
-----------
See FLATFILES.md for description.