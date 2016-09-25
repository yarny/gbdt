https://testpypi.python.org/pypi
# TUTORIALS

## Table of Contents
- [Run it in Python](#run-it-in-python)
  * [Import](#import)
  * [Load Data](#load-data)
  * [Training](#training)
  * [Testing](#testing)
  * [DataStore](#datastore)
  * [Partial Dependency Plot](#partial-dependency-plot)
  * [Forest Visualizer](#forest-visualizer)
- [Run it as C++ binary](#run-it-as-c-binary)
- [Disclaimer](#disclaimer)

## Run it in Python
An example can be found in [python_example.py](https://github.com/yarny/gbdt/blob/master/examples/benchm-ml/python_example.py). For a quick test, please run
```sh
./python_example.py logloss
```
In the following, we are going to explain the example code in more details.
### Import
Install the python package following [the instructions](https://github.com/yarny/gbdt/blob/master/docs/INSTALL.md).
```sh
import gbdt
```

### Load Data
#### TSVs
```python
training_data = gbdt.DataLoader.from_tsvs(
    tsvs=["train-0.1m.tsv"],
    bucketized_float_cols=float_features,
    string_cols=cat_features + [target_column])
```
This code snippet loads a selected columns of a tsv into DataStore. It loads all `float_features` as
bucketized float columns and all `cat_features` and `target_column` as string columns. Overall, the
package supports 3 kinds of columns:
* `bucketized_float_cols`: All float features are bucketized with equal frequency binning in the
preprocessing step to improve memory and computation efficiency with minimal loss of precision.
* `string_cols`: Categorical features as well as other auxilary columns.
* `raw_float_cols`: We also support a **raw** float mode to load weights and regression targets.

#### Parallel Loading of TSVs
If the tsv are large (>10G), we recommend you divide them into blocks and feed into the loader as a
list of tsv files. It enables parallel loading.
The only requirement is that first block contain the header.

#### Pandas' DataFrame
The package supports loading from Pandas' DataFrame. It will load all numeric columns as
`bucketized_float_cols` and all other columns as `string_cols` unless overridden by `type_overrides`.
```python
df = pandas.read_csv('train-0.1.m.tsv', sep='\t')
training_data = gbdt.DataLoader.from_df(df)
```
### Training
* Training Parameters (as defined by [`src/proto/config.proto`](https://github.com/yarny/gbdt/blob/master/src/proto/config.proto)):
```python
config = {'loss_func': 'logloss',
          'num_trees': 20,
          'num_leaves': 16,
          'example_sampling_rate': 0.5,
          'feature_sampling_rate': 0.8,
          'pair_sampling_rate': 20,
          'min_hessian': 50,
          'shrinkage' : 0.05}
```

* Train a model
```python
training_targets = map(lambda x: 1 if x=='Y' else -1, training_data[target_column])
forest = gbdt.train(training_data,
                    y=training_targets,
                    features=float_features + cat_features,
                    config=config)
```
* Output the model as json.
```python
print >>open('forest.json', 'w'), forest
```
* Feature improtance.
```python
forest.feature_importance()
forest.feature_importance_bar_chart()
```
![](images/feature_importance.png?raw=true)

### Testing
* Score the whole forest.
```python
predictions = forest.predict(data)
```
* Score sub-forests. (The following example score sub-forests with 10, 20, 30 trees.)
```python
predictions = forest.predict_at_checkpoints(data, [10, 20, 30]).
```
### DataStore
* Accessing Columns:
```sh
training_data['dep_delayed_15min'].
```
* Slice and dice (outputs DataStore)
```python
training_data[100]
training_data[100, 200]
training_data[[2, 10, 5, 11, 12]]
```
* To Pandas DataFrame
```python
training_data.to_df()
```

### Partial Dependency Plot
```python
x = random.sample(training_data['DepTime'], 200)
gbdt.plot_partial_dependency(forest, training_data, 'DepTime', x):
```
![](images/partial_dependency_plot.png?raw=true)
![](images/partial_dependency_plot2.png?raw=true)

### Forest Visualizer
```
visualizer = gbdt.ForestVisualizer(forest)
visualizer.visualize_tree(10)
```

## Run it as C++ binary
Compile the binary following [the instructions](https://github.com/yarny/gbdt/blob/master/docs/INSTALL.md).
* **Run training:**
```sh
../../bazel-bin/src/gbdt
 --config_file=benchm-ml.logloss.config \
 --tsvs=train-0.1m.tsv \
 --output_dir=. \
 --logtostderr \
 --num_threads=16 \
```
The config file is a json-formatted with schema defined by
[`src/proto/config.proto`](https://github.com/yarny/gbdt/blob/master/src/proto/config.proto).
The output model is `forest.json`.
* **Run testing:**
```sh
../../bazel-bin/src/gbdt \
  --config_file=benchm-ml.logloss.config \
  --tsvs=test.tsv \
  --output_dir=scores \
  --testing_model_file=forest.json \
  --logtostderr \
  --num_threads=16 \
```
Score files can be found at `scores` subdir.

## Disclaimer
The data in this directory comes from [benchm-ml](https://github.com/szilard/benchm-ml/tree/master/z-other-tools).