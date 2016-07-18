Gradient Boosting Decision Trees Algorithms (GBDT)
=======
Author: Jiang Chen (criver@gmail.com)
-----------
GBDT is a high performance and full featured C++ implementation of [Jerome H. Friedman's Gradient Boosting Decision Trees Algorithm](http://statweb.stanford.edu/~jhf/ftp/stobst.pdf) and its modern offsprings,. It features high efficiency, low memory footprint, collections of loss functions and built-in mechanisms to handle categorical features and missing values.


When is GBDT good for you?
-----------
* **You are looking beyond linear models.**
  * Gradient Boosting Decision Trees Algorithms is one of the best offshelf ML algorithms with built-in capabilities of non-linear transformation and feature crossing.
* **Your data is too big to load into memory with existing ML packages.**
  * GBDT reduces memory footprint dramatically with feature bucketization. For some tested datasets, it used 1/7 of the memory of its counterpart and took only 1/2 time to train. See [docs/PERFORMANCE_BENCHMARK.md](https://github.com/yarny/gbdt/blob/master/docs/PERFORMANCE_BENCHMARK.md) for more details.
* **You want better handling of categorical features and missing values.**
  * GBDT has built-in mechanisms to figure out how to split categorical features and place missing values in the trees.
* **You want to try different loss functions.**
  * GBDT implements various pointwise, pairwise, listingwis loss functions including mse, logloss, huberized hinge loss, pairwise logloss,
[GBRank](http://www.cc.gatech.edu/~zha/papers/fp086-zheng.pdf) and [LambdaMart](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf). It supports easily addition of your own custom loss functions.

Installation (python2.7 + linux x86_64):
---------
`pip install git+https://github.com/yarny/gbdt.git`

Documentations
---------
* Installation instructions can be found at [docs/INSTALL.md](https://github.com/yarny/gbdt/blob/master/docs/INSTALL.md).
* Example usages can be found at [examples/benchm-ml/TUTORIAL.md](https://github.com/yarny/gbdt/blob/master/examples/benchm-ml/TUTORIAL.md).
* Some details of the algorithms can be found at [docs/ALGORITHMS.md](https://github.com/yarny/gbdt/blob/master/docs/ALGORITHMS.md).
* Performance benchmark: [docs/PERFORMANCE_BENCHMARK.md](https://github.com/yarny/gbdt/blob/master/docs/PERFORMANCE_BENCHMARK.md)