Gradient Boosting Decision Tree Algorithms
=======
Author: Jiang Chen (criver@gmail.com)
-----------
# Overview
High performance C++ implementation of Jerome H. Fiedman's Gradient Boosting Decision Tree GBDT
family of algorithms (http://statweb.stanford.edu/~jhf/ftp/stobst.pdf) with many practical features
suited for problems and datasets of industrial grade.

# Algorithmic highlights
1. Implmentations of various pointwise, pairwise, listwise loss functions including: MSE,
LogLoss, Huberized Hinge Loss, Pairwise Logloss, LambdaMART.
2. Easily extensible to new loss function.
3. Uses Hessian for fast convergence.
4. Built-in intelligent missing value handling.
5. Categorical feature spports.

# Implmentation highlights
1. Heavily optimized for both memory footprint and computation time.
See BENCHMARK.md for the performance numbers. The key idea is to binning of float features,
which not only cuts the memory footprint significantly and reduce the core splitting algorithm's
complexity from O(Nlog(N)) to O(N).
2. Modern build tool [bazel](bazel.io).
3. Compliance to Google Style Guide and Excellent test coverage.
4. State-of-the-art google libraries: gtest, gflags, glog, protobuf3.