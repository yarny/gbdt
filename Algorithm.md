# Loss Functions
The package supports two basic families of loss functions:
* Pointwise
* Pairwise
Pointwise loss functions are the family of loss functions where the loss can be decomposed into
each training instance. The family includes MSE, LogLoss and Huberized Hinge.

Pairwise loss functions are the family of loss functions where the loss can be decomposed into
pairs of training instances. This includes Pairwise Logloss and [LambdaMART](http://research-srv.microsoft.com/pubs/132652/MSR-TR-2010-82.pdf).
LambdaMART is a listwise loss function, but it is implemented via rank-based weighting on pairs.

For the Pairwise loss, the package assumes that instances are divided into groups. For Search
Engine applications, all the documents retrieved by a query are one group of instances.
Random sampling of pairs are used to make the computation feasible.

# Hessian and Tree Building Algorithm
The package implements a general tree building algorithm that takes inputs of negative gradients
`g` and hessians `h`. It computes
* node scores as `g/h` and
* energy as `g*g/h`.

The tree algorithm greedily selects the best energy reduction node and split and grows the tree
iteratively until it reaches a predeterimined number of leaves.

# Missing Values Handling
Unlike the usual imputation approach, the algorithm tries to place the missing values to its
optimal branch at every split. It essentially tries to find the best imputed value for the
missing value algorithmically.

# Categorical Features
The package supports categorical membership split. The splitting algorithm tries to find
the division of the categories that best reduces energe.
