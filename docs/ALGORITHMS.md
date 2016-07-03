# Loss Functions
The package supports two basic families of loss functions:
* **Pointwise**
  * Pointwise loss functions are the family of loss functions where the loss can be decomposed into
each instance. The family includes MSE, LogLoss and Huberized Hinge.
* Pairwise
  * Pairwise loss functions are the family of loss functions where the loss can be decomposed into
pairs of training instances. This includes Pairwise Logloss, [GBRank](http://www.cc.gatech.edu/~zha/papers/fp086-zheng.pdf) [LambdaMART](http://research-srv.microsoft.com/pubs/132652/MSR-TR-2010-82.pdf).
LambdaMART is a listwise loss function, but it is implemented via rank-based weighting on pairs. For the Pairwise loss, the package assumes that instances are divided into groups. A natural example
of group are documents retrieved by a query in search engines.
Random sampling of pairs are used to make the computation feasible.

# Newton Step
The package implements a general tree building algorithm that takes a Newton step at each leaf node.
It takes takes inputs of negative gradients
`g` and hessians `h`. It computes
* **node scores** as `g/(h + lambda)` and
* **energy funnctions** as `g*g/(h + lambda)`
where `lambda` is the l2 regularlization parameter.

The tree algorithm greedily selects the best energy reduction node and split and grows the tree
iteratively until it reaches a predeterimined number of leaves.

# Missing Values Handling
Unlike the usual imputation approach for missing values, the algorithm tries to place the missing values to its
optimal branch at every split.

# Categorical Features
The package supports categorical membership split. The splitting algorithm sorts categorical featues
by their node scores and uses the sample splitting algorithm of continuous feature to find the best
energy reduction split. It is shown to be the optimal strategy.
