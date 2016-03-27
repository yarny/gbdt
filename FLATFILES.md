# Flatfiles Format
Each flatfile contains a column in the data matrix.
They are stored in a single directory (flatfiles_dir) with feature names being filenames.
For example, if you have three features A, B, and C, your flatfiles_dir
will have three files A, B, and C. Each flatfile will contain one data record per row with
the first row contains metadata about their types. These types tell the package how to
preprocess before learning. The flatfiles must be of equal length.

Currently, the package supports three types of flatfiles:
* StringType with the first meta row `# dtype=string`. StringType is for string or categorical
features.
* BinnedFloatType with first meta row `# dtype=binnedfloats`. BinnedFloatType is for
float features. Float features are binned before learning.
* RawFloatType with first meta row `# dtype=rawfloats`. RawFloatType is for targets, for which
we want to retain the original values and don't want to bin them.

The packages contains utilities to convert tsv to flatfiles.