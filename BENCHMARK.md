# Benchmark

* DataSet:  **42901690 rows and 135 features**, among which 97 are encoded
in short ints and 48 are encoded in bytes.
* Machine: It was tested in m3.2xlarge amazon instance on 01/18/2016.
* Parameters:	10 threads, 200 trees, 14 leaves, 0.5 example sampling rate,
0.9 feature sampling rate. RSME is the loss function.
* Memory footprint: **1.085G**
* Running time: **10m44s**

In terms of memory footprint, the performance is close to theoretically optimal.