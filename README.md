K-mersCL
========

This repository contains an implementation of a heterogenous parallel processing model  for super k-mer obtaining and distribution based on minimizers. This model has two parts: - A data structure based on bit arrays that allows the representation of super k-mers and minimizers from a set of reads with low memory demand and - A heterogenous processing model that use massive paralallel processing to the a task with high computing requirements: Obtaining such bit arrays, and serial processing for task with high memory requirements (and low computing requirement): Make explicit the obtaining and distribution of the super k-mers.


Implemented kernels
--------------------------------
* getSuperK_M: Obtains super k-mers based on Canonical Minimizer.
* getSuperK_Sig: Obtains super k-mers based on Signature.

*Note:*  OpenCL 1.2 o superior is required
