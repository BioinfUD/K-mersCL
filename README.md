K-mersCL
========

This repository contains an implementation of a heterogenous parallel processing model  for super k-mer obtaining and distribution based on minimizers. This model has two parts: - A data structure based on bit arrays that allows the representation of super k-mers and minimizers from a set of reads with low memory demand and - A heterogenous processing model that use massive paralallel processing to the a task with high computing requirements: Obtaining such bit arrays, and serial processing for task with high memory requirements (and low computing requirement): Make explicit the obtaining and distribution of the super k-mers.


Implemented kernels
--------------------------------
#### getSuperK2_M
Obtains super k-mers based on Canonical Minimizer.

`python2 -u getSuperK2_M.py --kmer <KMER_SIZE> --mmer <MMER_SIZE> --input_file <PATH_TO_FILE> --read_size <READ_LENGHT> --output_path <OUTPUT_PATH> --n_reads <NUMBER_OF_READS>`

### getSuperK2_M_signature: 
Obtains super k-mers based on Signature.

`python2 -u getSuperK2_M_signature.py --kmer <KMER_SIZE> --mmer <MMER_SIZE> --input_file <PATH_TO_FILE> --read_size <READ_LENGHT> --output_path <OUTPUT_PATH> --n_reads <NUMBER_OF_READS>`

*Note:*  OpenCL 1.2 o superior is required

Commands to test
-------------


Evaluation
-----------
### Configuration
Configure paths to tools like MSPK, and KMC in `config.py` file. 

### Dataset
https://drive.google.com/drive/folders/13rq3fmRr8g3AQXx7ZgSG-9iNq-hbQRoD?usp=sharing

### Used command :
`PYOPENCL_CTX='0:0' python multiple_execution.py --kmers 51,81 --mmers 5,7 --read_sizes 180,300,180,300 --input_files /home/nvera/benchmark/data/SRR768269_3m_180.fasta,/home/nvera/benchmark/data/SRR768269_3m_300.fasta,/home/nvera/benchmark/data/SRR768269_9m_180.fasta,/home/nvera/benchmark/data/SRR768269_9m_300.fasta --output_path $PWD/output --method kmerscl_signature,kmc,mspk,kmerscl --n_reads 3000000,3000000,9000000,9000000`

*Note:* `PYOPENCL_CTX='0:0'` was used to avoid pyopencl to ask for the GPU to be used.

### Modifications to reference tools
MSPKmerCounter: https://github.com/BioinfUD/MSPK_with_timing/blob/master/Partition.java#L129-L208
KMC2: https://github.com/BioinfUD/KMC_withtiming/blob/master/kmer_counter/splitter.h#L717-L812
