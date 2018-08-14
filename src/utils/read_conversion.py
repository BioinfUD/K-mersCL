import numpy as np
import sys
from numpy import ndarray
import os

BASE_TO_INT = {
    "A" : 0,
    "C" : 1,
    "G" : 2,
    "T" : 3
}

INT_TO_BASE = {v: k for k,v in BASE_TO_INT.iteritems()}

def base_to_int(b):
    return BASE_TO_INT.get(b, 99)


def int_to_base(b):
    return INT_TO_BASE.get(b, "U")

def integer_to_bases(number, m=4):
    mask = 0b11
    bases = ""
    for c in (range(m-1, -1, -1)):
        base = int_to_base((number >> c*2) & mask)
        bases = bases + base
    return bases

# For fasta file
def file_to_matrix(filename="/tmp/outfile.txt", r=180, n_reads=None):
    if n_reads is None:
        sys.stdout.write("Number of reads not specified , estimating the number\n")
        in_file = open(filename, "rU")
        first_line = in_file.readline().strip()
        second_line = in_file.readline().strip()
        avg_record_bytes = len(str(first_line)) + len(str(second_line)) + 2
        input_file_size = os.path.getsize(filename)
        estimated_reads = input_file_size / avg_record_bytes
    else:
        estimated_reads = int(n_reads)
    sys.stdout.write("Estimated/specified number of reads: {}\n".format(estimated_reads))
    reads_matrix = ndarray(shape=(estimated_reads, r), dtype=np.dtype("c"))
    skip_line = True
    counter = 0
    with open(filename) as f:
        for line in f:
            if skip_line:
                skip_line = False
                continue
            else:
                reads_matrix[counter] = line.strip()
                counter += 1
                skip_line = True

            if counter % 100000 == 0:
                sys.stdout.write("{} reads has been loaded\n".format(counter))
    reads_matrix = reads_matrix[0:counter]
    sys.stdout.write("Reads loaded into memory, converting bases to numbers \n")
    numbers_matrix = np.full(reads_matrix.shape, 99, dtype=np.uint8)
    for base, number in BASE_TO_INT.iteritems():
        numbers_matrix[reads_matrix == base] = number
    sys.stdout.write("{} reads has been loaded, initial estimate was {} reads\n".format(counter, estimated_reads))
    return numbers_matrix
