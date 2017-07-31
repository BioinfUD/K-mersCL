import numpy as np
import sys
from numpy import ndarray
import os


def base_to_int(b):
    if b == "A":
        return 0
    if b == "C":
        return 1
    if b == "G":
        return 2
    if b == "T":
        return 3
    else:
        return 99  # Not valid value


def int_to_base(b):
    if b == 0:
        return "A"
    if b == 1:
        return "C"
    if b == 2:
        return "G"
    if b == 3:
        return "T"
    else:
        return "U"  # Not valid value


def integer_to_bases(number, m=4):
    mask = 0b11
    bases = ""
    for c in (range(m-1, -1, -1)):
        base = int_to_base((number >> c*2) & mask)
        bases = bases + base
    return bases

# For fasta file
def file_to_matrix(filename="/tmp/outfile.txt", r=180, n_reads=None):
    in_file = open(filename, "rU")
    first_line = in_file.readline().strip()
    second_line = in_file.readline().strip()
    avg_record_bytes = len(str(first_line)) + len(str(second_line)) + 2
    input_file_size = os.path.getsize(filename)
    estimated_reads = n_reads if n_reads else (input_file_size/avg_record_bytes)
    sys.stdout.write("Estimated/specified number of reads: {}\n".format(estimated_reads))
    reads_matrix = ndarray(shape=(estimated_reads, r), dtype=np.uint32)
    reads_matrix[0] = map(base_to_int,list(str(second_line)))
    counter = 1
    in_file.readline()  # Skip id line
    line = in_file.readline()  # Skip id line
    while line:
        reads_matrix[counter] = map(base_to_int, list(str(line.strip())))
        if counter % 100000 == 0:
            sys.stdout.write("{} reads has been loaded\n".format(counter))
        in_file.readline()  # Skip id line
        line = in_file.readline()
        counter += 1
    sys.stdout.write("{} reads has been loaded, initial estimate was {} reads\n".format(counter, estimated_reads))
    return reads_matrix[0:counter]
