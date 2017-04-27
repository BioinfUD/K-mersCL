import numpy as np
from numpy import array
# File to matrix

def base_to_int(b):
    if b=="A":
        return 0
    if b=="C":
        return 1
    if b=="G":
        return 2
    if b=="T":
        return 3
    else:
        return 99 # Not valid value


def int_to_base(i):
    if b==0:
        return "A"
    if b==1:
        return "C"
    if b==2:
        return "G"
    if b==3:
        return "T"
    else:
        return "U" # Not valid value


def file_to_matrix(filename="/tmp/outfile.txt", r=190):
    r = r if r else 190
    in_file = open("/tmp/outfile.txt", "rU")
    parser = SeqIO.parse(in_file, "fasta")
    r = parser.next()
    A = array(map(base_to_int,list(r.seq.tostring())), dtype=np.uint32)
    for r in parser:
        newrow = array(map(base_to_int,list(r.seq.tostring())), dtype=np.uint32)
        if newrow.shape[0] == r:
            A = np.vstack([A, newrow])
    return A


def cut_minimizer_matrix(minimizer_matrix, counter_vector):
    n_reads = counter_vector.shape[0]
    cutted_matrix = []
    for i in range(n_reads):
        # Get the values
        pass
    return cutted_matrix
