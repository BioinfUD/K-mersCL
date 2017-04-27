from Bio.Seq import Seq
from Bio import SeqIO
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


def file_to_matrix(filename="/tmp/outfile.txt", r=180):
    r = r if r else 180
    in_file = open(filename, "rU")
    parser = SeqIO.parse(in_file, "fasta")
    record = parser.next()
    A = array(map(base_to_int,list(record.seq.tostring())), dtype=np.uint32)
    for record in parser:
        newrow = array(map(base_to_int,list(record.seq.tostring())), dtype=np.uint32)
        if newrow.shape[0] == r:
            A = np.vstack([A, newrow])
    return A
