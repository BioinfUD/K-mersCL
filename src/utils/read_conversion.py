from Bio.Seq import Seq
from Bio import SeqIO
import numpy as np
from numpy import array, ndarray
import os


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


def int_to_base(b):
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


def integer_to_bases(number, m=4):
    mask = 0b11
    bases = ""
    for c in (range(m-1, -1, -1)):
        #print "number: {}, number bin {}, c*2: {}".format(number, bin(number), c*2)
        base = int_to_base((number >> c*2) & mask)
        bases = bases + base
    return bases


def file_to_matrix(filename="/tmp/outfile.txt", r=180):
    in_file = open(filename, "rU")
    parser = SeqIO.parse(in_file, "fasta")
    record = parser.next()
    avg_record_bytes = r + len(str(record.id)) + 2
    input_file_size = os.path.getsize(filename)
    estimated_reads = input_file_size/avg_record_bytes
    print "Estimated number of reads: {}".format(estimated_reads)
    A = ndarray(shape=(estimated_reads, r), dtype=np.uint32)
    counter = 0	    
    for record in parser:
        A[counter] = map(base_to_int,list(record.seq.tostring()))
        if counter%100000 == 0:
            print "{} reads has been loaded".format(counter)
	counter+=1
    print "{} reads has been loaded, initial estimate was {} reads".format(counter, estimated_reads)
    return A[0:counter]
