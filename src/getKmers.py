#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pyopencl as cl
import numpy as np
from time import time
import pickle
import sys

def full_string(file_name, L):
    full_string = ""
    read_counter =  0
    print "Reading file"
    with open(file_name, "r") as file:
        continuar = "Yes"
        while continuar:
            # Ignore first line
            file.readline()
            # Add read to string
            full_string += file.readline().strip()
            read_counter += 1
            # Ignore two lines
            file.readline()
            continuar = file.readline()
    print "Done reading, {} reads processed".format(read_counter)
    return full_string, read_counter

if len(sys.argv) != 4:
    # Use test data
    R = "TCAGCTACGTCAGCTACAGTCA"
    L = len(R)
    K = 3
    S = 1
else:
    K = int(sys.argv[1])
    L = int(sys.argv[2])
    file_name = str(sys.argv[3])
    R, S = full_string(file_name, L)

R_h = np.chararray((len(R)))
R_h [:] = list(R)

############ EXTRAER KMERS ##################
print "############ EXTRAER KMERS ##################"
# OpenCL things
codigo_kernel = open("kernels/getKR.cl").read()
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
programa = cl.Program(contexto, codigo_kernel).build()
#Memoria para salida
numero_celdas = ((L-K+1)*K)
KR_h = np.chararray((numero_celdas))
#Memoria en dispositivo
#Copio R_h y Apartor espacio para KR_d
R_d =  cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=R_h)
KR_d = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, (L-K+1)*K )
#Dimensiones
rango_global = (K, L-K+1)
rango_local = (K, 1)
#Kernel Execution
getKR = programa.getKR
getKR.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])
rtime = time()
getKR(cola, rango_global, None, R_d, KR_d, K, L)
cola.finish()
rtime = time() - rtime
print "El kernel se ejecutó en ", rtime, "segundos"
#Retrieve output
cl.enqueue_copy(cola, KR_h, KR_d)
print "Kmer size: {}, Number of reads: {}, Read size: {}".format(K, S, L)
print "Input matrix"
print R_h
print "Output matrix"
print KR_h
