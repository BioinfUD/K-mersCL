# Este script transforma una matriz de kmers en base 4 a una matriz de  caracteres
import pyopencl as cl
import numpy as np
from time import time
# BASE 4 A CARACTERES
# TODO: Usar kernel N2C
#Memoria para lectura en host, kmers en base 4
SK_4 = [[0,2, 3,1],[2,3,2,1],[1,2,3,1],[1, 2,3,1],[0,1,2,3]] # Matriz con KMERS
h_SK_4 = np.ndarray((len(SK_4), len(SK_4[0]))).astype(np.uint8)
h_SK_4[:] = SK_4

#Output matrix
h_SK = np.chararray((len(SK_4), len(SK_4[0]))) # Defino dimensiones de matriz


s = h_SK_4.shape[0] # Numeero de kmes
k = h_SK_4.shape[1] # Tamano del kmer

print "#############  KMERS  BASE 4 A KMERS CARACTER #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/n2c.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
N2C = programa.N2C
N2C.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])
# Vector de salida
d_SK = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK.nbytes)
# Copy input data from host to device
d_SK4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
# Execution Range
rango_global = (k, s)
rango_local = (k, 1)
# Kernel Execution
N2C(cola, rango_global, rango_local, d_SK, d_SK4, k, s)
cola.finish()
# Retrieve output
cl.enqueue_copy(cola, h_SK, d_SK)
print h_SK
