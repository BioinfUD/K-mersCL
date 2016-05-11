
import pyopencl as cl
import numpy as np
from time import time

#Memoria para lectura en host
SK = [["A","C", "T"],["C","T","G"],["T","G","C"],["G", "C", "A"],["C","A","T"]] # Matriz con KMERS
#Conversi'on a numpy
SK_h = np.chararray((len(SK), len(SK[0]))) # Defino dimensiones de matriz
SK_h[:] = SK
S = SK_h.shape[0] # Numeero de kmes
K = SK_h.shape[1] # Tamano del kmer
print "#############  KMERS  A BASE 4 #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/c2n.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
C2N = programa.C2N
C2N.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])
# Vector de salida
numero_celdas = S * K
SK4_h = np.empty(numero_celdas).astype(np.uint8)
SK4_d = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, SK4_h.nbytes)
#Copio datos de entrada a dispositivo
SK_d = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=SK_h)
# Dimensiones de ejecucion
rango_global = (K, S)
# Ejecucion del kernel
C2N(cola, rango_global, None, SK_d, SK4_d, K, S)
cola.finish()
# Traigo datos
cl.enqueue_copy(cola, SK4_h, SK4_d)
print SK4_h
