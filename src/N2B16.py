# Este script define una matriz de kmers en base 4 y la transforma a una matriz de kmers en base 10 de 8 bits.
import pyopencl as cl
import numpy as np
from time import time
# BASE 4 A CARACTERES
#Memoria para lectura en host
SK_4 = [[0,0,0,3,3,3,3,1],[0,0,0,2,3,2,1,1],[0,0,0,1,2,3,1,1]] # Matriz con KMERS
#SK_4 = [[3,3,3,3],[2,3,2,1],[1,2,3,1],[1,2,3,1],[0,1,2,3]] # Matriz con KMERS
h_SK_4 =np.ndarray((len(SK_4), len(SK_4[0]))).astype(np.uint8)
h_TMP =np.ndarray((len(SK_4), len(SK_4[0]))).astype(np.uint16)
h_SK_4[:] = SK_4


# TODO: Numpy, anadir columnas 0, tener el k de entrada de salida.
# Si cols de SK_4 no es multiplo de 4

# h_SK_2_16 Storage for output
h_SK_2_16=np.ndarray((h_SK_4.shape[0], h_SK_4.shape[1]/8)).astype(np.uint16)

S = h_SK_4.shape[0] # Numeero de kmes
K = h_SK_4.shape[1] # Tamano del kmer

print "#############  KMERS BASE 4 A BASE 2 (16bits) #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/N2B16.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
N2B16 = programa.N2B16
N2B16.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32])


#Copio datos de entrada a dispositivo
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
#Bufer para la matriz intermedia
d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_TMP.nbytes)
#Bufer para la salida
d_SK_2_16 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_16.nbytes)
# Dimensiones de ejecucion
rango_global = (K, S)
# Ejecucion del kernel
N2B16(cola, rango_global, None, d_SK_4, d_TMP, d_SK_2_16, K, S)
cola.finish()
# Traigo datos
cl.enqueue_copy(cola, h_SK_2_16, d_SK_2_16)


print "Matriz entrada"
print h_SK_4
print "Matriz salida"
print h_SK_2_16
