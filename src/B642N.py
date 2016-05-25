# Este script define una matriz de kmers en base 4 y la transforma a una matriz de kmers en base 10 de 8 bits.
import pyopencl as cl
import numpy as np
from time import time
import scipy
# BASE 10 A BASE 4

# Matriz de entrada
SK_2_64 = [[6148914691236517205],[287390336450102269],[123006441751904693]]
h_SK_2_64 =np.ndarray((len(SK_2_64), len(SK_2_64[0]))).astype(np.uint64)
h_SK_2_64[:] = SK_2_64

#Parametros
k = 9

c_sk_4 = h_SK_2_64.shape[1]*32 # Ncols matriz salida
S = h_SK_2_64.shape[0] # Numero de kmes

# Matriz de salida
h_SK_4 = np.ndarray((S, c_sk_4)).astype(np.uint8)


print "#############  KMERS BASE 2 (64 bits - 32 bases) A BASE 4 (8bits por base) #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B642N.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B642N = programa.B642N
B642N.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])

#Copio datos de entrada a dispositivo
d_SK_2_64 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_64)

#Bufer para la salida
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_4.nbytes)

# Dimensiones de ejecucion
rango_global = (c_sk_4, S)
# Ejecucion del kernel
B642N(cola, rango_global, None, d_SK_4, d_SK_2_64, c_sk_4, S)
cola.finish()
# Traigo datos
cl.enqueue_copy(cola, h_SK_4, d_SK_4)
# Borro columnas que sobran
#h_SK_4 = scipy.delete(h_SK_4,range(c_sk_4-k),1)
print "Matriz de entrada"
print h_SK_2_64
print "Matriz de salida"
print h_SK_4
