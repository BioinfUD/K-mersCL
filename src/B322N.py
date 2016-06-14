# Este script define una matriz de kmers en base 4 y la transforma a una matriz de kmers en base 10 de 8 bits.
import pyopencl as cl
import numpy as np
from time import time
import scipy
# BASE 10 A BASE 4

# Matriz de entrada
SK_2_32 = [[1,66912997],[48562917,1],[48562613,1]]
h_SK_2_32 =np.ndarray((len(SK_2_32), len(SK_2_32[0]))).astype(np.uint32)
h_SK_2_32[:] = SK_2_32

#Parametros
k = 9

cSK_4 = h_SK_2_32.shape[1]*16 # Ncols matriz salida
S = h_SK_2_32.shape[0] # Numero de kmes

# Matriz de salida
h_SK_4 = np.ndarray((S, cSK_4)).astype(np.uint8)


print "#############  KMERS BASE 2 (32 bits - 16 bases) A BASE 4 (8bits por base) #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B322N.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B322N = programa.B322N
B322N.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])

#Copio datos de entrada a dispositivo
d_SK_2_32 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_32)

#Bufer para la salida
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_4.nbytes)

# Dimensiones de ejecucion
rango_global = (cSK_4, S)
# Ejecucion del kernel
B322N(cola, rango_global, None, d_SK_4, d_SK_2_32, cSK_4, S)
cola.finish()
# Traigo datos
cl.enqueue_copy(cola, h_SK_4, d_SK_4)
# Borro columnas que sobran
#h_SK_4 = scipy.delete(h_SK_4,range(cSK_4-k),1)
print "Matriz de entrada"
print h_SK_2_32
print "Matriz de salida"
print h_SK_4
