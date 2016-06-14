# Este script define una matriz de kmers en base 4 y la transforma a una matriz de kmers en base 10 de 8 bits.
import pyopencl as cl
import numpy as np
from time import time
import scipy
# BASE 10 A BASE 4

# Matriz de entrada
SK_2_16 = [[255,1,1021],[1,1,64],[1,0,220]]
h_SK_2_16 =np.ndarray((len(SK_2_16), len(SK_2_16[0]))).astype(np.uint16)
h_SK_2_16[:] = SK_2_16

#Parametros
k = 9

cSK_4 = h_SK_2_16.shape[1]*8 # Ncols matriz salida
S = h_SK_2_16.shape[0] # Numero de kmes

# Matriz de salida
h_SK_4 = np.ndarray((S, cSK_4)).astype(np.uint8)


print "#############  KMERS BASE 2 (16 bits - 8 bases) A BASE 4 (8bits por base) #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B162N.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B162N = programa.B162N
B162N.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])

#Copio datos de entrada a dispositivo
d_SK_2_16 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_16)

#Bufer para la salida
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_4.nbytes)

# Dimensiones de ejecucion
rango_global = (cSK_4, S)
# Ejecucion del kernel
B162N(cola, rango_global, None, d_SK_4, d_SK_2_16, cSK_4, S)
cola.finish()
# Traigo datos
cl.enqueue_copy(cola, h_SK_4, d_SK_4)
# Borro columnas que sobran
#h_SK_4 = scipy.delete(h_SK_4,range(cSK_4-k),1)
print "Matriz de entrada"
print h_SK_2_16
print "Matriz de salida"
print h_SK_4
