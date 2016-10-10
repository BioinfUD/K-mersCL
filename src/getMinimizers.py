import pyopencl as cl
import numpy as np
from time import time
import scipy

#Kmer size
k = 33
#Number of kmers
s = 12000000
#minimizers size
m = 4
# Number of possible minimizers
ms = k-m+1


# K-mer generation
total_bases = k*s
h_SK_4 = np.random.randint(4, size=total_bases).astype("uint8").reshape((s,k))

s = h_SK_4.shape[0] # Number of kmers
k = h_SK_4.shape[1] # Kmer size


# Output memory allocation
h_ML_4 = np.ndarray((s, m)).astype(np.uint8)
h_ML_P = np.ndarray((s, 1)).astype(np.uint8)


contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/getMinimizers.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
getMinimizers = programa.getMinimizers
getMinimizers.set_scalar_arg_dtypes([None, None, None, None, None, np.uint32, np.uint32, np.uint32, np.uint32])

###############################################
# Memory allocation and copy in device
###############################################
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)

# Output matrix
d_ML_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_ML_4.nbytes)
d_ML_P = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_ML_P.nbytes)

#Intermedial Matrix
size_ML_10_32 = (4 * s * 1)  # Bytes
size_MS_10_32 = (4 * s * ms)  # Bytes
d_ML_10_32 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_ML_10_32)
d_MS_10_32 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_MS_10_32)
h_MS_10_32 = np.ndarray((s, ms)).astype(np.uint32)
h_ML_10_32 = np.ndarray((s, 1)).astype(np.uint32)


# Kernel Execution
rango_global = (ms, s , m)
rango_local =  (ms, 1, m)

t1 = time()
getMinimizers(cola, rango_global, rango_local, d_SK_4, d_ML_4, d_ML_P, d_ML_10_32, d_MS_10_32, k, s, m, ms)
cola.finish()
print "Kernel took {} seconds in the execution".format(time()-t1)


#cl.enqueue_copy(cola, hTMP, d_TMP_P)
cl.enqueue_copy(cola, h_ML_P, d_ML_P)
cl.enqueue_copy(cola, h_ML_4, d_ML_4)
cl.enqueue_copy(cola, h_MS_10_32, d_MS_10_32)
cl.enqueue_copy(cola, h_ML_10_32, d_ML_10_32)
cola.finish()

print "Input Matrix"
print h_SK_4[250:255]
print "MS_10_32"
print h_MS_10_32[250:255]
print "ML_10_32"
print h_ML_10_32[250:255]
print "Output Matrix ML_4"
print h_ML_4[250:255]
print "Output vector ML_P"
print h_ML_P[250:255]
