# Get kmers complement using heterogeneous computing
import pyopencl as cl
import numpy as np
from time import time
from python_functions.utils import memory_usage_psutil


k = 32
s = 10000000
total_bases = (k * s)
h_SK_4 = np.random.randint(4, size=total_bases).astype("uint8").reshape((s,k))

# Output matrix
h_CASK_4 = np.ndarray(h_SK_4.shape).astype(np.uint8)
h_CA_I = np.ndarray(s).astype(np.uint8)

print "Get canonical: Doing for K {} S {}".format(k,s)

# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)

# Program for reverse complement
codigo_kernel_c = open("kernels/getCanonical.cl").read()
programa_c = cl.Program(contexto, codigo_kernel_c).build()
getCanonical = programa_c.getCanonical
#getCanonical.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32])
getCanonical.set_scalar_arg_dtypes([None, None, None, None, None, np.uint32, np.uint32, np.uint32])



# Copy input data from host to device
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
# Output matrix
d_CASK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_CASK_4.nbytes)
# Vector with flags
d_CA_I = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_CA_I.nbytes)
# Intermedial matrix
cSK_10_64  =  (k-1)/32 + 1
size_SK_10_64 = cSK_10_64 * s * 8
d_SK_10_64 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_SK_10_64)
d_CASK_10_64 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_SK_10_64)


# Kernel execution for canonical kmer
print "Executing kernel"
t1 = time()
# NDRange
rango_global = (k, s)
rango_local = (k,1)

getCanonical(cola, rango_global, rango_local, d_SK_4, d_CA_I, d_CASK_4, d_SK_10_64, d_CASK_10_64, k, s, cSK_10_64)
cola.finish()

print "Kernel took {} seconds in the execution".format(time()-t1)
cl.enqueue_copy(cola, h_CASK_4, d_CASK_4)
cl.enqueue_copy(cola, h_CA_I, d_CA_I)
cola.finish()
print h_SK_4[:10]
print h_CASK_4[:10]
print h_CA_I[:10]
