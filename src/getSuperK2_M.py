import numpy as np
import pyopencl as cl
import sys
from sys import argv
from time import time

from utils.read_conversion import file_to_matrix
from utils.superkmer_utils import cut_minimizer_matrix, extract_superkmers

def extract_superkmers(minimizer_matrix, m=4):
    n_superkmers = 0
    for row in minimizer_matrix:
        print "superkmers: {}".format(str(row))
        for v in row:
            minimizer = (v & 0b11111111111111000000000000000000) >> 18
            pos = (v & 0b00000000000000111111111100000000) >> 8
            size = v & 0b00000000000000000000000011111111
            end =  pos + size
            minimizer_str = str(minimizer)
            print "Valor {} Min {},  ini {}, size {}, end{}".format(v, minimizer, pos, size, end)
            n_superkmers+=1
    return n_superkmers

def customize_kernel_template(X, k, m, r, kernel_template):
    if (k <= 36):
        nt3 = 6
    elif (k <= 64):
        nt3 = 8
    elif (k <= 100):
        nt3 = 10
    else:
        nt3 = 12
    nm = r - m + 1;
    lsd = X // m;
    ts = ((nm-1)//lsd) + 1
    nt2 = ((nm-1)//ts) + 1
    a_mask = (2**((2*m)-2)) - 1
    params_template = {'read_size': str(r), 'second_mt': str(nt2), 'nt_max': str(max(nt2, nt3)), 'a_mask': str(a_mask)}
    kernel_template = kernel_template.format(**params_template)
    return kernel_template

def getSuperK_M(input_file, output_path, r):
    # Kernel parameters
    sys.stdout.write("Loading sequences from file {}\n".format(input_file))
    h_R2M_G = file_to_matrix(input_file, r)
    print h_R2M_G
    nr = h_R2M_G.shape[0]
    r = h_R2M_G.shape[1]
    m = 7
    k = 31
    nmk = k - m + 1
    X = nmk
    # OpenCL things
    contexto = cl.create_some_context()
    cola = cl.CommandQueue(contexto)
    codigo_kernel = open("kernels/getSuperK2_M.cl.tpl").read()
    codigo_kernel = customize_kernel_template(X, k, m, r, codigo_kernel)
    programa = cl.Program(contexto, codigo_kernel).build()
    getSuperK_M = programa.getSuperK_M
    getSuperK_M.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32,np.uint32])
    # Copy data from host to device
    sys.stdout.write("Copying data from host to device memory \n")
    d_R2M_G = cl.Buffer(contexto, cl.mem_flags.COPY_HOST_PTR, hostbuf=h_R2M_G)
    # Kernel parameters
    sys.stdout.write("Ejecutando X hilos, X: {}\n".format(X))
    rango_global = (X, nr)
    rango_local = (X, 1)
    # Output matrix
    nm = r - m + 1;
    h_counters = np.empty((nr,1)).astype(np.uint32)
    d_counters = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_counters.nbytes)
    # Execution
    getSuperK_M(cola, rango_global, rango_local, d_R2M_G, d_counters, nr, r, k, m)
    cola.finish()
    sys.stdout.write("Execution finished, copying data from device to host memory\n")
    cl.enqueue_copy(cola, h_counters, d_counters)
    cl.enqueue_copy(cola, h_R2M_G, d_R2M_G)
    # Cut the output matrix based on counters
    sys.stdout.write("Cutting the matrix based on available superkmers\n")
    minimizer_matrix = cut_minimizer_matrix(h_R2M_G, h_counters)
    del(h_R2M_G)
    sys.stdout.write("Writing superkmers to disk\n")
    extract_superkmers(minimizer_matrix, m=7)
    # extract_superkmers(minimizer_matrix, input_file, output_path, m=m)

if __name__ == "__main__":
    """
    Execution:
    python getSuperK_M filename.fasta outputpath N
    filename <- Fasta file with reads
    outputpath <- Folder where the buckets will be placed (must exists)
    N <- Read size (optional)
    """
    input_file = argv[1]
    output_path = argv[2]
    r = int(argv[3]) if argv[3] else None
    getSuperK_M(input_file, output_path, r)
