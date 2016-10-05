#pragma OPENCL EXTENSION cl_khr_global_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics:  enable
#pragma OPENCL EXTENSION cl_khr_local_int64_base_atomics : enable

__kernel void N2D(
   __global uchar* SK_4, // 8 bits (1 nucleotide per cell)
   volatile __global ulong* SK_10, // 64 bits (32 bits per kmer)
   const unsigned int cSK_4, // Number of columns of input matrix
   const unsigned int cSK_10, // Number of columns of input matrix
   const unsigned int s
 ) // Number of kmers
{

   int i = get_global_id(0);
   int j = get_global_id(1);
   if((i <= cSK_4-1)&& (j<=s-1))  {
     SK_10[(j*cSK_10)+i] = 0;
    }

    if((i <= cSK_4-1)&& (j<=s-1))  {
      ulong n  = (ulong)SK_4[(j*cSK_4)+i] << (ulong) (64 - 2*((i%32)+1));
      atomic_or(&SK_10[(j*cSK_10)+i/32], n );
      //SK_10[(j*cSK_10)+i/16] = n;
    //  atomic_inc(&SK_10[(j*cSK_10)+i/16] );
  //    atomic_inc(&SK_10[(j*cSK_10)+i/16] );
      //atomic_inc(&SK_10[(j*cSK_10)+i/16] );
     }

}
