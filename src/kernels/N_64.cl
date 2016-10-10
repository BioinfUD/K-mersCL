#pragma OPENCL EXTENSION cl_khr_global_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics:  enable
#pragma OPENCL EXTENSION cl_khr_local_int64_base_atomics : enable


__kernel void N_64(
   __global uchar* SK_4, // 8 bits (1 nucleotide per cell)
   __global ulong* SK_10_64, // Output matrix
   const unsigned int k, // K-mer size
   const unsigned int s, // Number of K-mers
   const unsigned int cSK_10_64 // Number of K-mers
   )
{

   int x = get_global_id(0);
   int y = get_global_id(1);

   // Initialize output matrix
   if ((x<cSK_10_64) && (y<s)) {
    SK_10_64[(y*cSK_10_64) + x] = 0;
    }

   // FUNCTION: Get decimal value
   if ((x<k) && (y<s)) {

     ulong n = (ulong) SK_4[y*k + x]<<((k-x-1)%32)*2;
     int l = (k-1)/32  - (k-x-1)/32;
     atomic_or(&SK_10_64[(y*cSK_10_64) + l], n );

    }

}
