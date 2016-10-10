__kernel void N_32(
   __global uchar* SK_4, // 8 bits (1 nucleotide per cell)
   __global uint* SK_10_32, // Output matrix
   const unsigned int k, // K-mer size
   const unsigned int s, // Number of K-mers
   const unsigned int cSK_10_32 // Number of K-mers
   )
{

   int x = get_global_id(0);
   int y = get_global_id(1);

   // Initialize output matrix
   if ((x<cSK_10_32) && (y<s)) {
        SK_10_32[(y*cSK_10_32) + x] = 0;
    }

   // FUNCTION: Get decimal value
   if ((x<k) && (y<s)) {

     ulong n = (ulong) SK_4[y*k + x]<<((k-x-1)%16)*2;
     int l = (k-1)/16  - (k-x-1)/16;
     atomic_or(&SK_10_32[(y*cSK_10_32) + l], n );

    }

}
