__kernel void B64_N(
   __global uchar* SK_4, // 8 bits (1 nucleotide per cell)
   __global ulong* SK_10_64, // Output matrix
   const unsigned int k, // K-mer size
   const unsigned int s, // Number of K-mers
   const unsigned int cSK_10_64 // Number of K-mers
   )
{

   int x = get_global_id(0);
   int y = get_global_id(1);


   // FUNCTION: Splitting
   if ((x<k) && (y<s)) {
     SK_4[y*k + x] = SK_10_64[y*cSK_10_64 + ((k-1)/32-(k-x-1)/32)]>>(((k-x-1)%32)*2)&3;

    }

}
