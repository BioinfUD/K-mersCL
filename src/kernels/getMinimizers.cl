__kernel void getMinimizers(
   __global uchar* SK_4, // Input matrix with base 4 representation of kmers
   __global uchar* ML_4, // Output matrix with minimizers
   __global uchar* ML_P, // Output matrix with positions of minimizers in k-mers
   __global uint* ML_10_32, // Output matrix with positions of minimizers in k-mers
   __global uint* MS_10_32, // Output matrix with positions of minimizers in k-mers
   const unsigned int k,  // Number of M-mer
   const unsigned int s,  // Rows of SK_4 (Total kmers)
   const unsigned int m,  // M-mer size
   const unsigned int ms  // Number of M-mer
 )
{

   int x = get_global_id(0);
   int y = get_global_id(1);
   int z = get_global_id(2);

   // FUNCTION: Get decimal value
   if ((x<ms) && (y<s) && (z<m)){
     MS_10_32[y*ms + x] = 0;

   }

   if ((y<s) && (z==0) && (x==0)) {
     ML_10_32[y] = 2147483648;
   }
   barrier(CLK_GLOBAL_MEM_FENCE);

  if ((x<ms) && (y<s) && (z<m)) // Hilos de procesamiento que ejecutan esta funcion
  {

    atomic_or(&MS_10_32[y*ms + x] , SK_4[y*k + (x+z)]<< (m-z-1)*2);
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
  // FUNCTION: Get minimal of minimizers
  if ((x<ms) && (y<s) && (z==0)) // Hilos de procesamiento que ejecutan esta funcion
  {

   atomic_min(&ML_10_32[y],MS_10_32[y*ms + x]);// Min Atomico

    // Es probable que se requiera una barrera de sincronizacion
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (MS_10_32[y*ms + x] == ML_10_32[y] ) {
        	ML_P[y]=x;
        }

  }

  // FUNCTION: Llenado de la matriz ML_4

  if ((x==0) && (y<s) && (z<m)) // Hilos de procesamiento que ejecutan esta funcion
  {
    ML_4 [y*m + z]=SK_4[y*k+(z+ML_P[y])];
  }





}
