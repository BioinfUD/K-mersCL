__kernel void B642B32(
   __global ulong* SK_2_64, // Matriz con kmers de salida (base 4)
   __global uint* SK_2_32, // Matriz representaci'on compacta (4 kmers por byte), entrada.
   const unsigned int cSK_2_64, // Numero de columnas de SK_2_16
   const unsigned int cSK_2_32, // Numero de columnas de SK_2_16
  const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);

   // Splitting

   if( (i <= cSK_2_32-1) && (j<=s-1) )  {
      SK_2_32[(j*cSK_2_32)+i] =(ulong) (SK_2_64[(j*cSK_2_64)+(i/2)] >>  (64-32*(i%2+1))) & 4294967295;

    }




}
