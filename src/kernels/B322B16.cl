__kernel void B322B16(
   __global uint* SK_2_32, // Matriz con kmers de salida (base 4)
   __global ushort* SK_2_16, // Matriz representaci'on compacta (4 kmers por byte), entrada.
   const unsigned int cSK_2_32, // Numero de columnas de SK_2_16
   const unsigned int cSK_2_16, // Numero de columnas de SK_2_16
  const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);

   // Splitting

   if( (i <= cSK_2_16-1) && (j<=s-1) )  {
     // I / N -> N es 16/8
      SK_2_16[(j*cSK_2_16)+i] = (SK_2_32[(j*cSK_2_32)+(i/2)] >>  (32-16*(i%2+1))) & 65535 ;


    }




}
