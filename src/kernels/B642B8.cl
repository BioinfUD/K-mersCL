__kernel void B642B8(
   __global ulong* SK_2_64, // Matriz con kmers de salida (base 4)
   __global uchar* SK_2_8, // Matriz representaci'on compacta (4 kmers por byte), entrada.
   const unsigned int cSK_2_64, // Numero de columnas de SK_2_16
   const unsigned int cSK_2_8, // Numero de columnas de SK_2_16
  const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);

   //Splitting
   if( (i <= cSK_2_8-1) && (j<=s-1) )  {
     // I / N -> N es 16/8
      SK_2_8[(j*cSK_2_8)+i] = (SK_2_64[(j*cSK_2_64)+(i/8)] >>  (64-8*(i%8+1))) & 255;



    }




}
