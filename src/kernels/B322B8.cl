__kernel void B322B8(
   __global uint* SK_2_32, // Matriz con kmers de salida (base 4)
   __global uchar* SK_2_8, // Matriz representaci'on compacta (4 kmers por byte), entrada.
   const unsigned int c_sk_2_32, // Numero de columnas de SK_2_16
   const unsigned int c_sk_2_8, // Numero de columnas de SK_2_16
  const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);
    // 11

   if( (i <= c_sk_2_8-1) && (j<=s-1) )  {
     // I / N -> N es 16/8
      SK_2_8[(j*c_sk_2_8)+i] = SK_2_32[(j*c_sk_2_32)+(i/4)] >> (32 - (i%4+1)*8);



    }




}
