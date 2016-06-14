// Kernel obtiene base 4 a partir de kmers en representacion de 8 bit

__kernel void B82N(
   __global uchar* SK_4, // Matriz con kmers de salida (base 4)
   __global uchar* SK_2_8, // Matriz representaci'on compacta (4 kmers por byte), entrada.
   const unsigned int cSK_4, // Numero de columnas de SK
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);

   // Splitting
   if((i <= cSK_4-1)&& (j<=s-1) )  {
     SK_4[(j*cSK_4)+i]  = (SK_2_8[(j*(cSK_4/4))+(i/4)]>>(6-((i%4)*2))) & 3;

    }




}
