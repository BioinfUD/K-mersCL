// Kernel obtiene base 4 a partir de kmers en representacion de 16 bit
__kernel void B162N(
   __global uchar* SK_4, // Matriz con kmers de salida (base 4)
   __global short* SK_2_16, // Matriz representaci'on compacta (8 kmers por 4 bytes), entrada.
   const unsigned int cSK_4, // Numero de columnas de SK
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);

   // Splitting
   if((i <= cSK_4-1)&& (j<=s-1) )  {
     SK_4[(j*cSK_4)+i]  = (SK_2_16[(j*(cSK_4/8))+(i/8)]>>(14-((i%8)*2))) & 3;

    }



}
