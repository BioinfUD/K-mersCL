// Kernel obtiene base 4 a partir de kmers en representacion de 64 bit

__kernel void D2N(
   __global uchar* SK_4, // Matriz con kmers de salida (base 4)
   __global ulong* SK_10, // Matriz representaci'on compacta (32 bases por 8 bytes), entrada.
   const unsigned int cSK_4, // Numero de columnas de SK
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);

    // Splitting
   if((i <= cSK_4-1)&& (j<=s-1) )  {
     SK_4[(j*cSK_4)+i]  = (SK_10[(j*(cSK_4/32))+(i/32)]>>(62-((i%32)*2))) & 3;

    }




}
