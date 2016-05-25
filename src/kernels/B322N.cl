// Kernel obtiene base 4 a partir de kmers en representacion de 32 bit 
__kernel void B322N(
   __global uchar* SK_4, // Matriz con kmers de salida (base 4)
   __global uint* SK_2_32, // Matriz representaci'on compacta (16 kmers por 4 byte), entrada.
   const unsigned int c_sk_4, // Numero de columnas de SK
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);
    // 11

   if((i <= c_sk_4-1)&& (j<=s-1) )  {
     //uchar mask = 192 >> ((i%4)*2); // 192 1100000 corrido
     SK_4[(j*c_sk_4)+i]  = (SK_2_32[(j*(c_sk_4/16))+(i/16)]>>(30-((i%16)*2))) & 3;

    }




}
