__kernel void B82N(
   __global uchar* SK_4, // Matriz con kmers de salida (base 4)
   __global uchar* SK_2_8, // Matriz representaci'on compacta (4 kmers por byte), entrada.
   const unsigned int k, //N cols matriz salida
   const unsigned int c_sk_4, // Numero de columnas de SK
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);
    // 11

   if((i <= c_sk_4-1)&& (j<=s-1) )  {
     //uchar mask = 192 >> ((i%4)*2); // 192 1100000 corrido
     SK_4[(j*c_sk_4)+i]  = (SK_2_8[(j*(c_sk_4/4))+(i/4)]>>(6-((i%4)*2))) & 3;

    }




}
