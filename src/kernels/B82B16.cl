__kernel void B82B16(
   __global uchar* SK_2_8, // Matriz con kmers de salida (base 4)
   __global ushort* SK_2_16, // Matriz representaci'on compacta (4 kmers por byte), entrada.
   const unsigned int c_sk_2_8, // Numero de columnas de SK_2_8
   const unsigned int c_sk_2_16, // Numero de columnas de SK_2_8
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);
    // 11

   if( (i <= c_sk_2_16-1) && (j<=s-1) )  {
      //  SK_2_16 [(j*c_sk_2_16)+i] = j;
      SK_2_16[(j*c_sk_2_16)+i] = (SK_2_8[(j*c_sk_2_8)+(i*2)]<< 8)|SK_2_8[(j*c_sk_2_8)+(i*2)+1];
      //SK_2_16[(j*c_sk_2_16)+i] = SK_2_16[(j*(c_sk_2_16))+i] << 8;
      //SK_2_16[(j*c_sk_2_16)+i] |= ;


    }




}
