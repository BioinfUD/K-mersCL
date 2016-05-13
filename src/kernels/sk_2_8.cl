__kernel void N2B8(
   __global uchar* SK_4, // 8 bits (1 kmer por posicion) entrada // # DE COLUMNAS, DEBE SER MULTIPLO DE 4. SI K NO ES MULTIPLO DE 4, SE DEBE AGREGAR LAS COLUMNAS INICIALES IZQUIERDA= CON 0
   __global uchar* SL_SK_4, // Matriz intermedia donde se guardan corrimientos
   __global uchar* SK_2_8, // 8 bits (4 kmers) salida, n cols = c_sk_4 / 4
   const unsigned int c_sk_4, // Numero de columnas de SK
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);
   if((i <= c_sk_4-1)&& (j<=s-1))  {

   SL_SK_4[(j*c_sk_4)+i] = SK_4[(j*c_sk_4)+i] << (6 - (i % (4*2)));
    }
    if ((i<(c_sk_4/4)) && (j<=s-1)) {
      SK_2_8[(j*c_sk_4/4)+i] = SL_SK_4[(j*c_sk_4)+(i*4)] | SL_SK_4[(j*c_sk_4)+(i*4)+1] | SL_SK_4[(j*c_sk_4)+(i*4)+2] |  SL_SK_4[(j*c_sk_4)+(i*4)+3];
    }


   // c_sk_4/4 n cols SK_2_8

   // Obtengo numero
   // Valido para que no exista un overflow



}
