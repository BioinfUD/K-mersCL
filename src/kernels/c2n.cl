/*
Este kernel transforma un conjunto de kmers a su represetaci'on en entero de 64btis. Los par'ametros son:
SK  = Matriz de kmers de entrada
SK_4 = Matriz donde se almacena la salida. 8 bits por elemento
k = Tamano del kmers
s = Numero de kmers en el conjunto
*/
__kernel void C2N(
   __global char* SK,
   __global uchar* SK_4, // 8 bits
   const unsigned int k,
   const unsigned int s)
{
   int i = get_global_id(0);
   int j = get_global_id(1);
   // Valido para que no exista un overflow

   if((i <= k-1)&& (j<=s-1))  {
     SK_4[(j*k)+i]=7;
     switch (SK[(j*k)+i]) {
       // Evaluo codigo ascii
       case 65:
        SK_4[(j*k)+i] = 0;
        break;
       case 67:
         SK_4[(j*k)+i] = 1;
         break;
       case 71:
         SK_4[(j*k)+i] = 2;
         break;
       case 84:
         SK_4[(j*k)+i] = 3;
         break;
        default:
         SK_4[(j*k)+i] = (j*k)+i;
          break;
     }
   }

}
