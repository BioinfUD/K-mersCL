/*
BASE 4 A CARACTERES
SK  = Matriz de kmers de salida
SK_4 = Matriz de kmers, entrada, base 4.
k = Tamano del kmers
s = Numero de kmers en el conjunto
*/
__kernel void N2C(
   __global uchar* SK,
   __global uchar* SK_4, // 8 bits
   const unsigned int k,
   const unsigned int s)
{
   int i = get_global_id(0);
   int j = get_global_id(1);
   // Valido para que no exista un overflow

   if((i <= k-1)&& (j<=s-1))  {
     switch (SK_4[(j*k)+i]) {
       // Evaluo codigo ascii
       case 0:
        SK[(j*k)+i] = 65;
        break;
       case 1:
         SK[(j*k)+i] = 67;
         break;
       case 2:
         SK[(j*k)+i] = 71;
         break;
       case 3:
         SK[(j*k)+i] = 84;
         break;
        default:
         SK_4[(j*k)+i] = (j*k)+i;
          break;
     }
   }

}
