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
   int x = get_global_id(0);
   int y = get_global_id(1);
   // Valido para que no exista un overflow

   if((x <= k-1)&& (y<=s-1))  {
     switch (SK_4[(y*k)+x]) {
       // Evaluo codigo ascii
       case 0:
        SK[(y*k)+x] = 65;
        break;
       case 1:
         SK[(y*k)+x] = 67;
         break;
       case 2:
         SK[(y*k)+x] = 71;
         break;
       case 3:
         SK[(y*k)+x] = 84;
         break;
        default:
         SK_4[(y*k)+x] = (y*k)+x;
          break;
     }
   }

}
