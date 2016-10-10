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
   int x = get_global_id(0);
   int y = get_global_id(1);
   // Valido para que no exista un overflow

   if((x <= k-1)&& (y<=s-1))  {
     switch (SK[(y*k)+x]) {
       // Evaluo codigo ascii
       case 65:
        SK_4[(y*k)+x] = 0;
        break;
       case 67:
         SK_4[(y*k)+x] = 1;
         break;
       case 71:
         SK_4[(y*k)+x] = 2;
         break;
       case 84:
         SK_4[(y*k)+x] = 3;
         break;
        default:
         SK_4[(y*k)+x] = (y*k)+x;
          break;
     }
   }

}
