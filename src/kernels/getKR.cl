/* Argumentos del kernel
Kernel que genera una matriz de todos los kmers de  una lectura.
R Vector con caracteres de las lectura
KR  Vector de salida de los caracteres de los kmer
k tamano del kmer
l tamano de la lectura
*/

__kernel void getKR(
   __global uchar* R,
   __global uchar* KR,
   const unsigned int k,
   const unsigned int l
   )
{
   int i = get_global_id(0);
   int j = get_global_id(1);
   // Valido para que no exista un overflow
    if(i <= k-1 && j<=l-k)  {

       KR[(j*k)+i] = R[j+i];

   }
}
