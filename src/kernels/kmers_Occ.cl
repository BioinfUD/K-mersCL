__kernel void suma3v(
   __global float* a,
   __global float* b,
   __global float* c,
   __global float* r,
   const unsigned int count)
{
   int i = get_global_id(0);
   if(i < count)  {
       r[i] = a[i] + b[i] + c[i];
   }
}
