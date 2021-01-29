#include <stdio.h>

void runCPULoop(int N)
{
  printf("Perform CPU code ...\n");
  for (int i = 0; i < N; ++i)
  {
	  printf("-> iteration %d\n", i);
  }
}


__global__ void runGPULoop()
{
  printf("Perform GPU code ...\n");
  printf("~> iteration %d\n", threadIdx.x);
}	

int main()
{
  

  runCPULoop(10);
  
  runGPULoop<<<1, 10>>>();
  cudaDeviceSynchronize();
}
