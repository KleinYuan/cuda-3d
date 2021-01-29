#include <stdio.h>

void runCPULoop(int N)
{
  printf("Perform CPU code ...\n");
  for (int i = 0; i < N; ++i)
  {
	  printf("-> iteration %d\n", i);
  }
}


// threadIdx.x + blockIdx.x * blockDim.x
__global__ void runGPULoop()
{
  printf("Perform GPU code ...\n");
  printf("~> iteration %d\n", threadIdx.x + blockIdx.x * blockDim.x);
}	

int main()
{
  
  int N = 10;
  runCPULoop(N);
  
  runGPULoop<<<1, N>>>();
  cudaDeviceSynchronize();

  // Another way to run it with two blocks but 5 threads each
  runGPULoop<<<2, 5>>>();
  cudaDeviceSynchronize();
}
