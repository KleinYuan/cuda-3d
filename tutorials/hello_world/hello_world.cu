#include <stdio.h>

void runCPU()
{
  printf("Perform CPU code ...\n");
}


__global__ void runGPU()
{
  printf("Perform GPU code ...\n");

  if (threadIdx.x == 2 && blockIdx.x == 1){
  printf("Accessing the third thread of the second block!\n");
  }
}	

int main()
{
  
  // launch a kernal, 1 block, 1 thread / block
  runGPU<<<1, 1>>>();
  cudaDeviceSynchronize();

  runCPU();
  
  // launch another kernel, 2 blocks (or thread block), 4 threads / block 
  // <<<#blocks, #threads/block >>>: this is called configurations
  // <<<2, 4>>> is a two-block-4-threads grid.
  // Threads are grouped into thread blocks
  // Blocks are grouped into a grid --> highest entity in CUDA thread hierarchy
  runGPU<<<2, 4>>>();
  cudaDeviceSynchronize();
}
