/*
#include <stdio.h>

// CPU Version
void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

void addVectorsInto(float *result, float *a, float *b, int N)
{
  for(int i = 0; i < N; ++i)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  a = (float *)malloc(size);
  b = (float *)malloc(size);
  c = (float *)malloc(size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  addVectorsInto(c, a, b, N);

  checkElementsAre(7, c, N);

  free(a);
  free(b);
  free(c);
}

*/

/* GPU Version */

#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess){
      fprintf(stderr, "CUDA Runtime Error: %s \n", cudaGetErrorString(result));
  }
  return result;
}

__global__ void initWith(float num, float *a, int N)
{
  
  int idxWithinTheGrid = threadIdx.x + blockDim.x * blockIdx.x;
  int gridStride = gridDim.x * blockDim.x;
  
  for(int i = idxWithinTheGrid; i < N; i += gridStride)
  {
    a[i] = num;
  }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{

int idxWithinTheGrid = threadIdx.x + blockDim.x * blockIdx.x;
  int gridStride = gridDim.x * blockDim.x;
  
  for(int i = idxWithinTheGrid; i < N; i += gridStride)
  {
    result[i] = a[i] + b[i];
  }
  
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);
  
  int num_blocks = 256;
  int threads_per_block = 256;
  
  initWith<<<num_blocks, threads_per_block>>>(3, a, N);
  initWith<<<num_blocks, threads_per_block>>>(4, b, N);
  initWith<<<num_blocks, threads_per_block>>>(0, c, N);

  addVectorsInto<<<num_blocks, threads_per_block>>>(c, a, b, N);
  checkCuda(cudaDeviceSynchronize());

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

