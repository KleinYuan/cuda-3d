#include <unistd.h>
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define N 25
#define MAX 100

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void uniform_sample(curandState_t* states, float* numbers) {
  /* curand works like rand - except that it takes a state as a parameter */
  numbers[blockIdx.x] = curand_uniform(&states[blockIdx.x]);
}

// Mainly modified based on http://ianfinlayson.net/class/cpsc425/notes/cuda-random
int main() {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t* states;

  /* allocate space on the GPU for the random states */
  cudaMalloc((void**) &states, N * sizeof(curandState_t));

  /* invoke the GPU to initialize all of the random states */
  init<<<N, 1>>>(time(0), states);

  /* allocate an array of unsigned ints on the CPU and GPU */
  float host_data[N];
  float* device_data;
  cudaMalloc((void**) &device_data, N * sizeof(float));

  /* invoke the kernel to get some random numbers */
  uniform_sample<<<N, 1>>>(states, device_data);

  /* copy the random numbers back */
  cudaMemcpy(host_data, device_data, N * sizeof(float), cudaMemcpyDeviceToHost);

  /* print them out */
  for (int i = 0; i < N; i++) {
    printf("%f\n", host_data[i]);
  }

  /* free the memory we allocated for the states and numbers */
  cudaFree(states);
  cudaFree(device_data);

  return 0;
}
