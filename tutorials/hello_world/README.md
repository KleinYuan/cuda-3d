# Hello World

This hellow world project contains basic practice codes from the NVIDIA course [Fundamentals of Accelerated Computing with CUDA C/C++](https://courses.nvidia.com/courses/course-v1:DLI+C-AC-01+V1/about). The purpose is to provide fundamental examples to introduce the basic concepts, archs, syntax, coordinating-parallel-threads, memory management and debug, with native CUDA.


![coordinating](https://user-images.githubusercontent.com/8921629/106368230-f5cc0000-62fc-11eb-8384-1b7cf0a2347c.png)


The project is tested on a 2080ti GPU (sm_75):


# Hello World Hello World

- [X] [hello_world.cu](hello_world.cu)
- [X] [loop_accelerate.cu](loop_accelerate.cu)
- [X] [coordinating_parallel.cu](coordinating_parallel.cu)

![map](https://user-images.githubusercontent.com/8921629/106368236-054b4900-62fd-11eb-9cb9-a98deb86a566.png)

How to run it? Check the Makefile.

```
make run-hello-world
>>nvcc -arch=sm_75 -o hello-world hello_world.cu -run
>>Perform GPU code ...
>>Perform CPU code ...
>>Perform GPU code ...
>>Perform GPU code ...
>>Perform GPU code ...
>>Perform GPU code ...
>>Perform GPU code ...
>>Perform GPU code ...
>>Perform GPU code ...
>>Perform GPU code ...
>>Accessing the third thread of the second block!
```

# Memory

Go thru [C basics](https://www.learn-c.org/)  before moving forward...
Basically `malloc` ~> `cudaMallocManaged` and `free` ~> `cudaFree`.

in CPU:

```
int N = 2<<20;
int *p = (int *)malloc(N * sizeof(int));
free(p);
```

in GPU:

```
int N = 2<<20;
size_t size = N * sizeof(int);
int *p;

cudaMallocManaged(&p, size);
cudaFree(p);
```

It shall be noted that, when we mean `cudaMallocManaged`, it allocates memory that is accessable by Both CPU and GPU.

- [X] [memory101.cu](memory101.cu)

Best practices of CUDA memory management can be found [here](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations).

# Block Configuration Mismatch to the Number of Threads Needed

A common example has to do with the desire to choose optimal block sizes. For example, due to GPU hardware traits, blocks that contain a number of threads that are a multiple of 32 are often desirable for performance benefits. Assuming that we wanted to launch blocks each containing 256 threads (a multiple of 32), and needed to run 1000 parallel tasks (a trivially small number for ease of explanation), then there is no number of blocks that would produce an exact total of 1000 threads in the grid, since there is no integer value 32 can be multiplied by to equal exactly 1000.

This scenario can be easily addressed in the following way:

- Write an execution configuration that creates more threads than necessary to perform the allotted work.
- Pass a value as an argument into the kernel (N) that represents to the total size of the data set to be processed, or the total threads that are needed to complete the work.
- After calculating the thread's index within the grid (using tid+bid*bdim), check that this index does not exceed N, and only perform the pertinent work of the kernel if it does not.


- [X] [mismatched_config.cu](mismatched_config.cu)


# Strides

Either by choice, often to create the most performant execution configuration, or out of necessity, the number of threads in a grid may be smaller than the size of a data set. Consider an array with 1000 elements, and a grid with 250 threads (using trivial sizes here for ease of explanation). Here, each thread in the grid will need to be used 4 times. One common method to do this is to use a grid-stride loop within the kernel.

In a grid-stride loop, each thread will calculate its unique index within the grid using tid+bid*bdim, perform its operation on the element at that index within the array, and then, add to its index the number of threads in the grid and repeat, until it is out of range of the array. For example, for a 500 element array and a 250 thread grid, the thread with index 20 in the grid would:

- Perform its operation on element 20 of the 500 element array
- Increment its index by 250, the size of the grid, resulting in 270
- Perform its operation on element 270 of the 500 element array
- Increment its index by 250, the size of the grid, resulting in 520
- Because 520 is now out of range for the array, the thread will stop its work

CUDA provides a special variable giving the number of blocks in a grid, gridDim.x. Calculating the total number of threads in a grid then is simply the number of blocks in a grid multiplied by the number of threads in each block, gridDim.x * blockDim.x. With this in mind, here is a verbose example of a grid-stride loop within a kernel:

```
__global void kernel(int *a, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;

  for (int i = indexWithinTheGrid; i < N; i += gridStride)
  {
    // do work on a[i];
  }
}
```

- [X] [grid_stride.cu](grid_stride.cu)

![grid_stride](https://user-images.githubusercontent.com/8921629/106368246-198f4600-62fd-11eb-8a51-fb57b45c41c4.png)



# Error Handling

- [X] [error101.cu](error101.cu)
- [X] [error_macro.cu](error_macro.cu)

# Comprehensive Examples

- [X] [vector_add.cu](vector_add.cu)

# Advanced-1: dim3

```
Grids and blocks can be defined to have up to 3 dimensions. Defining them with multiple dimensions does not impact their performance in any way, but can be very helpful when dealing with data that has multiple dimensions, for example, 2d matrices. To define either grids or blocks with two or 3 dimensions, use CUDA's dim3 type as such:

```
dim3 threads_per_block(16, 16, 1);
dim3 number_of_blocks(16, 16, 1);
someKernel<<<number_of_blocks, threads_per_block>>>();
```

Given the example just above, the variables gridDim.x, gridDim.y, blockDim.x, and blockDim.y inside of someKernel, would all be equal to 16.
```

- [X] [matrix_mul.cu](matrix_mul.cu)

# Advanced-2: application

- [ ] [heat_conduct.cu](heat_conduct.cu)



