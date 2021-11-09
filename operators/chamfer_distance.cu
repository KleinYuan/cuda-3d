#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <limits>
#include <numeric>
#include <algorithm>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess){
      fprintf(stderr, "CUDA Runtime Error: %s \n", cudaGetErrorString(result));
  }
  return result;
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Chamfer Loss is equal to D = E(min euclidean_dist(S1->S2)) + E(min euclidean_dist(S2->S1))
// http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf
// http://vision.cs.utexas.edu/378h-fall2015/slides/lecture4.pdf
// You can choose to not do sqrt ... which does not matter that much
__global__ void chamferDistanceOneWay(float *xyz1, float *xyz2, float *result, int num_points_xyz1, int num_points_xyz2, int dim_points){
  int idxWithinTheGridX = threadIdx.x + blockDim.x * blockIdx.x;
  int gridStrideX = gridDim.x * blockDim.x;
  int idxWithinTheGridY = threadIdx.y + blockDim.y * blockIdx.y;
  int gridStrideY = gridDim.y * blockDim.y;

  for(int i = idxWithinTheGridX; i < num_points_xyz1; i += gridStrideX)
  {
    float x1=xyz1[i*dim_points+0];
    float y1=xyz1[i*dim_points+1];
    float z1=xyz1[i*dim_points+2];

    for(int j = idxWithinTheGridY; j < num_points_xyz2; j += gridStrideY)
    {
        float x2=xyz2[j*dim_points+0];
        float y2=xyz2[j*dim_points+1];
        float z2=xyz2[j*dim_points+2];
        float dist = sqrt(powf((x1 - x2), 2) + powf((y1 - y2), 2) + powf((z1 - z2), 2));
        atomicMin(&result[i], dist);
        // printf("Comparing %f with %f\n", result[i], dist);
    }
  }
}

// TODO(kaiwen): implement the gradient

int main() {
  const int num_points_xyz1 = 3;
  const int num_points_xyz2 = 2;
  const int dim_points = 3;
  // random_xyz1 and random_xyz2 are two point clouds with 3 points
  // each point is ordered in xyzxyzxyz.. order
  // namely, the first point of random_xyz1 is (x=0.1, y=1.1, z=2.1), and so on
  float random_xyz1 [num_points_xyz1 * dim_points] = {0.1, 1.1, 2.1, -0.1, 5.1, -8.1, -23.1, -24.2, 12.2};
  float random_xyz2 [num_points_xyz2 * dim_points] = {5.2, 5.1, -32.2, -100.3, 0.0, 4.1};
  float init_chamfer_distance_forward [num_points_xyz1];
  std::fill_n(init_chamfer_distance_forward, num_points_xyz1, std::numeric_limits<float>::max());
  float init_chamfer_distance_backward [num_points_xyz2];
  std::fill_n(init_chamfer_distance_backward, num_points_xyz2, std::numeric_limits<float>::max());
  float *host_xyz1 = &random_xyz1[0];
  float *host_xyz2 = &random_xyz2[0];
  float *chamfer_distance_forward = &init_chamfer_distance_forward[0];
  float *chamfer_distance_backward = &init_chamfer_distance_backward[0];
  float* device_xyz1;
  float* device_xyz2;
  float* device_chamfer_distance_forward;
  float* device_chamfer_distance_backward;
  float chamfer_distance = 0.f;

  const size_t num_bytes_xyz1 = sizeof(float) * num_points_xyz1 * dim_points;
  const size_t num_bytes_xyz2 = sizeof(float) * num_points_xyz2 * dim_points;

  // initialize the data in device
  cudaMallocManaged(&device_xyz1, num_bytes_xyz1);
  cudaMallocManaged(&device_xyz2, num_bytes_xyz2);
  cudaMallocManaged(&device_chamfer_distance_forward, sizeof(float) * num_points_xyz1);
  cudaMallocManaged(&device_chamfer_distance_backward, sizeof(float) * num_points_xyz2 );
  cudaMemcpy(device_xyz1, host_xyz1, num_bytes_xyz1, cudaMemcpyHostToDevice);
  cudaMemcpy(device_xyz2, host_xyz2, num_bytes_xyz2, cudaMemcpyHostToDevice);
  cudaMemcpy(device_chamfer_distance_forward, chamfer_distance_forward, sizeof(float) * num_points_xyz1, cudaMemcpyHostToDevice);
  cudaMemcpy(device_chamfer_distance_backward, chamfer_distance_backward, sizeof(float) * num_points_xyz2 , cudaMemcpyHostToDevice);
  //
  chamferDistanceOneWay<<<dim3(1), dim3(num_points_xyz1, num_points_xyz2)>>>
    (device_xyz1, device_xyz2, device_chamfer_distance_forward, num_points_xyz1, num_points_xyz2, dim_points);
  chamferDistanceOneWay<<<dim3(1), dim3(num_points_xyz2, num_points_xyz1)>>>
    (device_xyz2, device_xyz1, device_chamfer_distance_backward, num_points_xyz2, num_points_xyz1, dim_points);
  cudaMemcpy(chamfer_distance_forward, device_chamfer_distance_forward, sizeof(float) * num_points_xyz1, cudaMemcpyDeviceToHost);
  cudaMemcpy(chamfer_distance_backward, device_chamfer_distance_backward, sizeof(float) * num_points_xyz2, cudaMemcpyDeviceToHost);
  // Sync
  checkCuda(cudaDeviceSynchronize());

  float chamfer_distance_forward_sum = 0.f;
  float chamfer_distance_backward_sum = 0.f;

  for (int i = 0; i < num_points_xyz1; i++) {
    chamfer_distance_forward_sum = chamfer_distance_forward_sum + chamfer_distance_forward[i];
    printf("Forward Chamfer Distance is %f\n", chamfer_distance_forward[i]);
  }

  chamfer_distance_forward_sum = chamfer_distance_forward_sum / num_points_xyz1;

  for (int i = 0; i < num_points_xyz2; i++) {
    chamfer_distance_backward_sum = chamfer_distance_backward_sum + chamfer_distance_backward[i];
    printf("Backward Chamfer Distance is %f\n", chamfer_distance_backward[i]);
  }

  chamfer_distance_backward_sum = chamfer_distance_backward_sum / num_points_xyz2;

  chamfer_distance = chamfer_distance_forward_sum + chamfer_distance_backward_sum;
  printf("Mean Forward Chamfer Distance is %f\n", chamfer_distance_forward_sum);
  printf("Mean Backward Chamfer Distance is %f\n", chamfer_distance_backward_sum);
  printf("Final Chamfer Distance is %f\n", chamfer_distance);
  /*
    Forward Chamfer Distance is 34.907021
    Forward Chamfer Distance is 24.675900
    Forward Chamfer Distance is 60.255623
    Backward Chamfer Distance is 24.675900
    Backward Chamfer Distance is 81.308617
    Mean Forward Chamfer Distance is 39.946182
    Mean Backward Chamfer Distance is 52.992256
    Final Chamfer Distance is 92.938438
  */
  cudaFree(device_xyz1);
  cudaFree(device_xyz2);
  cudaFree(device_chamfer_distance_forward);
  cudaFree(device_chamfer_distance_backward);

  return 0;
}
