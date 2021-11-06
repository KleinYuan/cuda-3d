#include <stdio.h>
#include <assert.h>
#include <math.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess){
      fprintf(stderr, "CUDA Runtime Error: %s \n", cudaGetErrorString(result));
  }
  return result;
}

__global__ void sumEuclideanDistance(float *xyz1, float *xyz2, float *result, int num_points, int dim_points){
  int idxWithinTheGrid = threadIdx.x + blockDim.x * blockIdx.x;
  int gridStride = gridDim.x * blockDim.x;

  for(int i = idxWithinTheGrid; i < num_points; i += gridStride)
  {
    float x1=xyz1[i*dim_points+0];
    float y1=xyz1[i*dim_points+1];
    float z1=xyz1[i*dim_points+2];

    float x2=xyz2[i*dim_points+0];
    float y2=xyz2[i*dim_points+1];
    float z2=xyz2[i*dim_points+2];

    float dist = sqrt(powf((x1 - x2), 2) + powf((y1 - y2), 2) + powf((z1 - z2), 2));

    atomicAdd(result, dist);

  }
}

// TODO(kaiwen): implement the gradient

int main() {
  const int num_points = 3;
  const int dim_points = 3;
  // random_xyz1 and random_xyz2 are two point clouds with 3 points
  // each point is ordered in xyzxyzxyz.. order
  // namely, the first point of random_xyz1 is (x=0.1, y=1.1, z=2.1), and so on
  float random_xyz1 [num_points * dim_points] = {0.1, 1.1, 2.1, -0.1, 5.1, -8.1, -23.1, -24.2, 12.2};
  float random_xyz2 [num_points * dim_points] = {5.2, 5.1, -32.2, -100.3, 0.0, 4.1, -6.1, -12.2, 5.8};
  float init_euclidean_distance = 0.f;
  float *host_xyz1 = &random_xyz1[0];
  float *host_xyz2 = &random_xyz2[0];
  float *mean_euclidean_distance = &init_euclidean_distance;
  float* device_xyz1;
  float* device_xyz2;
  float* device_sum_euclidean_distance;

  const size_t num_bytes = sizeof(float) * num_points * dim_points;

  // initialize the data in device
  cudaMallocManaged(&device_xyz1, num_bytes);
  cudaMallocManaged(&device_xyz2, num_bytes);
  cudaMallocManaged(&device_sum_euclidean_distance, sizeof(float));
  cudaMemcpy(device_xyz1, host_xyz1, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_xyz2, host_xyz2, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_sum_euclidean_distance, &init_euclidean_distance, sizeof(float), cudaMemcpyHostToDevice);
  //
  sumEuclideanDistance<<<num_points, 1>>>(device_xyz1, device_xyz2, device_sum_euclidean_distance, num_points, dim_points);
  cudaMemcpy(mean_euclidean_distance, device_sum_euclidean_distance, sizeof(float), cudaMemcpyDeviceToHost);
  // Sync
  checkCuda(cudaDeviceSynchronize());
  *mean_euclidean_distance = *mean_euclidean_distance / (float)num_points;
  printf("Mean Euclidean Distance is %f\n", *mean_euclidean_distance);
  // You will get "Mean Euclidean Distance is 52.582134"
  // (0.1,1.1,2.1) and (5.2,5.1,-32.2) ground truth distance is 34.907019
  // (-0.1, 5.1, -8.1) and (-100.3,0.0,4.1) ground truth distance is 101.068739
  // (-23.1, -24.2, 12.2) and (-6.1, -12.2, 5.8) ground truth distance is 21.770622
  // the overall ground truth mean is 52.58212666666666, which matches the CUDA one
  cudaFree(device_xyz1);
  cudaFree(device_xyz2);
  cudaFree(device_sum_euclidean_distance);

  return 0;
}
