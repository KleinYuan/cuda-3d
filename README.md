# CUDA-3D

- [X] Tutorials of CUDA fundamentals
  - [X] Hello World
  - [X] Atomic Functions
  - [X] Thrust
  - [X] cuBLAS
  - [ ] cuDNN
  - [ ] Memory
- [ ] Popular operators for 3d point clouds
  - [X] Uniform Sampling
  - [X] Euclidean Distance
  - [ ] Chamfer Distance
  - [ ] Earth Mover Distance
  - [ ] Euclidean Clustering
  - [ ] Depth Clustering

# Structure

```
.
├── operators
│   ├── euclidean_distance.cu
│   ├── Makefile
│   ├── README.md
│   └── uniform_sample.cu
├── README.md
└── tutorials
    ├── atomic
    │   ├── atomic101.cu
    │   ├── atomic_find_max.cu
    │   ├── Makefile
    │   └── README.md
    ├── cublas
    │   ├── find_max_mag.cu
    │   ├── Makefile
    │   └── README.md
    ├── hello_world
    │   ├── coordinating_parallel.cu
    │   ├── error101.cu
    │   ├── error_macro.cu
    │   ├── grid_stride.cu
    │   ├── hello_world.cu
    │   ├── loop_accelerate.cu
    │   ├── Makefile
    │   ├── matrix_mul.cu
    │   ├── memory101.cu
    │   ├── mismatched_config.cu
    │   ├── README.md
    │   ├── TROUBLESHOOT.md
    │   └── vector_add.cu
    ├── memory
    └── thrust
        ├── find_max.cu
        ├── Makefile
        └── README.md
```
