# CUDA-3D

- [X] Tutorials of CUDA fundamentals
  - [X] Hello World
  - [X] Atomic Functions
  - [ ] Thrust
  - [X] cuBLAS
  - [ ] cuDNN
  - [ ] Memory
- [ ] Native CUDA implementations for 3d point clouds operations, feature engineering and basic algorithms

# Structure

```
.
├── operators
│   └── README.md
├── README.md
└── tutorials
    ├── atomic
    │   ├── atomic101.cu
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
    └── memory
```
