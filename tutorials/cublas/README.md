# CUBLAS

## Find Max Mag
Finding the max magnitude (NOT VALUE!!!) in a CUDA array can be easily done in cublas.
The example of [find_max_mag.cu](find_max_mag.cu) is for that.

```````
nvcc -lcublas -o find_max find_max.cu -run
# or make find_max_mag
```````

will yield the following:

```
Max Id is 6
```

As you can see `{0.1, 1.1, 2.1, 3.1, 5.1, -8.1}`, the 5th is the largest.

It shall be noted that 
- [cublasIsamax](https://docs.nvidia.com/cuda/cublas/index.html#cublasi-lt-t-gt-amax) function yields the max magnitude instead of max value. So, abs(-8.1) is the largest one here.
- CUBLAS is based on FORTRAN, namely, it starts from 1 insetad of 0 and the return is thereby 6 instead of 5
