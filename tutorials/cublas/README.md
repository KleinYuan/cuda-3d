# CUBLAS

## Find Max
Finding the max value in a CUDA array can be easily done in cublas.
The example of [find_max.cu](find_max.cu) is for that.

```
nvcc -lcublas -o find_max find_max.cu -run
# or make find_max
```

will yield the following:

```
Max Id is 5
```

As you can see `{0.1, 1.1, 2.1, 3.1, 5.1, 4.1}`, the 5th is the largest.

It shall be noted that CUBLAS is based on FORTRAN, namely, it starts from 1 insetad of 0.
