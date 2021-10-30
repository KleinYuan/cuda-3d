# Thrust

## Find Max

Referring to this [stackoverflow question](https://stackoverflow.com/questions/27925979/thrustmax-element-slow-in-comparison-cublasisamax-more-efficient-implementat)

Finding the max value in a CUDA array can be easily done in thrust.
The example of [find_max.cu](find_max.cu) is for that.

```````
nvcc -o find_max find_max.cu -run
# or make find_max
```````

will yield the following:

```
Max Id is 4
```

You can compare this example with [tutorial/cublas/find_max_mag.cu](../cublas/find_max_mag.cu), which yields the max ID as 6.

