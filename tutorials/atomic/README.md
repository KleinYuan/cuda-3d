# Atomic Functions

## Atomic 101

An atomic function performs a read-modify-write atomic operation on one 32-bit or 64-bit word residing in global or shared memory. 
For example, atomicAdd() reads a word at some address in global or shared memory, 
adds a number to it, and writes the result back to the same address. 
The operation is atomic in the sense that it is guaranteed to 
be performed without interference from other threads. In other words, 
no other thread can access this address until the operation is complete

Atomic functions can only be used in device functions.

This 101 example is to show the behavior of two parallel atomic functions:

``` 
  // Perform atomicAddIdx, adding the index to the element
  atomicAddIdx<<<4, 1>>>(device_data, N, 0, 2);
  // Perform atomicAddIdxSquared, adding the index * index to the element
  atomicAddIdxSquared<<<4, 1>>>(device_data, N, 2, 4);
```

You can run `make atomic101` to see

```
[atomicAddIdx] Perform GPU code 0.100000 + 0...at address: 0x7f6cae000000 
[atomicAddIdx] Perform GPU code 2.100000 + 1...at address: 0x7f6cae000010 
[atomicAddIdxSquared] Perform GPU code 6.100000 + 4...at address: 0x7f6cae000020 
[atomicAddIdxSquared] Perform GPU code 12.100000 + 9...at address: 0x7f6cae000030 
Updated number on 0 is 0.100000
Updated number on 1 is 2.100000
Updated number on 2 is 6.100000
Updated number on 3 is 12.100000
```

As you can see, the `atomicAddIdx` updates the first two elements via adding the index: `0.1 -> 0.1` and `1.1 -> 2.1`;
`atomicAddIdxSquared` updates the last two elements via adding the index squared: `2.1 -> 6.1` and `3.1 -> 12.1`.

If we switch the `start_idx` and `stop_idx` so that the two kernels update the same address:

Namely, 
```
  atomicAddIdx<<<4, 1>>>(device_data, N, 0, 4);
  atomicAddIdxSquared<<<4, 1>>>(device_data, N, 0, 4);
```

Delete the executable and re-run the make command again, you will get:

```
[atomicAddIdx] Perform GPU code 6.100000 + 9...at address: 0x7f96f4000030 
[atomicAddIdx] Perform GPU code 4.100000 + 4...at address: 0x7f96f4000020 
[atomicAddIdx] Perform GPU code 2.100000 + 1...at address: 0x7f96f4000010 
[atomicAddIdx] Perform GPU code 0.100000 + 0...at address: 0x7f96f4000000 
[atomicAddIdxSquared] Perform GPU code 8.100000 + 4...at address: 0x7f96f4000020 
[atomicAddIdxSquared] Perform GPU code 15.100000 + 9...at address: 0x7f96f4000030 
[atomicAddIdxSquared] Perform GPU code 3.100000 + 1...at address: 0x7f96f4000010 
[atomicAddIdxSquared] Perform GPU code 0.100000 + 0...at address: 0x7f96f4000000 
Updated number on 0 is 0.100000
Updated number on 1 is 3.100000
Updated number on 2 is 8.100000
Updated number on 3 is 15.100000
```

As you can see both kernel updates the value in the same address:

- 0x7f96f4000030: 0.1 + 0 + 0 = 0.1
- 0x7f96f4000010: 1.1 + 1 + 1 = 3.1
- 0x7f96f4000020: 2.1 + 2 + 2*2 = 8.1
- 0x7f96f4000030: 3.1 + 3 + 3*3 = 15.1

## Reference

- https://on-demand.gputechconf.com/gtc/2013/presentations/S3101-Atomic-Memory-Operations.pdf

## Atomic Max

Using Atomic function to find the max value in a given array:

```
make atomic_find_max
```

And it will yield:

```
nvcc -arch=sm_75 -o atomic_find_max atomic_find_max.cu -run
[findMaxKernel] Perform GPU code, comparing 0.000000 (old)  and 2.100000 (new) at 0x7ff656000020 
[findMaxKernel] Perform GPU code, comparing 2.100000 (old)  and 3.100000 (new) at 0x7ff656000030 
[findMaxKernel] Perform GPU code, comparing 3.100000 (old)  and 0.100000 (new) at 0x7ff656000000 
[findMaxKernel] Perform GPU code, comparing 3.100000 (old)  and 1.100000 (new) at 0x7ff656000010 
max_val is 3.100000
```
which is correct.

You can compare this with the thrust and cublas ones:

- [thrust/find_max.cu](../thrust/find_max.cu)
- [cublas/find_max_mag.cu](../cublas/find_max_mag.cu)
