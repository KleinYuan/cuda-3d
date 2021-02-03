# Issues

###  Invalid configuration argument

If you run `make run-error101`, you may see some errors like `Error has been yield: invalid configuration argument`. This mainly results from the block dimension is bigger than the device is capable. Namely, your GPU is not good enough.

For example, on 1080ti, if you keep using:

```
  size_t threads_per_block = 2048;
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
```

then, you shall see the error.

This is because 1080 ti only allows max 1024 threads per block. If you wonder to query those info, just do: `/usr/local/cuda/samples/1_Utilities/deviceQuery`. If this command does not work for you, follow [this instruction](https://github.com/KleinYuan/RGGNet/blob/master/MACHINE-SETUP.md#step5-test-cuda).

### No kernel image is available for execution on the device

If your GPU is not 2080ti (Turing Architecture), you shall see such errors when you `make run-error101` command: `no kernel image is available for execution on the device`. If you do not use this commands, the errors won't yield and you won't see any GPU code performed.

This is due to that the architecture code is wrong. By default, I set the arch as 

```
run-error101:
        nvcc -arch=sm_75 -o error101 error101.cu -run
```

You need to replace the `sm_75` with the correct one. Again, the device query above shall tell you the `CUDA Capability Major/Minor version number`. For example, 1080ti will give you:


```
Device 0: "GeForce GTX 1080 Ti"
  CUDA Driver Version / Runtime Version          10.2 / 10.2
  CUDA Capability Major/Minor version number:    6.1
......
```

The `6.1` here means `sm_61`.
