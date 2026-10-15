%%writefile cv.h
#pragma once
#include <cstdio>

namespace cv {

__host__ __device__ inline const char* execution_space()
{
#ifdef __CUDA_ARCH__
    return "GPU";
#else
    return "CPU";
#endif
}

__host__ __device__ inline void where_am_I(const char* guess)
{
    printf("I think I'm on the %s, and I'm actually on the %s\n",
           guess, execution_space());
}

} // namespace cv