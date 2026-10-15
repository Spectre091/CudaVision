%%writefile ex2.cu
#include <cstdio>
#include <thrust/universal_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include "cv.h"

int main()
{
    thrust::universal_vector<float> vec{1, 2, 3, 54, 7};

    // ===== EXERCISE START =====

    printf("--- cpu ---\n");
    // TODO: use thrust::for_each with the execution policy that runs on the CPU.
    //       Pass a lambda (with the right annotation for CPU code) that takes an int val
    //       and prints: "printing <val> on <cv::execution_space()>"


    printf("--- gpu ---\n");
    // TODO: use thrust::for_each with the execution policy that runs on the GPU.
    //       Pass a lambda (with the right annotation for GPU code) that takes an int val
    //       and prints the same message.


    cudaDeviceSynchronize();  // flush GPU printf output

    // ===== EXERCISE END =====
    return 0;
}