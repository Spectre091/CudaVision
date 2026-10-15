%%writefile ex3b.cu
#include <cstdio>
#include <thrust/universal_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include "cv.h"

int main()
{
    thrust::universal_vector<int> vec{1};

    // TODO: replace each "???" with your guess: "CPU" or "GPU"
    thrust::for_each(thrust::device, vec.begin(), vec.end(),
        [] __host__ __device__(int) { cv::where_am_I("???"); });
    cudaDeviceSynchronize();  // flush GPU printf before the next line prints

    thrust::for_each(thrust::host, vec.begin(), vec.end(),
        [] __host__ __device__(int) { cv::where_am_I("???"); });

    cv::where_am_I("???");
    return 0;
}