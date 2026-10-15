#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <random>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/universal_vector.h>

float cpu_median(const thrust::universal_vector<float>& vec)
{
    thrust::universal_vector<float> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());
    return sorted_vec[sorted_vec.size() / 2];
}

float cpu_variance(const thrust::universal_vector<float>& x, float mean)
{
    float sum = 0.0f;
    for (const float value : x) {
        const float difference = value - mean;
        sum += difference * difference;
    }
    return sum / static_cast<float>(x.size());
}

float median(thrust::universal_vector<float> vec)
{
    // TODO: use the respective thrust::sort implementation for the GPU.
    cudaDeviceSynchronize();
    return vec[vec.size() / 2];
}

// This one is quite a bit harder.
float variance(const thrust::universal_vector<float>& x, float mean)
{
    // TODO: build a transform iterator with a __host__ __device__ lambda
    //       that computes (xi - mean) * (xi - mean),
    // Use:  thrust::make_transform_iterator
    

    // TODO: fill in the reduce parameters
    const float sum = thrust::reduce(
        /* TODO */, /* TODO */, /* TODO */ + x.size(), 0.0f);
        cudaDeviceSynchronize();
    return sum / static_cast<float>(x.size());
}

int main()
{
    constexpr int n = 1 << 20;
    thrust::universal_vector<float> x(n);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(50.0f, 10.0f);
    for (int i = 0; i < n; ++i) {
        x[i] = dist(rng);
    }

    const auto mean_start = std::chrono::steady_clock::now();
    const float mean =
        thrust::reduce(thrust::device, x.begin(), x.end(), 0.0f) / n;
    cudaDeviceSynchronize();
    const auto mean_end = std::chrono::steady_clock::now();

    const auto cpu_median_start = std::chrono::steady_clock::now();
    const float cpu_med = cpu_median(x);
    const auto cpu_median_end = std::chrono::steady_clock::now();

    const auto median_start = std::chrono::steady_clock::now();
    const float med = median(x);
    const auto median_end = std::chrono::steady_clock::now();

    const auto cpu_variance_start = std::chrono::steady_clock::now();
    const float cpu_var = cpu_variance(x, mean);
    const auto cpu_variance_end = std::chrono::steady_clock::now();

    const auto variance_start = std::chrono::steady_clock::now();
    const float var = variance(x, mean);
    const auto variance_end = std::chrono::steady_clock::now();

    const auto milliseconds = [](auto start, auto end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    std::printf("n        = %d\n", n);
    std::printf("mean     = %f  (%f ms)\n",
                mean, milliseconds(mean_start, mean_end));
    std::printf("cpu median   = %f  (%f ms)\n",
                cpu_med, milliseconds(cpu_median_start, cpu_median_end));
    std::printf("gpu median   = %f  (%f ms)\n",
                med, milliseconds(median_start, median_end));
    std::printf("cpu variance = %f  (%f ms)\n",
                cpu_var, milliseconds(cpu_variance_start, cpu_variance_end));
    std::printf("gpu variance = %f  (%f ms)\n",
                var, milliseconds(variance_start, variance_end));
    return 0;
}