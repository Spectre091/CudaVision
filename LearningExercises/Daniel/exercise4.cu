#include <thrust/universal_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <chrono>

float cpu_median(const thrust::universal_vector<float>& vec)
{
    thrust::universal_vector<float> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());
    return sorted_vec[sorted_vec.size() / 2];
}

float gpu_median(const thrust::universal_vector<float>& vec)
{
    thrust::universal_vector<float> sorted_vec = vec;
    // TODO: Replace std::sort with the respective thrust implementation to run on the GPU.
    std::sort(thrust::device, sorted_vec.begin(), sorted_vec.end());
    cudaDeviceSynchronize();
    return sorted_vec[sorted_vec.size() / 2];
}

int main() {
    // ignore -- populate universal vector with data.
    constexpr std::size_t element_count = 1 << 20; // around 1 million elements.
    thrust::universal_vector<float> v(element_count);
    for (std::size_t i = 0; i < element_count; ++i) {
        v[i] = static_cast<float>((i * 1103515245ULL + 12345ULL) % 1000000ULL);
    }
    // ignore -- 

    double c_time;
    double g_time;
    float c_med;
    float g_med;

    auto cpu_start = std::chrono::steady_clock::now();
    c_med = cpu_median(v);
    auto cpu_end = std::chrono::steady_clock::now();
    c_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    auto gpu_start = std::chrono::steady_clock::now();
    g_med = gpu_median(v);
    auto gpu_end = std::chrono::steady_clock::now();
    g_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    std::cout << "The cpu median is: " << c_med << " and took: "
              << c_time << " ms to calculate." << std::endl;
    std::cout << "The gpu median is: " << g_med << " and took: "
              << g_time << " ms to calculate." << std::endl;
    
    return 0;
}
