#include <cstdio>
#include <chrono>
#include <random>
#include <algorithm>
#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

void cpu_relax(thrust::universal_vector<float>& temp, float k, float room)
{
    const auto relax = [k, room](float t) {
        return t + k * (room - t);
    };
    std::transform(temp.begin(), temp.end(), temp.begin(), relax);
}

void gpu_relax(thrust::universal_vector<float>& temp, float k, float room)
{
    // TODO: write a __host__ __device__ lambda (capture k and room by value)
    //       that takes a temperature t and returns t + k * (room - t)


    // TODO: use thrust::transform on the GPU to apply the lambda
    //       to every element of temp, writing the results back into temp
    cudaDeviceSynchronize();
}

int main()
{
    constexpr std::size_t n = 1 << 20;
    constexpr float k = 0.5f;
    constexpr float room = 20.0f;
    thrust::universal_vector<float> initial(n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-20.0f, 100.0f);
    for (std::size_t i = 0; i < n; ++i) {
        initial[i] = distribution(rng);
    }

    thrust::universal_vector<float> cpu_temp = initial;
    thrust::universal_vector<float> gpu_temp = initial;

    const auto cpu_start = std::chrono::steady_clock::now();
    cpu_relax(cpu_temp, k, room);
    const auto cpu_end = std::chrono::steady_clock::now();

    const auto gpu_start = std::chrono::steady_clock::now();
    gpu_relax(gpu_temp, k, room);
    const auto gpu_end = std::chrono::steady_clock::now();

    const auto milliseconds = [](auto start, auto end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    printf("n          = %zu\n", n);
    printf("cpu first  = %f  (%f ms)\n",
           cpu_temp[0], milliseconds(cpu_start, cpu_end));
    printf("gpu first  = %f  (%f ms)\n",
           gpu_temp[0], milliseconds(gpu_start, gpu_end));
    return 0;
}