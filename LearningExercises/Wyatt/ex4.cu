%%writefile exercise4.cu
#include <cstdio>
#include <random>
#include <thrust/universal_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>

// ===== EXERCISE 4 START =====
float median(thrust::universal_vector<float> vec)
{
    // TODO: sort vec on the GPU and return the middle element
    return 0.0f;
}

float variance(const thrust::universal_vector<float>& x, float mean)
{
    // TODO: build a transform iterator with a __host__ __device__ lambda
    //       that computes (xi - mean) * (xi - mean),
    //       then reduce it on the GPU and divide by x.size()
    return 0.0f;
}
// ===== EXERCISE 4 END =====

int main()
{
    const int n = 1 << 20;
    thrust::universal_vector<float> x(n);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(50.0f, 10.0f);  // mean 50, stddev 10
    for (int i = 0; i < n; i++) x[i] = dist(rng);

    float mean = thrust::reduce(thrust::device, x.begin(), x.end()) / n;

    printf("n        = %d\n", n);
    printf("mean     = %f  (expected ~50)\n", mean);
    printf("median   = %f  (expected ~50)\n", median(x));
    printf("variance = %f  (expected ~100)\n", variance(x, mean));
    return 0;
}
