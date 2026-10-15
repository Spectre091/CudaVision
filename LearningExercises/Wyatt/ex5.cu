%%writefile exercise5.cu
#include <cstdio>
#include <random>
#include <thrust/universal_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>

// ===== EXERCISE 5 START =====
struct squared_diff {
    float mean;
    // TODO: add a __host__ __device__ operator() that takes a float xi
    //       and returns (xi - mean) * (xi - mean)
};

float variance(const thrust::universal_vector<float>& x, float mean)
{
    // TODO: build a transform iterator using squared_diff{mean},
    //       then reduce it on the GPU and divide by x.size()
    return 0.0f;
}
// ===== EXERCISE 5 END =====

int main()
{
    const int n = 1 << 20;
    thrust::universal_vector<float> x(n);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(50.0f, 10.0f);
    for (int i = 0; i < n; i++) x[i] = dist(rng);

    float mean = thrust::reduce(thrust::device, x.begin(), x.end()) / n;

    printf("n        = %d\n", n);
    printf("mean     = %f  (expected ~50)\n", mean);
    printf("variance = %f  (expected ~100)\n", variance(x, mean));
    return 0;
}