%%writefile ex3.cu
#include <cstdio>
#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

int main()
{
    thrust::universal_vector<float> temp{42, 24, 50};
    float k = 0.5f, room = 20;

    // ===== EXERCISE 3 START =====
    // TODO: write a __host__ __device__ lambda (capture k and room by value)
    //       that takes a temperature t and returns t + k * (room - t)

    // TODO: use thrust::transform on the GPU to apply the lambda
    //       to every element of temp, writing the results back into temp
    // ===== EXERCISE 3 END =====

    printf("temp = {");
    for (size_t i = 0; i < temp.size(); i++)
        printf("%g%s", (float)temp[i], i + 1 < temp.size() ? ", " : "");
    printf("}   (expected {31, 22, 35})\n");
    return 0;
}