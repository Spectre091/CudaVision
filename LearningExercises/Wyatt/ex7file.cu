// Exercise 7: turn row totals into row averages
//
// Build and run in Google Colab (start each line with !):
//   nvcc -arch=native --extended-lambda -o ex7 ex7file.cu
//   ./ex7
//
// Same grid as Exercise 6, but now we want the average temperature of each
// row instead of the total. Run it once before you change anything and write
// down the GB/s, then do the TODOs and run it again.


//%%writefile ex7file.cu
#include <cstdio>
#include <cmath>
#include <chrono>

#include <thrust/universal_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/discard_iterator.h>

// ===== EXERCISE 7 START =====
struct mean_functor {
    int width;
    __host__ __device__ float operator()(float x) const { return x / width; }
};

thrust::universal_vector<float> row_averages(int height, int width,
    const thrust::universal_vector<int>& row_ids,
    const thrust::universal_vector<float>& temp)
{
    thrust::universal_vector<float> means(height);
    // TODO 1: wrap means.begin() in thrust::make_transform_output_iterator
    //         with mean_functor{width}, so each sum is divided as it's written
    auto means_output = means.begin();                                             //change This!!!
    auto row_ids_begin = row_ids.begin();  // paste your Exercise 6 line here, same code change. 
    auto row_ids_end = row_ids_begin + temp.size(); //keep same.
    thrust::reduce_by_key(thrust::device, row_ids_begin, row_ids_end,
        temp.begin(), thrust::make_discard_iterator(), means_output);
    // TODO 2: delete this extra pass once means_output divides
    thrust::transform(thrust::device, means.begin(), means.end(),    //remove as you dont need anymore after applying ex7.
                      means.begin(), mean_functor{width});
    return means;
}
// ===== EXERCISE 7 END =====


// ---------- you don't need to change anything below this line ----------

// The heat grid: the top row is 90 degrees, everything else is 15.
thrust::universal_vector<float> make_grid(int height, int width)
{
    thrust::universal_vector<float> temp(static_cast<size_t>(height) * width, 15.0f);
    thrust::fill(thrust::device, temp.begin(), temp.begin() + width, 90.0f);
    return temp;
}

// The stored list of keys: the row number of every cell (0,0,0,...,1,1,1,...).
thrust::universal_vector<int> make_row_ids(int height, int width)
{
    thrust::universal_vector<int> row_ids(static_cast<size_t>(height) * width);
    thrust::tabulate(thrust::device, row_ids.begin(), row_ids.end(),
        [=] __host__ __device__ (int i) { return i / width; });
    return row_ids;
}

// The fastest your GPU's memory can move data, in GB/s.
double peak_bandwidth_gbs()
{
    int device = 0, clock_khz = 0, bus_bits = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device);
    return 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
}

int main()
{
    const int height = 16;
    const int width = 16777216;   // 16.7 million columns, about 1 GB of temperatures
    const int runs = 5;

    thrust::universal_vector<float> temp = make_grid(height, width);
    thrust::universal_vector<int> row_ids = make_row_ids(height, width);

    // Warm-up run: moves the data onto the GPU before we start timing.
    thrust::universal_vector<float> means = row_averages(height, width, row_ids, temp);
    cudaDeviceSynchronize();

    auto begin = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; r++) {
        means = row_averages(height, width, row_ids, temp);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    const double seconds = std::chrono::duration<double>(end - begin).count() / runs;
    const double gigabytes = temp.size() * sizeof(float) / 1e9;
    const double throughput = gigabytes / seconds;
    const double peak = peak_bandwidth_gbs();

    const float r0 = means[0], r1 = means[1], r2 = means[2];
    std::printf("row 0 = %g\nrow 1 = %g\nrow 2 = %g\n", r0, r1, r2);
    const bool correct = std::fabs(r0 - 90.0) < 0.01 && std::fabs(r1 - 15.0) < 0.01;
    std::printf("averages are %s\n\n", correct ? "correct" : "WRONG");

    std::printf("time per run:         %.2f ms\n", seconds * 1e3);
    std::printf("achieved throughput:  %.1f GB/s\n", throughput);
    if (peak > 0) {
        std::printf("your GPU's limit:     %.1f GB/s\n", peak);
        std::printf("percent of limit:     %.1f%%\n", 100.0 * throughput / peak);
    }
}
