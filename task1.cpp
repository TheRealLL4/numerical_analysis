// clang -fopenmp task1.cpp -otask1.exe

// All tests below use BLOCK_SIZE=8, DESIRED_ERROR=1e-1, laplacian=test_laplacian and boundary=test_boundary.
// Tests were conducted on a i7-7700K machine, that has 4 cores and 8 hyperthreads
//  size iterations seconds  (1 thread / 8 threads)    speedup
// ~100   2285        0.21             / 2285 0.18       1.16
// ~200   4264        1.67             / 4264 1.3        1.28
// ~1000  8500        90.3             / 8500 24.6       3.67
// ~2000  9578       978.8             / 9578 230.87     4.24

// Results seem consistent with that in the book. Since the block size is dependent solely
// on the cache line size, small grids do not get almost any benefit from parallelism but
// large grids certainly do. Block size could be lowered at the expense of cache utilization.

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <omp.h>

typedef uint32_t u32;
typedef int32_t s32;
typedef double f64;

typedef f64 (*Unit_Square_Function)(f64 x, f64 y);

struct Net
{
    u32 size;
    f64 **values;
    f64 **laplacian;
};

// Cache line size on x64 is typically 64 bytes, we use 8-byte f64 values
const u32 BLOCK_SIZE = 8;
const f64 DESIRED_ERROR = 1e-1;

#define MIN(a, b) ((a) < (b) ? (a) : (b))

f64 blockwise(Net *net, s32 block_i, s32 block_j)
{
    s32 i_min = 1 + block_i * BLOCK_SIZE;
    s32 i_max = MIN(i_min + BLOCK_SIZE, net->size - 1);
    s32 j_min = 1 + block_j * BLOCK_SIZE;
    s32 j_max = MIN(j_min + BLOCK_SIZE, net->size - 1);

    const f64 step = 1.0 / (net->size - 1);

    f64 max_error = 0;
    for (s32 i = i_min; i < i_max; i++) {
        for (s32 j = j_min; j < j_max; j++) {
            f64 previous = net->values[i][j];
            net->values[i][j] =
                0.25 * (net->values[i - 1][j] + net->values[i + 1][j] + net->values[i][j - 1] + net->values[i][j + 1] - step * step * net->laplacian[i][j]);

            f64 current_error = fabs(previous - net->values[i][j]);
            if (max_error < current_error) {
                max_error = current_error;
            }
        }
    }

    return max_error;
}

u32 solve(Net *net)
{
    // Exclude the border and get one more block if it doesn't fit perfectly
    s32 num_blocks = (net->size - 2) / BLOCK_SIZE;
    if ((net->size - 2) % BLOCK_SIZE != 0) {
        num_blocks++;
    }

    f64 max_error = 0;
    f64 *errors = (f64 *) malloc(num_blocks * sizeof(f64));

    u32 total_iterations = 0;
    do {
        max_error = 0;
        for (s32 block_id = 0; block_id < num_blocks; block_id++) {
            errors[block_id] = 0;
        }

        for (s32 wave_size = 0; wave_size < num_blocks; wave_size++) {
            s32 block_i, block_j;
            f64 current_error;
#pragma omp parallel for shared(net, wave_size, errors) private(block_i, block_j, current_error)
            for (block_i = 0; block_i < wave_size + 1; block_i++) {
                block_j = wave_size - block_i;
                current_error = blockwise(net, block_i, block_j);
                errors[block_i] = fmax(errors[block_i], current_error);
            }
        }

        for (s32 wave_size = num_blocks - 2; wave_size > -1; wave_size--) {
            s32 block_i, block_j;
            f64 current_error;
#pragma omp parallel for shared(net, wave_size, errors) private(block_i, block_j, current_error)
            for (block_i = 0; block_i < wave_size + 1; block_i++) {
                block_j = 2 * (num_blocks - 1) - wave_size - block_i;
                current_error = blockwise(net, block_i, block_j);
                errors[block_i] = fmax(errors[block_i], current_error);
            }
        }

        for (s32 block_i = 0; block_i < num_blocks; block_i++) {
            max_error = fmax(max_error, errors[block_i]);
        }

        total_iterations++;
    } while (max_error > DESIRED_ERROR);

    free(errors);

    return total_iterations;
}

// \Delta u = f
// u = g on the boundary
void init(Net *net, u32 size, Unit_Square_Function laplacian, Unit_Square_Function boundary)
{
    net->size = size;
    net->laplacian = (f64 **) malloc(size * sizeof(f64 *));
    net->values = (f64 **) malloc(size * sizeof(f64 *));
    for (u32 i = 0; i < net->size; i++) {
        net->laplacian[i] = (f64 *) malloc(size * sizeof(f64));
        net->values[i]    = (f64 *) malloc(size * sizeof(f64));
    }

    const f64 step = 1.0 / (size - 1);
    for (u32 x = 0; x < size; x++) {
        for (u32 y = 0; y < size; y++) {
            net->laplacian[x][y] = laplacian(x * step, y * step);

            if ((x == 0) || (y == 0) || (x == size - 1) || (y == size - 1)) {
                net->values[x][y] = boundary(x * step, y * step);
            }
        }
    }
}

// Request harmonic function
f64 test_laplacian(f64 x, f64 y)
{
    return 0;
}

// Use the function from the book
f64 test_boundary(f64 x, f64 y)
{
    if (y == 0) {
        return 100 - 200 * x;
    }

    if (x == 0) {
        return 100 - 200 * y;
    }

    if (y == 1) {
        return -100 + 200 * x;
    }

    return -100 + 200 * y;
}

void test(Net *net, u32 num_threads)
{
    omp_set_num_threads(num_threads);

    f64 start, end, elapsed;
    start = omp_get_wtime();

    u32 iterations = solve(net);

    end = omp_get_wtime();
    elapsed = end - start;

    printf("Elapsed: %.5f seconds. Iterations: %d.\n", elapsed, iterations);
}

int main()
{
    Net net;
    init(&net, 2026, test_laplacian, test_boundary);

    test(&net, 8);

    return 0;
}

