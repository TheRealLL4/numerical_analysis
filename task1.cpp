// clang -fopenmp task1.cpp -otask1.exe

// All tests below use BLOCK_SIZE=8, DESIRED_ERROR=1e-1, only laplacian and boundary functions vary.
// Tests were conducted on a i7-7700K @4.2GHz machine, that has 4 cores and 8 hyperthreads, using clang version 17.0.6
// These values are an arithmetic mean of the results over 10 runs:

// laplacian=book_laplacian, boundary=book_boundary:
// Size  Iterations Seconds (1 thread / 8 threads)  Speedup
// 100   2285                    0.21 / 0.18         1.16
// 200   4264                    1.67 / 1.3          1.28
// 1000  8500                    90.3 / 24.6         3.67
// 2000  9578                   978.8 / 230.87       4.24
// (Result are consistent with those in the book.)

// laplacian=laplacian2, boundary=boundary2:
// Size  Iterations Seconds (1 thread / 8 threads)  Speedup
// 100   2764                   0.41  / 0.21          1.95
// 200   4879                   0.83  / 0.4           2.08
// 1000  9298                   130.4 / 43.7          2.98
// 2000  11431                  1074  / 250.8         4.28
// (Solution converges more slowly because of the high oscillation.)

// laplacian=laplacian3, boundary=boundary3:
// Size  Iterations Seconds (1 thread / 8 threads)  Speedup
// 100   4                       0.01 / 0.06         0.16
// 200   7                       0.02 / 0.07         0.28
// 1000  10                      0.04 / 0.07         0.57
// 2000  12                      0.07 / 0.08         0.88
// (Solution converges almost instantly thus the overhead of launching multiple threads is no longes negligible.)

// laplacian=laplacian4, boundary=boundary4:
// Size  Iterations Seconds (1 thread / 8 threads)  Speedup
// 100   1898                    0.28 / 0.21         1.31
// 200   3491                    0.94 / 0.53         1.78
// 1000  4675                    6.64 / 2.02         3.29
// 2000  8924                    44.2 / 10.67        4.15
// (Covengence is much faster in this case which indicates that large but bounded functions do not affect the algorithm much.)

// laplacian=laplacian5, boundary=boundary5:
// Size  Iterations Seconds (1 thread / 8 threads)  Speedup
// 100   5500                     5.7 / 3.8          1.5
// 200   8935                    11.4 / 5.2          2.19
// 1000  11782                 1089.7 / 328.7        3.31
// 2000  15839                 1489.3 / 368.9        4.03
// (In this case we see that the speed of convegence is the smallest of all examples because the function is discontinuous at the origin.)

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <omp.h>

typedef uint32_t u32;
typedef int32_t  s32;
typedef double   f64;

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

// \Delta u = laplacian on the inside
// u = boundary on the boundary
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

////////////////////// Tests //////////////////////

// Request harmonic function
f64 book_laplacian(f64 x, f64 y)
{
    return 0;
}

// Use the function from the book
f64 book_boundary(f64 x, f64 y)
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

// Heavily oscillating function
f64 boundary2(f64 x, f64 y)
{
    return sin(100 * (x * x + y * y));
}

f64 laplacian2(f64 x, f64 y)
{
    return 400 * cos(100 * (x * x + y * y)) - 40000 * (x * x + y * y) * sin(100 * (x * x + y * y));
}

// Very simple, constant function u(x, y) = 3
f64 laplacian3(f64 x, f64 y)
{
    return 0;
}

f64 boundary3(f64 x, f64 y)
{
    return 3;
}

// Rapidly growing function near the origin u(x, y) = 1.0 / (x^2 + y^2 + 1E-9)
#define CUBE(x) ((x) * (x) * (x))
f64 laplacian4(f64 x, f64 y)
{
    return 4.0 * (-1E-9 + x * x + y * y) / CUBE(1E-9 + x * x + y * y);
}

f64 boundary4(f64 x, f64 y)
{
    return 1.0 / (x * x + y * y + 1E-9);
}

// Should converge to log(x^2 + y^2) that is discontinuous at (0, 0)
f64 laplacian5(f64 x, f64 y)
{
    return 0;
}

f64 boundary5(f64 x, f64 y)
{
    return log(x * x + y * y);
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
    return 0;
}

