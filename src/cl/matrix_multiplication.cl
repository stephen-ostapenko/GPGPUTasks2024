#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
    __global float *a, __global float *b, __global float *c,
    unsigned int m, unsigned int k, unsigned int n
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= n || j >= m) {
        return;
    }

    c[j * n + i] = 0;

    for (unsigned int t = 0; t < k; t++) {
        c[j * n + i] += a[j * k + t] * b[t * n + i];
    }
}

#ifdef TILE_SIZE

__kernel void matrix_multiplication_local(
    __global float *a, __global float *b, __global float *c,
    unsigned int m, unsigned int k, unsigned int n
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0;
    for (unsigned int t = 0; t * TILE_SIZE < k; t++) {
        unsigned int tile_i = t * TILE_SIZE + local_i;
        unsigned int tile_j = t * TILE_SIZE + local_j;

        tile_a[local_j][local_i] = a[j * k + tile_i];
        tile_b[local_j][local_i] = b[tile_j * n + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            sum += tile_a[local_j][k] * tile_b[k][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < m && j < n) {
        c[n * j + i] = sum;
    }
}

#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)

#define TILE_PARTS TILE_SIZE / WORK_PER_THREAD

__kernel void matrix_multiplication_local_wpt(
    __global float *a, __global float *b, __global float *c,
    unsigned int m, unsigned int k, unsigned int n
) {
    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    unsigned int i = TILE_SIZE * get_group_id(0) + local_i;
    unsigned int j = TILE_SIZE * get_group_id(1) + local_j;

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float res[WORK_PER_THREAD];
    for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
        res[w] = 0.f;
    }

    for (unsigned int t = 0; t * TILE_SIZE < k; t++) {
        for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
            tile_a[local_j + w * TILE_PARTS][local_i] = a[(local_i + t * TILE_SIZE) + (j + w * TILE_PARTS) * k];
            tile_b[local_j + w * TILE_PARTS][local_i] = b[i + (local_j + t * TILE_SIZE + w * TILE_PARTS) * n];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            float tile_b_value = tile_b[k][local_i];

            for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
                res[w] += tile_a[local_j + w * TILE_PARTS][k] * tile_b_value;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned int w = 0; w < WORK_PER_THREAD; w++) {
        c[(j + w * TILE_PARTS) * n + i] = res[w];
    }
}

#endif
