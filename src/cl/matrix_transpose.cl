#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(
    __global const float *as,
    __global float *as_t,
    unsigned int m, unsigned int k
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= k || j >= m) {
        return;
    }

    as_t[i * m + j] = as[j * k + i];
}

#define TILE_SIZE 16

__kernel void matrix_transpose_local_bad_banks(
    __global const float *as,
    __global float *as_t,
    unsigned int m, unsigned int k
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

    tile[local_j][local_i] = as[j * k + i];

    unsigned int res_i = get_group_id(0) * TILE_SIZE + local_j;
    unsigned int res_j = get_group_id(1) * TILE_SIZE + local_i;

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[res_i * m + res_j] = tile[local_i][local_j];
}

__kernel void matrix_transpose_local_good_banks(
    __global const float *as,
    __global float *as_t,
    unsigned int m, unsigned int k
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    tile[local_j][local_i] = as[j * k + i];

    unsigned int res_i = get_group_id(0) * TILE_SIZE + local_j;
    unsigned int res_j = get_group_id(1) * TILE_SIZE + local_i;

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[res_i * m + res_j] = tile[local_i][local_j];
}
