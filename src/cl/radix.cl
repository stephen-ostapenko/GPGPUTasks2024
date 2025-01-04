__kernel void set_zeros(__global unsigned int *a, unsigned int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        a[idx] = 0;
    }
}

#define TRANSPOSE_TILE_SIZE 16

__kernel void matrix_transpose_local_memory_kernel(
    __global const unsigned int *a,
    __global unsigned int *a_t,
    unsigned int m, unsigned int k
) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (j >= m || i >= k) {
        barrier(CLK_LOCAL_MEM_FENCE);
        return;
    }

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile[TRANSPOSE_TILE_SIZE][TRANSPOSE_TILE_SIZE + 1];

    tile[local_j][local_i] = a[j * k + i];

    unsigned int res_i = get_group_id(0) * TRANSPOSE_TILE_SIZE + local_j;
    unsigned int res_j = get_group_id(1) * TRANSPOSE_TILE_SIZE + local_i;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (res_i < k && res_j < m) {
        a_t[res_i * m + res_j] = tile[local_i][local_j];
    }
}

__kernel void prefix_sum_kernel(
    __global const unsigned int *a, __global unsigned int *b,
    unsigned int n, unsigned int start, int step, unsigned int offset
) {
    int pos = start + get_global_id(0) * step;
    if (pos < 0 || pos >= n) {
        return;
    }
    b[pos] = a[pos];
    if (pos >= offset) {
        b[pos] += a[pos - offset];
    }
}

unsigned int get_bits(unsigned int x, unsigned int bits_num, unsigned int shift) {
    return (x >> shift) & ((1 << bits_num) - 1);
}

__kernel void calc_cnt_kernel(
    __global const unsigned int *a, __global unsigned int *cnt,
    unsigned int n, unsigned int bits_num, unsigned int shift
) {
    unsigned int gid = get_global_id(0);
    unsigned int wid = get_group_id(0);
    const unsigned int total_counters = 1 << bits_num;

    if (gid < n) {
        atomic_inc(&cnt[wid * total_counters + get_bits(a[gid], bits_num, shift)]);
    }
}

__kernel void radix_sort_kernel(
    __global const unsigned int *a, __global unsigned int *b,
    __global const unsigned int *cnt_t,
    unsigned int n, unsigned int bits_num, unsigned int shift, unsigned int workgroups_num
) {
    unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }

    unsigned int lid = get_local_id(0);
    unsigned int wid = get_group_id(0);
    unsigned int digit = get_bits(a[gid], bits_num, shift);

    unsigned int eq_cnt = 0;
    for (int i = 1; i <= lid; i++) {
        eq_cnt += (get_bits(a[gid - i], bits_num, shift) == digit);
    }

    unsigned int lt_cnt = 0;
    unsigned int cnt_pos = digit * workgroups_num + wid;
    if (cnt_pos > 0) {
        lt_cnt = cnt_t[cnt_pos - 1];
    }

    b[lt_cnt + eq_cnt] = a[gid];
}
