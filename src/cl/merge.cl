#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

int lower_bound(__global const int *begin, __global const int *end, int x) {
    __global const int *l = begin, *r = end;
    while (l + 1 < r) {
        __global const int *m = l + (r - l) / 2;
        if (*m < x) {
            l = m;
        } else {
            r = m;
        }
    }
    if (*l >= x) {
        return 0;
    }
    return r - begin;
}

int upper_bound(__global const int *begin, __global const int *end, int x) {
    __global const int *l = begin, *r = end;
    while (l + 1 < r) {
        __global const int *m = l + (r - l) / 2;
        if (*m <= x) {
            l = m;
        } else {
            r = m;
        }
    }
    if (*l > x) {
        return 0;
    }
    return r - begin;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size) {
    unsigned int gid = get_global_id(0);
    unsigned int block_idx = gid / (block_size * 2);
    unsigned int idx = gid % (block_size * 2);

    __global const int *begin = as + 2 * block_size * block_idx;
    __global const int *mid = begin + block_size;
    __global const int *end = begin + 2 * block_size;
    __global int *out = bs + 2 * block_size * block_idx;

    if (idx < block_size) {
        int x = *(begin + idx);
        unsigned int pos = lower_bound(mid, end, x) + idx;
        *(out + pos) = x;
    } else {
        idx -= block_size;
        int x = *(mid + idx);
        unsigned int pos = upper_bound(begin, mid, x) + idx;
        *(out + pos) = x;
    }
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
