__kernel void prefix_sum_kernel(__global int *a, __global int *b, unsigned int n, unsigned int start, int step, unsigned int offset) {
    int pos = start + get_global_id(0) * step;
    if (pos < 0 || pos >= n) {
        return;
    }
    b[pos] = a[pos];
    if (pos >= offset) {
        b[pos] += a[pos - offset];
    }
}