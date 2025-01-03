__kernel void bitonic(__global int *a, unsigned int log_chunk_size, unsigned int log_block_size) {
    unsigned int block_size = 1 << log_block_size;

    unsigned int idx = get_global_id(0);
    unsigned int chunk_id = idx >> log_chunk_size;
    unsigned int block_id = idx >> log_block_size;

    unsigned int block_idx = idx & (block_size - 1);
    bool flip = chunk_id & 1;

    unsigned int i = (block_id << (log_block_size + 1)) + block_idx;
    unsigned int j = i + block_size;

    if ((a[i] > a[j]) ^ flip) {
        int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
}
