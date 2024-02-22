__global__ void rgb_to_rgba_shift_segment(unsigned char* __restrict__ image_bytes, const __grid_constant__ unsigned int start_index) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    const unsigned int rgb_base_addr = index * 3;
    const unsigned int rgba_base_addr = index * 4;
    
    const uchar4 rgba_pixel = make_uchar4(image_bytes[rgb_base_addr], image_bytes[rgb_base_addr+1], image_bytes[rgb_base_addr+2], 0);
    memcpy(&image_bytes[rgba_base_addr], &rgba_pixel, 4);
}
__global__ void rgb_to_rgba_shift_segment_final(unsigned char* __restrict__ image_bytes) {
    const unsigned int index = threadIdx.x;
    const unsigned int rgb_base_addr = index * 3;

    const uchar4 rgba_pixel = make_uchar4(image_bytes[rgb_base_addr], image_bytes[rgb_base_addr+1], image_bytes[rgb_base_addr+2], 0);

    __syncthreads();

    const unsigned int rgba_base_addr = index * 4;
    memcpy(&image_bytes[rgba_base_addr], &rgba_pixel, 4);
}
struct BlockMove {
    unsigned int size = 0;
    unsigned int block_size = 0;
    unsigned int position = 0;
    __device__ BlockMove(
        const unsigned int size = 0,
        const unsigned int block_size = 0,
        const unsigned int position = 0
        ) : size(size), block_size(block_size), position(position) {}
};
extern "C" __global__ void rgb_to_rgba(
        unsigned char* __restrict__ image_bytes,
        const __grid_constant__ unsigned int block_move_count,
        const __grid_constant__ unsigned int final_position,
        const BlockMove *const __restrict__ block_moves
    ) {
    #pragma unroll 64
    for (int i = 0; i < block_move_count; i++) {
        const unsigned int block_size = block_moves[i].block_size;
        rgb_to_rgba_shift_segment<<<block_moves[i].size / block_size, block_size>>>(image_bytes, block_moves[i].position);
    }

    rgb_to_rgba_shift_segment_final<<<1, final_position>>>(image_bytes);
}
