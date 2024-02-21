__global__ void rgb_to_rgba_shift_segment(unsigned char* __restrict__ image_bytes, unsigned int start_index) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    const unsigned int rgba_base_addr = index * 4;
    const unsigned int rgb_base_addr = index * 3;

    image_bytes[rgba_base_addr] = image_bytes[rgb_base_addr];
    image_bytes[rgba_base_addr+1] = image_bytes[rgb_base_addr+1];
    image_bytes[rgba_base_addr+2] = image_bytes[rgb_base_addr+2];
    image_bytes[rgba_base_addr+3] = 0;
}
__global__ void rgb_to_rgba_shift_segment_final(unsigned char* __restrict__ image_bytes) {
    const unsigned int index = threadIdx.x;
    const unsigned int rgb_base_addr = index * 3;

    char temp_byte_buffer[3];

    temp_byte_buffer[0] = image_bytes[rgb_base_addr];
    temp_byte_buffer[1] = image_bytes[rgb_base_addr+1];
    temp_byte_buffer[2] = image_bytes[rgb_base_addr+2];

    __syncthreads();

    const unsigned int rgba_base_addr = index * 4;
    image_bytes[rgba_base_addr] = temp_byte_buffer[0];
    image_bytes[rgba_base_addr+1] = temp_byte_buffer[1];
    image_bytes[rgba_base_addr+2] = temp_byte_buffer[2];
    image_bytes[rgba_base_addr+3] = 0;
}
extern "C" __global__ void rgb_to_rgba_single_pass(unsigned long* __restrict__ image, unsigned int pitch, unsigned int size) {
    
}
extern "C" __global__ void rgb_to_rgba(unsigned long* __restrict__ image, unsigned int pitch, unsigned int size) {
    unsigned char* __restrict__ image_bytes = (unsigned char*) image;

    unsigned int free_segment_size = size / 4;
    unsigned int position = size - free_segment_size;

    for (unsigned int i = 9; i > 0; i--) {
        const unsigned int block_size = 2 << i;
        while (free_segment_size > 0 && free_segment_size % block_size == 0) {
            rgb_to_rgba_shift_segment<<<free_segment_size / block_size, block_size>>>(image_bytes, position);
            free_segment_size /= 4;
            free_segment_size *= 3;
            position -= free_segment_size;
            i = 10;
        }
    }

    for (unsigned int i = 1024; i > 0; i--) {
        while (free_segment_size > 0 && free_segment_size % i == 0) {
            rgb_to_rgba_shift_segment<<<free_segment_size / i, i>>>(image_bytes, position);
            free_segment_size /= 4;
            free_segment_size *= 3;
            position -= free_segment_size;
            //i = 1025;
        }
    }
    rgb_to_rgba_shift_segment_final<<<1, position>>>(image_bytes);
}