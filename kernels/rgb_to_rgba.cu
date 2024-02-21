__global__ void rgb_to_rgba_shift_segment(unsigned char* __restrict__ image_bytes, const unsigned int start_index) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    const unsigned int rgba_base_addr = index * 4;
    const unsigned int rgb_base_addr = index * 3;
    
    memcpy(&image_bytes[rgba_base_addr], &image_bytes[rgb_base_addr], 4);
    image_bytes[rgba_base_addr+3] = 0;
}
__global__ void rgb_to_rgba_shift_segment_final(unsigned char* __restrict__ image_bytes) {
    const unsigned int index = threadIdx.x;
    const unsigned int rgb_base_addr = index * 3;

    char temp_byte_buffer[4];
    memcpy(&temp_byte_buffer, &image_bytes[rgb_base_addr], 4);
    temp_byte_buffer[3] = 0;

    __syncthreads();

    const unsigned int rgba_base_addr = index * 4;
    memcpy(&image_bytes[rgba_base_addr], &temp_byte_buffer, 4);
}
extern "C" __global__ void rgb_to_rgba(unsigned long* __restrict__ image, const unsigned int pitch, const unsigned int size) {
    unsigned char* __restrict__ image_bytes = (unsigned char*) image;

    unsigned int free_segment_size = size / 4;
    unsigned int position = size - free_segment_size;

    #pragma unroll
    for (unsigned int i = 1024; i > 1; i >>= 1) {
        //printf("%u %u\n", i, free_segment_size);
        if (free_segment_size % i == 0) {
            rgb_to_rgba_shift_segment<<<free_segment_size / i, i>>>(image_bytes, position);
            free_segment_size /= 4;
            free_segment_size *= 3;
            position -= free_segment_size;
            i = 2048;
        }
    }
    
    #pragma unroll
    for (unsigned int i = 1024; i > 0; i--) {
       // printf("%u %u\n", i, free_segment_size);
        #pragma unroll
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