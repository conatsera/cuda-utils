#ifndef SIZE_PIXELS
#define SIZE_PIXELS 4096 * 3072
#endif

struct BlockMove
{
    unsigned int size = 0;
    unsigned int block_size = 0;
    unsigned int position = 0;
    __device__ constexpr BlockMove(
        unsigned int size = 0,
        unsigned int block_size = 0,
        unsigned int position = 0) : size(size), block_size(block_size), position(position) {}
};

__device__ constexpr unsigned int calculate_block_count()
{
    int free_segment_size = SIZE_PIXELS / 4;
    int i = 0;
    while (free_segment_size > 0)
    {
        i++;
        free_segment_size /= 4;
        free_segment_size *= 3;
    }
    return i;
}

constexpr const unsigned int kBlockCount = calculate_block_count();

__device__ constexpr unsigned int count_trailing_zeros(unsigned int num)
{
    for (int i = 31; i >= 0; i--)
    {
        if (num & (1 << i) == (1 << i))
        {
            return 31 - i;
        }
    }
    return 0;
}

__device__ constexpr unsigned int calculate_optimal_block_size(const unsigned int size)
{
    const unsigned int pow2_divisor = count_trailing_zeros(size);
    if (pow2_divisor < 2)
    {
        int i = 1023;
        while (i > 0)
        {
            if (size % i == 0)
            {
                return i;
            }
            i -= 1;
        }
    }
    else if (pow2_divisor < 10)
    {
        return 1 << pow2_divisor;
    }
    else
    {
        return 1024;
    }
}

struct BlockMoves
{
    __device__ constexpr BlockMoves() : moves()
    {
        int free_segment_size = SIZE_PIXELS / 4;
        int position = SIZE_PIXELS - free_segment_size;
        for (int i = 0; i < kBlockCount; i++)
        {
            moves[i] = BlockMove(free_segment_size, calculate_optimal_block_size(free_segment_size), position);
            free_segment_size /= 4;
            free_segment_size *= 3;
            position -= free_segment_size;
        }
    }
    BlockMove moves[kBlockCount];
};

__global__ void rgb_to_rgba_shift_segment(unsigned char *__restrict__ image_bytes, const __grid_constant__ unsigned int start_index)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x + start_index;
    const unsigned int rgb_base_addr = index * 3;
    const unsigned int rgba_base_addr = index * 4;

    const uchar4 rgba_pixel = make_uchar4(image_bytes[rgb_base_addr], image_bytes[rgb_base_addr + 1], image_bytes[rgb_base_addr + 2], 0);
    memcpy(&image_bytes[rgba_base_addr], &rgba_pixel, 4);
}

__global__ void rgb_to_rgba_shift_segment_final(unsigned char *__restrict__ image_bytes)
{
    const unsigned int index = threadIdx.x;
    const unsigned int rgb_base_addr = index * 3;

    const uchar4 rgba_pixel = make_uchar4(image_bytes[rgb_base_addr], image_bytes[rgb_base_addr + 1], image_bytes[rgb_base_addr + 2], 0);

    __syncthreads();

    const unsigned int rgba_base_addr = index * 4;
    memcpy(&image_bytes[rgba_base_addr], &rgba_pixel, 4);
}

extern "C" __global__ void rgb_to_rgba(unsigned char *__restrict__ image_bytes)
{
    constexpr const auto block_moves = BlockMoves();
#pragma unroll
    for (int i = 0; i < kBlockCount; i++)
    {
        const unsigned int block_size = block_moves.moves[i].block_size;
        rgb_to_rgba_shift_segment<<<block_moves.moves[i].size / block_size, block_size>>>(image_bytes, block_moves.moves[i].position);
    }

    rgb_to_rgba_shift_segment_final<<<1, block_moves.moves[kBlockCount-1].position>>>(image_bytes);
}
