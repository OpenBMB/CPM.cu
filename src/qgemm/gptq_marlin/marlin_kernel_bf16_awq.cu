#include "marlin_kernel_impl.cuh"

#ifdef ENABLE_DTYPE_BF16

// Helper macros for template instantiation
#define MARLIN_INST_SPLIT(dtype, qtype, block_size, split, warp_m, warp_n, pack_size) \
    template __global__ void marlin::Marlin<dtype, qtype, block_size, split, warp_m, warp_n, pack_size, false, true, -1> \
        (const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool); \
    template __global__ void marlin::Marlin<dtype, qtype, block_size, split, warp_m, warp_n, pack_size, false, true, 2> \
        (const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool); \
    template __global__ void marlin::Marlin<dtype, qtype, block_size, split, warp_m, warp_n, pack_size, false, true, 4> \
        (const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool); \
    template __global__ void marlin::Marlin<dtype, qtype, block_size, split, warp_m, warp_n, pack_size, false, true, 8> \
        (const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);

#define MARLIN_INST_ALL_SPLITS(dtype, qtype, block_size, warp_m, warp_n, pack_size) \
    MARLIN_INST_SPLIT(dtype, qtype, block_size, 1, warp_m, warp_n, pack_size) \
    MARLIN_INST_SPLIT(dtype, qtype, block_size, 2, warp_m, warp_n, pack_size) \
    MARLIN_INST_SPLIT(dtype, qtype, block_size, 3, warp_m, warp_n, pack_size) \
    MARLIN_INST_SPLIT(dtype, qtype, block_size, 4, warp_m, warp_n, pack_size)

#define MARLIN_INST_QTYPE(dtype, block_size, warp_m, warp_n, pack_size) \
    MARLIN_INST_ALL_SPLITS(dtype, vllm::kU4.id(), block_size, warp_m, warp_n, pack_size) \
    MARLIN_INST_ALL_SPLITS(dtype, vllm::kU8.id(), block_size, warp_m, warp_n, pack_size)

#define MARLIN_INST_CONFIG(block_size, warp_m, warp_n, pack_size) \
    MARLIN_INST_QTYPE(__nv_bfloat16, block_size, warp_m, warp_n, pack_size)

// Instantiate all combinations
MARLIN_INST_CONFIG(256, 16, 4, 4)  // AWQ_CALL_IF(vllm::kU4/kU8, 16, 4, 256)
MARLIN_INST_CONFIG(256, 8, 8, 4)   // AWQ_CALL_IF(vllm::kU4/kU8, 8, 8, 256)
MARLIN_INST_CONFIG(128, 8, 4, 4)   // AWQ_CALL_IF(vllm::kU4/kU8, 8, 4, 128)
MARLIN_INST_CONFIG(128, 4, 8, 4)   // AWQ_CALL_IF(vllm::kU4/kU8, 4, 8, 128)

#undef MARLIN_INST_SPLIT
#undef MARLIN_INST_ALL_SPLITS
#undef MARLIN_INST_QTYPE
#undef MARLIN_INST_CONFIG

#endif 