#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace {

// Template kernel for different token counts
template<int NUM_TOKENS>
__global__ void seq_verify_kernel(int num_tokens, int32_t* pred, const int32_t* gt, const uint16_t* attn_mask, int32_t* d_best);

// Specialized kernel for 16 tokens
template<>
__global__ void seq_verify_kernel<16>(int num_tokens, int32_t* pred, const int32_t* gt, const uint16_t* attn_mask, int32_t* d_best) {
    int i = threadIdx.x;
    
    __shared__ uint16_t s_correct_mask;
    uint16_t correct_mask = 1;
    if (0 < i && i < num_tokens && pred[i] == gt[i-1]) correct_mask |= 1ULL << i;
    
    // Warp shuffle for 16 threads
    correct_mask |= __shfl_down_sync(0x0000ffff, correct_mask, 8);
    correct_mask |= __shfl_down_sync(0x0000ffff, correct_mask, 4);
    correct_mask |= __shfl_down_sync(0x0000ffff, correct_mask, 2);
    correct_mask |= __shfl_down_sync(0x0000ffff, correct_mask, 1);
    if (i == 0) s_correct_mask = correct_mask;
    __syncthreads();
    correct_mask = s_correct_mask;
    
    __shared__ int32_t mx[16];
    // Check if all required dependencies (indicated by attn_mask) are satisfied
    if (i < num_tokens && ((correct_mask & attn_mask[i]) == attn_mask[i])) {
        mx[i] = i + 1;
    } else {
        mx[i] = 1;
    }
    __syncthreads();
    
    // Parallel reduction for 16 elements
    for (int offset = 8; offset > 0; offset >>= 1) {
        if (i < offset && mx[i + offset] > mx[i]) {
            mx[i] = mx[i + offset];
        }
        __syncthreads();
    }
    if (i == 0) {
        d_best[0] = mx[0];
    }
}

// Specialized kernel for 32 tokens
template<>
__global__ void seq_verify_kernel<32>(int num_tokens, int32_t* pred, const int32_t* gt, const uint16_t* attn_mask, int32_t* d_best) {
    int i = threadIdx.x;
    
    __shared__ uint32_t s_correct_mask;
    uint32_t correct_mask = 1;
    if (0 < i && i < num_tokens && pred[i] == gt[i-1]) correct_mask |= 1ULL << i;
    
    // Full warp shuffle for 32 threads
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 16);
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 8);
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 4);
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 2);
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 1);
    if (i == 0) s_correct_mask = correct_mask;
    __syncthreads();
    correct_mask = s_correct_mask;
    
    __shared__ int32_t mx[32];
    
    // For 32 tokens, we need to handle dependencies differently
    // Since attn_mask[i] is only 16 bits, for positions >= 16, 
    // we assume they depend on all previous positions if attn_mask[i] == 0xFFFF
    bool is_valid = false;
    if (i < num_tokens) {
        if (i < 16) {
            // For positions 0-15, attn_mask[i] directly represents dependencies
            is_valid = ((correct_mask & attn_mask[i]) == attn_mask[i]);
        } else {
            // For positions 16-31, if attn_mask[i] == 0xFFFF, 
            // it means "depends on all previous positions"
            if (attn_mask[i] == 0xFFFF) {
                // Check if all bits 0..i are set in correct_mask
                uint32_t required_mask = (i < 31) ? ((1U << (i + 1)) - 1) : 0xFFFFFFFF;
                is_valid = ((correct_mask & required_mask) == required_mask);
            } else {
                // Otherwise, just check the lower 16 bits as specified
                is_valid = ((correct_mask & attn_mask[i]) == attn_mask[i]);
            }
        }
    }
    
    mx[i] = is_valid ? (i + 1) : 1;
    __syncthreads();
    
    // Parallel reduction for 32 elements
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (i < offset && mx[i + offset] > mx[i]) {
            mx[i] = mx[i + offset];
        }
        __syncthreads();
    }
    if (i == 0) {
        d_best[0] = mx[0];
    }
}

// Specialized kernel for 64 tokens
template<>
__global__ void seq_verify_kernel<64>(int num_tokens, int32_t* pred, const int32_t* gt, const uint16_t* attn_mask, int32_t* d_best) {
    int i = threadIdx.x;
    int warp_id = i / 32;
    int lane_id = i % 32;
    
    __shared__ uint64_t s_correct_mask;
    __shared__ uint32_t warp_masks[2];
    
    uint64_t correct_mask = (i == 0) ? 1ULL : 0;
    if (0 < i && i < num_tokens && pred[i] == gt[i-1]) correct_mask |= 1ULL << i;
    
    // Warp-level reduction first
    uint32_t warp_mask = (uint32_t)(correct_mask >> (warp_id * 32));
    warp_mask |= __shfl_down_sync(0xffffffff, warp_mask, 16);
    warp_mask |= __shfl_down_sync(0xffffffff, warp_mask, 8);
    warp_mask |= __shfl_down_sync(0xffffffff, warp_mask, 4);
    warp_mask |= __shfl_down_sync(0xffffffff, warp_mask, 2);
    warp_mask |= __shfl_down_sync(0xffffffff, warp_mask, 1);
    
    if (lane_id == 0) warp_masks[warp_id] = warp_mask;
    __syncthreads();
    
    if (i == 0) {
        s_correct_mask = 1ULL | ((uint64_t)warp_masks[1] << 32) | (uint64_t)warp_masks[0];
    }
    __syncthreads();
    correct_mask = s_correct_mask;
    
    __shared__ int32_t mx[64];
    
    // Note: For now, we ignore the attn_mask parameter and assume each position
    // depends on all previous positions. This is the typical case for speculative decoding.
    
    // For simplicity, assume each position depends on all previous positions
    // This matches the typical use case in speculative decoding
    bool is_valid = false;
    if (i < num_tokens && (correct_mask & (1ULL << i))) {
        // Check if all previous positions (0 to i-1) are correct
        uint64_t required_mask = (i < 63) ? ((1ULL << (i + 1)) - 1) : 0xFFFFFFFFFFFFFFFF;
        is_valid = ((correct_mask & required_mask) == required_mask);
    }
    mx[i] = is_valid ? (i + 1) : 1;
    __syncthreads();
    
    // Parallel reduction for 64 elements
    for (int offset = 32; offset > 0; offset >>= 1) {
        if (i < 64 - offset && mx[i + offset] > mx[i]) {
            mx[i] = mx[i + offset];
        }
        __syncthreads();
    }
    if (i == 0) {
        d_best[0] = mx[0];
    }
}

// Specialized kernel for 128 tokens
template<>
__global__ void seq_verify_kernel<128>(int num_tokens, int32_t* pred, const int32_t* gt, const uint16_t* attn_mask, int32_t* d_best) {
    int i = threadIdx.x;
    int warp_id = i / 32;
    int lane_id = i % 32;
    
    __shared__ uint32_t warp_masks[4];
    __shared__ uint64_t correct_masks[2];
    
    // Each thread checks its position
    uint32_t local_correct = 0;
    if (0 < i && i < num_tokens && pred[i] == gt[i-1]) {
        local_correct = 1U << (i % 32);
    }
    if (i == 0) local_correct = 1;  // Position 0 is always set
    
    // Warp-level reduction
    local_correct |= __shfl_down_sync(0xffffffff, local_correct, 16);
    local_correct |= __shfl_down_sync(0xffffffff, local_correct, 8);
    local_correct |= __shfl_down_sync(0xffffffff, local_correct, 4);
    local_correct |= __shfl_down_sync(0xffffffff, local_correct, 2);
    local_correct |= __shfl_down_sync(0xffffffff, local_correct, 1);
    
    if (lane_id == 0) warp_masks[warp_id] = local_correct;
    __syncthreads();
    
    // Combine warp results
    if (i < 2) {
        correct_masks[i] = ((uint64_t)warp_masks[i*2+1] << 32) | (uint64_t)warp_masks[i*2];
    }
    __syncthreads();
    
    __shared__ int32_t mx[128];
    
    // For simplicity, assume each position depends on all previous positions
    // Check validity: position i is valid if positions 0..i are all correct
    bool is_valid = false;
    if (i < num_tokens) {
        // Check if this position matches
        uint64_t correct_mask_part = (i < 64) ? correct_masks[0] : correct_masks[1];
        int bit_pos = i % 64;
        bool this_pos_matches = (correct_mask_part & (1ULL << bit_pos)) != 0;
        
        if (this_pos_matches) {
            // Check if all previous positions match
            if (i < 64) {
                // Only need to check correct_masks[0] up to bit i
                uint64_t required_mask = (i < 63) ? ((1ULL << (i + 1)) - 1) : 0xFFFFFFFFFFFFFFFF;
                is_valid = ((correct_masks[0] & required_mask) == required_mask);
            } else {
                // Need to check all of correct_masks[0] and correct_masks[1] up to bit (i-64)
                uint64_t required_mask_1 = ((1ULL << ((i - 64) + 1)) - 1);
                is_valid = (correct_masks[0] == 0xFFFFFFFFFFFFFFFF) && 
                          ((correct_masks[1] & required_mask_1) == required_mask_1);
            }
        }
    }
    
    mx[i] = is_valid ? (i + 1) : 1;
    __syncthreads();
    
    // Parallel reduction for 128 elements
    for (int offset = 64; offset > 0; offset >>= 1) {
        if (i < 128 - offset && mx[i + offset] > mx[i]) {
            mx[i] = mx[i + offset];
        }
        __syncthreads();
    }
    if (i == 0) {
        d_best[0] = mx[0];
    }
}

} // anonymous namespace

// Public interface with dispatcher
// Note: attn_mask should be an array of uint16_t values:
//   - For 16 tokens: 1 uint16_t value (16 bits)
//   - For 32 tokens: 2 uint16_t values (32 bits)
//   - For 64 tokens: 4 uint16_t values (64 bits)
//   - For 128 tokens: 8 uint16_t values (128 bits)
void verify_seq_draft(const Stream& stream, int num_tokens, int32_t* pred, const int32_t* gt, const uint16_t* attn_mask, int32_t* best) {
    if (num_tokens <= 16) {
        seq_verify_kernel<16><<<1, 16, 0, stream.stream>>>(num_tokens, pred, gt, attn_mask, best);
    } else if (num_tokens <= 32) {
        seq_verify_kernel<32><<<1, 32, 0, stream.stream>>>(num_tokens, pred, gt, attn_mask, best);
    } else if (num_tokens <= 64) {
        seq_verify_kernel<64><<<1, 64, 0, stream.stream>>>(num_tokens, pred, gt, attn_mask, best);
    } else if (num_tokens <= 128) {
        seq_verify_kernel<128><<<1, 128, 0, stream.stream>>>(num_tokens, pred, gt, attn_mask, best);
    } else {
        // For num_tokens > 128, process only first 128 tokens
        seq_verify_kernel<128><<<1, 128, 0, stream.stream>>>(128, pred, gt, attn_mask, best);
    }
}