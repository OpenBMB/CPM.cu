#pragma once
#include <cuda_runtime.h>
#include "../trait.cuh"
#include "../utils.cuh"

namespace {
template <typename T, typename T2>
__global__ void rms_norm_kernel(int dim, const T2* input, const T2* weight, T2* output, float eps) {
    // __shared__ T2 s_input[2048];
    __shared__ float shared_sum;
    __shared__ float warp_sum[16];
    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 val = input[row * dim + i];
        // s_input[i] = val;
        float val1 = float(val.x);
        float val2 = float(val.y);
        sum1 += val1 * val1;
        sum2 += val2 * val2;
    }
    float sum = sum1 + sum2;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (col % 32 == 0) warp_sum[col / 32] = sum;
    __syncthreads();
    if (col < 16) {
        sum = warp_sum[col];
        sum += __shfl_down_sync(0x0000ffff, sum, 8);
        sum += __shfl_down_sync(0x0000ffff, sum, 4);
        sum += __shfl_down_sync(0x0000ffff, sum, 2);
        sum += __shfl_down_sync(0x0000ffff, sum, 1);
    }
    if (col == 0) {
        shared_sum = rsqrtf(sum / (2*dim) + eps);
    }
    __syncthreads();
    sum = shared_sum;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 inp = input[row * dim + i];
        T2 w = weight[i];
        output[row * dim + i] = T2(
            T(sum * float(inp.x) * float(w.x)),
            T(sum * float(inp.y) * float(w.y))
        );
    }
}

template <typename T, typename T2>
__global__ void add_and_rms_norm_kernel(int dim, T2* input, const T2* prev_output, const T2* weight, T2* output, float eps) {
    // __shared__ T2 s_input[2048];
    __shared__ float shared_sum;
    __shared__ float warp_sum[16];
    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 val = input[row * dim + i];
        T2 prev = prev_output[row * dim + i];
        val = val + prev;
        input[row * dim + i] = val;
        float val1 = float(val.x);
        float val2 = float(val.y);
        sum1 += val1 * val1;
        sum2 += val2 * val2;
    }
    float sum = sum1 + sum2;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (col % 32 == 0) warp_sum[col / 32] = sum;
    __syncthreads();
    if (col < 16) {
        sum = warp_sum[col];
        sum += __shfl_down_sync(0x0000ffff, sum, 8);
        sum += __shfl_down_sync(0x0000ffff, sum, 4);
        sum += __shfl_down_sync(0x0000ffff, sum, 2);
        sum += __shfl_down_sync(0x0000ffff, sum, 1);
    }
    if (col == 0) {
        shared_sum = rsqrtf(sum / (2*dim) + eps);
    }
    __syncthreads();
    sum = shared_sum;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 inp = input[row * dim + i];
        T2 w = weight[i];
        output[row * dim + i] = T2(
            T(sum * float(inp.x) * float(w.x)),
            T(sum * float(inp.y) * float(w.y))
        );
    }
}

template <typename T>
void rms_norm(const Stream& stream, int num_tokens, int dim, const T* input, const T* weight, T* output, float eps) {
    using T2 = typename TypeTraits<T>::half2;
    rms_norm_kernel<T, T2><<<num_tokens, 512, 0, stream.stream>>>(dim/2, (T2*)input, (T2*)weight, (T2*)output, eps);
}

template <typename T>
void add_and_rms_norm(const Stream& stream, int num_tokens, int dim, T* input, const T* prev_output, const T* weight, T* output, float eps) {
    using T2 = typename TypeTraits<T>::half2;
    add_and_rms_norm_kernel<T, T2><<<num_tokens, 512, 0, stream.stream>>>(dim/2, (T2*)input, (T2*)prev_output, (T2*)weight, (T2*)output, eps);
}

template <typename T, typename T2>
__global__ void rms_norm2_kernel(int dim, const T2* input_hidden, const T2* hidden_weight, const T2* input_embed, const T2* embed_weight, T2* output, float eps) {
    __shared__ float warp_sum[32];

    float R_hidden[4][2];
    float R_embed[4][2];

    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum_hidden = 0.0f;
    float sum_embed = 0.0f;

    int iIter = 0;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 val_hidden = input_hidden[row * dim + i];
        float val1_hidden = float(val_hidden.x);
        float val2_hidden = float(val_hidden.y);
        
        R_hidden[iIter][0] = val1_hidden;
        R_hidden[iIter][1] = val2_hidden;

        sum_hidden = __fmaf_rn(val1_hidden, val1_hidden, sum_hidden);
        sum_hidden = __fmaf_rn(val2_hidden, val2_hidden, sum_hidden);

        T2 val_embed = input_embed[row * dim + i];
        float val1_embed = float(val_embed.x);
        float val2_embed = float(val_embed.y);

        R_embed[iIter][0] = val1_embed;
        R_embed[iIter][1] = val2_embed;

        sum_embed = __fmaf_rn(val1_embed, val1_embed, sum_embed);
        sum_embed = __fmaf_rn(val2_embed, val2_embed, sum_embed);

        ++iIter;
    }

    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 16);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 8);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 4);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 2);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 1);

    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 16);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 8);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 4);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 2);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 1);

    if ((col & 31) == 0) { 
        int offset = col >> 5;
        warp_sum[offset] = sum_hidden;
        warp_sum[offset + 16] = sum_embed; 
    }

    __syncthreads();

    if (col < 32) {
        float warp_sum_ = warp_sum[col];
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 8, 16);
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 4, 16);
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 2, 16);
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 1, 16);

        float factor = __frcp_rn(dim << 1);
        if ((col == 0) || (col == 16)) {
            warp_sum[col] = __frsqrt_rn(__fmaf_rn(warp_sum_, factor, eps));
        }
    }
    
    __syncthreads();
    
    sum_hidden = warp_sum[0];
    sum_embed = warp_sum[16];
    
    iIter = 0;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 w_hidden = hidden_weight[i];
        T2 w_embed = embed_weight[i];

        int offset = row * dim * 2 + i;
        output[offset + dim] = T2(
            T(sum_hidden * R_hidden[iIter][0] * float(w_hidden.x)),
            T(sum_hidden * R_hidden[iIter][1] * float(w_hidden.y))
        );
        output[offset] = T2(
            T(sum_embed * R_embed[iIter][0] * float(w_embed.x)),
            T(sum_embed * R_embed[iIter][1] * float(w_embed.y))
        );
        ++iIter;
    }
}

template <typename T, typename T2>
__global__ void add_and_rms_norm2_kernel(int dim, const T2* input_hidden, const T2* prev_hidden, const T2* hidden_weight, const T2* input_embed, const T2* embed_weight, T2* output, float eps) {
    __shared__ float warp_sum[32];

    float R_hidden[4][2];
    float R_embed[4][2];

    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum_hidden = 0.0f;
    float sum_embed = 0.0f;

    int iIter = 0;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 val_hidden = input_hidden[row * dim + i];
        T2 prev = prev_hidden[row * dim + i];
        val_hidden = val_hidden + prev;
        
        float val1_hidden = float(val_hidden.x);
        float val2_hidden = float(val_hidden.y);
        
        R_hidden[iIter][0] = val1_hidden;
        R_hidden[iIter][1] = val2_hidden;

        sum_hidden = __fmaf_rn(val1_hidden, val1_hidden, sum_hidden);
        sum_hidden = __fmaf_rn(val2_hidden, val2_hidden, sum_hidden);

        T2 val_embed = input_embed[row * dim + i];
        float val1_embed = float(val_embed.x);
        float val2_embed = float(val_embed.y);

        R_embed[iIter][0] = val1_embed;
        R_embed[iIter][1] = val2_embed;

        sum_embed = __fmaf_rn(val1_embed, val1_embed, sum_embed);
        sum_embed = __fmaf_rn(val2_embed, val2_embed, sum_embed);

        ++iIter;
    }

    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 16);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 8);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 4);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 2);
    sum_hidden += __shfl_down_sync(0xffffffff, sum_hidden, 1);

    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 16);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 8);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 4);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 2);
    sum_embed += __shfl_down_sync(0xffffffff, sum_embed, 1);

    if ((col & 31) == 0) { 
        int offset = col >> 5;
        warp_sum[offset] = sum_hidden;
        warp_sum[offset + 16] = sum_embed; 
    }

    __syncthreads();

    if (col < 32) {
        float warp_sum_ = warp_sum[col];
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 8, 16);
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 4, 16);
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 2, 16);
        warp_sum_ += __shfl_xor_sync(0xffffffff, warp_sum_, 1, 16);

        float factor = __frcp_rn(dim << 1);
        if ((col == 0) || (col == 16)) {
            warp_sum[col] = __frsqrt_rn(__fmaf_rn(warp_sum_, factor, eps));
        }
    }
    
    __syncthreads();
    
    sum_hidden = warp_sum[0];
    sum_embed = warp_sum[16];
    
    iIter = 0;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 w_hidden = hidden_weight[i];
        T2 w_embed = embed_weight[i];

        int offset = row * dim * 2 + i;
        output[offset + dim] = T2(
            T(sum_hidden * R_hidden[iIter][0] * float(w_hidden.x)),
            T(sum_hidden * R_hidden[iIter][1] * float(w_hidden.y))
        );
        output[offset] = T2(
            T(sum_embed * R_embed[iIter][0] * float(w_embed.x)),
            T(sum_embed * R_embed[iIter][1] * float(w_embed.y))
        );
        ++iIter;
    }
}

template <typename T>
void rms_norm2(const Stream& stream, int num_tokens, int dim, const T* input_hidden, const T* hidden_weight, const T* input_embed, const T* embed_weight, T* output, float eps) {
    using T2 = typename TypeTraits<T>::half2;
    rms_norm2_kernel<T, T2><<<num_tokens, 512, 0, stream.stream>>>((dim >> 1), (T2*)input_hidden, (T2*)hidden_weight, (T2*)input_embed, (T2*)embed_weight, (T2*)output, eps);
}

template <typename T>
void add_and_rms_norm2(const Stream& stream, int num_tokens, int dim, const T* input_hidden, const T* prev_hidden, const T* hidden_weight, const T* input_embed, const T* embed_weight, T* output, float eps) {
    using T2 = typename TypeTraits<T>::half2;
    add_and_rms_norm2_kernel<T, T2><<<num_tokens, 512, 0, stream.stream>>>((dim >> 1), (T2*)input_hidden, (T2*)prev_hidden, (T2*)hidden_weight, (T2*)input_embed, (T2*)embed_weight, (T2*)output, eps);
}

}

template <typename T>
struct Norm {
    T* output;
    virtual void init_weight_ptr(Memory* memory) = 0;
    virtual int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) = 0;
    virtual void load_to_storage(std::string name, void* ptr) = 0;
    virtual void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output, T* tgt=nullptr, T* embed=nullptr) = 0;
};

template <typename T>
struct RMSNorm : Norm<T> {
    int dim;
    float eps;
    T* weight;

    RMSNorm(int dim, float eps) {
        this->dim = dim;
        this->eps = eps;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(dim * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * dim * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        cudaMemcpy((void*)weight, ptr, dim * sizeof(T), cudaMemcpyHostToDevice);
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output, T* tgt=nullptr, T* embed=nullptr) {
        if (tgt == nullptr) tgt = this->output;
        if (prev_output == nullptr) {
            rms_norm(stream, num_tokens, this->dim, input, this->weight, tgt, this->eps);
        } else {
            add_and_rms_norm(stream, num_tokens, this->dim, input, prev_output, this->weight, tgt, this->eps);
        }
    }
};


template<typename T>
struct Skip : Norm<T> {
    int dim;

    Skip(int dim, float eps = 1e-5) {
        this->dim = dim;
    }

    void init_weight_ptr(Memory* memory) { }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * dim * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {}

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output, T* tgt=nullptr, T* embed=nullptr) {
        if (tgt == nullptr) tgt = this->output;
        if (prev_output == nullptr) {
            cudaMemcpy(tgt, input, sizeof(T) * this->dim * num_tokens, cudaMemcpyDeviceToDevice);
        } else {
            elementwise_add(stream, num_tokens, this->dim, input, prev_output, tgt);
        }
    }
};


template <typename T>
struct RMSNorm2 : Norm<T> {
    int dim;
    float eps;

    T* weight; // hidden_weight dim + embed_weight dim

    T* output;

    RMSNorm2(int dim, float eps) {
        this->dim = dim;
        this->eps = eps;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(2 * dim * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t end_offset = memory->allocate((void**)&this->output, offset, 2 * num_tokens * dim * sizeof(T));
        Norm<T>::output = this->output;
        return end_offset;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("hidden_norm") != std::string::npos) {
            cudaMemcpy((void*)weight, ptr, dim * sizeof(T), cudaMemcpyHostToDevice);
        } else if (name.find("input_layernorm") != std::string::npos) {
            cudaMemcpy((void*)(weight + dim), ptr, dim * sizeof(T), cudaMemcpyHostToDevice);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output, T* tgt=nullptr, T* embed=nullptr) {
        if (tgt == nullptr) tgt = this->output;
        if (prev_output == nullptr) {
            rms_norm2(stream, num_tokens, this->dim, input, this->weight, embed, this->weight + this->dim, tgt, this->eps);
        } else {
            add_and_rms_norm2(stream, num_tokens, this->dim, input, prev_output, this->weight, embed, this->weight + this->dim, tgt, this->eps);
        }
    }
};