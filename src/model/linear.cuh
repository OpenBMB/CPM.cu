#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../trait.cuh"
#include "../utils.cuh"
#include "elementwise.cuh"

template <typename T>
void linear(const Stream& stream, int num_tokens, int dim_in, int dim_out, const T* input, const T* weight, T* output, bool transposed=true, bool inplace=false) {
    float alpha = 1.0f;
    float beta = inplace ? 1.0f : 0.0f;
    if (transposed) {
        cublasCheck(cublasGemmEx(stream.cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_out, num_tokens, dim_in,
            &alpha,
            weight, TypeTraits<T>::cublas_type(), dim_in,
            input, TypeTraits<T>::cublas_type(), dim_in,
            &beta,
            output, TypeTraits<T>::cublas_type(), dim_out,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));
    } else {
        cublasCheck(cublasGemmEx(stream.cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_out, num_tokens, dim_in,
            &alpha,
            weight, TypeTraits<T>::cublas_type(), dim_out,
            input, TypeTraits<T>::cublas_type(), dim_in,
            &beta,
            output, TypeTraits<T>::cublas_type(), dim_out,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));
    }
}

template <typename T>
struct Linear {
    int dim_in;
    int dim_out;
    bool transposed;
    bool has_bias;
    T* output;
    T* weight;
    T* bias;

    Linear(int dim_in, int dim_out, bool transposed=true, bool has_bias=false) {
        this->dim_in = dim_in;
        this->dim_out = dim_out;
        this->transposed = transposed;
        this->has_bias = has_bias;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(dim_in * dim_out * sizeof(T));
        if (has_bias) {
            bias = (T*)memory->allocate_for_model(dim_out * sizeof(T));
        }
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * dim_out * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("weight") != std::string::npos) {
            cudaMemcpy((void*)weight, ptr, dim_in * dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else if (name.find("bias") != std::string::npos) {
            cudaMemcpy((void*)bias, ptr, dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* tgt=nullptr, bool inplace=false) {
        if (tgt == nullptr) tgt = this->output;
        linear<T>(stream, num_tokens, dim_in, dim_out, input, weight, tgt, transposed, inplace);
        if (has_bias) {
            batched_add<T>(stream, num_tokens, dim_out, tgt, bias, tgt);
        }
    }
};

template <typename T>
struct LMHead : Linear<T> {
    T* tmp_hidden_size;
    float head_scale;

    LMHead(int dim_in, int dim_out, float head_scale = 1.0, bool transposed=true, bool has_bias=false) : Linear<T>(dim_in, dim_out, transposed, has_bias) {
        this->head_scale = head_scale;
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        offset = Linear<T>::init_output_ptr(memory, num_tokens, offset);
        offset = memory->allocate((void**)&this->tmp_hidden_size, offset, num_tokens * this->dim_in * sizeof(T));
        return offset;
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* tgt=nullptr, bool inplace=false) {
        elementwise_scale(stream, num_tokens, this->dim_in, input, head_scale, tmp_hidden_size);
        Linear<T>::prefill(stream, num_tokens, tmp_hidden_size, tgt, inplace);
    }
};