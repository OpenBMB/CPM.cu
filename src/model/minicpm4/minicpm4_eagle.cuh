#pragma once
#include <type_traits>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <algorithm>
#include <vector>
#include <string>
#include <cuda_fp16.h>
#include "../tree_drafter.cuh"
#include "../eagle.cuh"
#include "minicpm4_model.cuh"
#include "minicpm4_w4a16_gptq_marlin_model.cuh"
#include "../w4a16_gptq_marlin/w4a16_gptq_marlin_layer.cuh"
#include "../w4a16_gptq_marlin/w4a16_gptq_marlin_linear.cuh"

// LogitsHook for debugging speculative inference differences
template<typename T>
class LogitsHook {
private:
    bool enabled;
    std::vector<std::vector<float>> logits_history;
    std::vector<int> tokens_history;
    std::string config_name;
    
public:
    LogitsHook(const std::string& name = "default") : enabled(false), config_name(name) {
        // Check if logging is enabled via environment variable
        const char* enable_logging = getenv("CPMCU_LOG_LOGITS");
        enabled = (enable_logging != nullptr && strcmp(enable_logging, "1") == 0);
        
        if (enabled) {
            printf("🔧 LogitsHook enabled for config: %s\n", config_name.c_str());
        }
    }
    
    void log_logits(int step, int vocab_size, const T* logits, int selected_token = -1) {
        if (!enabled) return;
        
        // Allocate host memory for copying GPU data
        std::vector<float> step_logits(vocab_size);
        
        // Copy GPU data to host memory safely
        if (std::is_same_v<T, half>) {
            // For half precision, we need to handle the conversion carefully
            std::vector<half> host_logits(vocab_size);
            cudaMemcpy(host_logits.data(), logits, vocab_size * sizeof(half), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize(); // Ensure copy is complete
            
            // Convert half to float
            for (int i = 0; i < vocab_size; i++) {
                step_logits[i] = __half2float(host_logits[i]);
            }
        } else {
            // For float types, direct copy and convert
            std::vector<T> host_logits(vocab_size);
            cudaMemcpy(host_logits.data(), logits, vocab_size * sizeof(T), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize(); // Ensure copy is complete
            
            for (int i = 0; i < vocab_size; i++) {
                step_logits[i] = static_cast<float>(host_logits[i]);
            }
        }
        
        logits_history.push_back(step_logits);
        tokens_history.push_back(selected_token);
        
        // Print summary
        auto max_it = std::max_element(step_logits.begin(), step_logits.end());
        auto min_it = std::min_element(step_logits.begin(), step_logits.end());
        
        printf("📊 Step %d [%s]: selected_token=%d, logits_range=[%.6f, %.6f]\n", 
               step, config_name.c_str(), selected_token, *min_it, *max_it);
        
        // Save to file periodically
        if (step % 10 == 0 || step < 5) {
            save_to_file();
        }
    }
    
    void save_to_file() {
        if (!enabled || logits_history.empty()) return;
        
        std::string filename = "logits_" + config_name + "_steps" + std::to_string(logits_history.size()) + ".txt";
        FILE* f = fopen(filename.c_str(), "w");
        if (!f) return;
        
        fprintf(f, "# Logits history for config: %s\n", config_name.c_str());
        fprintf(f, "# Format: step,selected_token,top5_token_ids,top5_logits\n");
        
        for (size_t step = 0; step < logits_history.size(); step++) {
            const auto& logits = logits_history[step];
            int token = tokens_history[step];
            
            // Find top 5 tokens
            std::vector<std::pair<float, int>> logit_pairs;
            for (size_t i = 0; i < logits.size(); i++) {
                logit_pairs.push_back({logits[i], (int)i});
            }
            std::sort(logit_pairs.rbegin(), logit_pairs.rend());
            
            fprintf(f, "%zu,%d", step, token);
            for (int i = 0; i < 5 && i < (int)logit_pairs.size(); i++) {
                fprintf(f, ",%d", logit_pairs[i].second);
            }
            for (int i = 0; i < 5 && i < (int)logit_pairs.size(); i++) {
                fprintf(f, ",%.6f", logit_pairs[i].first);
            }
            fprintf(f, "\n");
        }
        
        fclose(f);
        printf("💾 Saved logits to: %s\n", filename.c_str());
    }
    
    ~LogitsHook() {
        if (enabled) {
            save_to_file();
        }
    }
};

template<typename T, class ModelType, class LayerType, class Fc1Type, class Fc2Type>
struct MiniCPM4EagleImpl : Model {
    int num_layers;
    int num_iter;
    int topk_per_iter;
    int tree_size;
    int total_tried;
    float residual_scale;
    bool use_input_norm;
    bool use_attn_norm;

    ModelType* model = nullptr;
    KVCacheManager<T>* kv_caches = nullptr;
    std::vector<LayerType*> layers;
    Fc1Type *fc1 = nullptr;
    Fc2Type *fc2 = nullptr;
    Linear<T>* lm_head = nullptr;
    int32_t* token_id_remap = nullptr;
    RMSNorm<T> *input_norm1 = nullptr;
    RMSNorm<T> *input_norm2 = nullptr;
    functions::TopK<T>* topk_func = nullptr;
    functions::TopK<T>* topk_func_2 = nullptr;

    T *prev_hidden_state, *prev_embed;
    int num_prev, num_history_tokens;
    int32_t *eagle_position_ids, *eagle_cache_length;
    int *eagle_original_length, eagle_padded_length;
    uint64_t *eagle_mask_2d, *tmp_mask_2d;
    T* eagle_logits;
    T* tried_history_val; int32_t* tried_history_pos;
    int32_t* tried_history_parent = nullptr;
    bool is_first_draft;
    int frspec_vocab_size;

    int32_t *h_best, *d_best;    

    T* tmp_kvcache;

    T* a_tmp = nullptr;
    float* c_tmp = nullptr;

    // Logits debugging hook
    LogitsHook<T>* logits_hook = nullptr;

    MiniCPM4EagleImpl(
        ModelType* model,
        int num_layers,
        int intermediate_size,
        int num_attention_heads,
        int num_key_value_heads,
        int head_dim,
        float rms_norm_eps,
        int num_iter,
        int topk_per_iter,
        int tree_size,
        int group_size = 128,
        int eagle_window_size = 0,
        int frspec_vocab_size = 0,
        float residual_scale = 1.0f,
        bool use_input_norm = true,
        bool use_attn_norm = true
    ) {
        this->model = model;
        this->num_layers = num_layers;
        this->num_iter = num_iter;
        this->topk_per_iter = topk_per_iter;
        this->tree_size = tree_size;
        assert(this->tree_size <= 64);
        this->total_tried = topk_per_iter * topk_per_iter * (num_iter - 1) + topk_per_iter;
        this->frspec_vocab_size = frspec_vocab_size > 0 ? frspec_vocab_size : this->model->vocab_size;
        this->residual_scale = residual_scale;
        this->use_input_norm = use_input_norm;
        this->use_attn_norm = use_attn_norm;

        kv_caches = new KVCacheManager<T>(num_layers, num_key_value_heads, head_dim);
        if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
            fc1 = new W4A16GPTQMarlinLinear<T>(this->model->hidden_size, this->model->hidden_size, group_size, true, true);
            fc2 = new W4A16GPTQMarlinLinear<T>(this->model->hidden_size, this->model->hidden_size, group_size);
        } else {
            fc1 = new Linear<T>(this->model->hidden_size, this->model->hidden_size, true, true);
            fc2 = new Linear<T>(this->model->hidden_size, this->model->hidden_size);
        }
        if (use_input_norm) {
            input_norm1 = new RMSNorm<T>(this->model->hidden_size, rms_norm_eps);
            input_norm2 = new RMSNorm<T>(this->model->hidden_size, rms_norm_eps);
        }
        for (int i = 0; i < num_layers; i++) {
            if constexpr (std::is_same_v<LayerType, W4A16GPTQMarlinLayer<T>>) {
                layers.push_back(new W4A16GPTQMarlinLayer<T>(this->model->hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, group_size, this->residual_scale, eagle_window_size));
            } else {
                layers.push_back(new Layer<T>(this->model->hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, this->residual_scale, eagle_window_size));
            }
        }
        if (this->frspec_vocab_size != this->model->vocab_size) {
            lm_head = new Linear<T>(this->model->hidden_size, this->frspec_vocab_size);
        } else {
            lm_head = this->model->lm_head;
        }

        assert(this->topk_per_iter <= this->tree_size-1);

        topk_func = new functions::TopK<T>(this->frspec_vocab_size, this->topk_per_iter);
        
        // FIX: Use a consistent maximum size for topk_func_2 regardless of num_iter
        // This ensures deterministic behavior across different configurations
        // The maximum possible total_tried is when num_iter is at its maximum reasonable value
        int max_reasonable_iter = 10; // Conservative upper bound
        int max_total_tried = topk_per_iter * topk_per_iter * (max_reasonable_iter - 1) + topk_per_iter;
        int max_tree_candidates = 64 - 1; // Maximum tree_size is 64, so max candidates is 63
        topk_func_2 = new functions::TopK<T>(max_total_tried, max_tree_candidates);

        // Initialize logits debugging hook
        std::string hook_name = "iter" + std::to_string(num_iter) + "_tree" + std::to_string(tree_size);
        logits_hook = new LogitsHook<T>(hook_name);
    }

    void init_weight_ptr(Memory* memory) {
        fc1->init_weight_ptr(memory);
        fc2->init_weight_ptr(memory);
        if (use_input_norm) {
            input_norm1->init_weight_ptr(memory);
            input_norm2->init_weight_ptr(memory);
        }
        for (int i = 0; i < num_layers; i++) {
            layers[i]->init_weight_ptr(memory);
        }
        if (this->frspec_vocab_size != this->model->vocab_size) {
            lm_head->init_weight_ptr(memory);
        }
        if (!use_attn_norm) {
            layers[0]->attn->attn_norm = new Skip<T>(this->model->hidden_size);
        }
        kv_caches->rotary_embedding = this->model->kv_caches->rotary_embedding;
        token_id_remap = (int32_t*)memory->allocate_for_model(this->frspec_vocab_size * sizeof(int32_t));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
            offset = memory->allocate((void**)&this->a_tmp, offset, 2 * num_tokens * this->model->hidden_size * sizeof(T));
            int reduce_max_m = marlin::determine_reduce_max_m(num_tokens, marlin::max_par);
            int reduce_n = 2 * this->model->hidden_size;
            offset = memory->allocate((void**)&this->c_tmp, offset, reduce_max_m * reduce_n * sizeof(float));
        }
        offset = fc1->init_output_ptr(memory, num_tokens, offset);
        offset = fc2->init_output_ptr(memory, num_tokens, offset);
        if (use_input_norm) {
            offset = input_norm1->init_output_ptr(memory, num_tokens, offset);
            offset = input_norm2->init_output_ptr(memory, num_tokens, offset);
        }
        int64_t layer_end = 0;
        for (int i = 0; i < num_layers; i++) {
            layer_end = layers[i]->init_output_ptr(memory, num_tokens, offset);
        }
        offset = layer_end;
        if (this->frspec_vocab_size != this->model->vocab_size) {
            offset = lm_head->init_output_ptr(memory, 64, offset);
        }
        offset = memory->allocate((void**)&eagle_logits, offset, this->topk_per_iter * this->frspec_vocab_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tmp_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        // FIX: Allocate consistent memory size regardless of num_iter configuration
        int max_reasonable_iter = 10; // Must match the value in constructor
        int max_total_tried = topk_per_iter * topk_per_iter * (max_reasonable_iter - 1) + topk_per_iter;
        offset = memory->allocate((void**)&tried_history_val, offset, max_total_tried * sizeof(T));
        offset = memory->allocate((void**)&tried_history_pos, offset, max_total_tried * sizeof(int32_t));
        if (this->num_iter > 1) {
            offset = memory->allocate((void**)&tried_history_parent, offset, this->topk_per_iter * (this->num_iter - 1) * sizeof(int32_t));
        }
        cudaMallocHost(&eagle_original_length, sizeof(int32_t));

        offset = topk_func->init_output_ptr(memory, this->topk_per_iter, offset);
        offset = topk_func_2->init_output_ptr(memory, 1, offset);

        offset = memory->allocate((void**)&prev_hidden_state, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&prev_embed, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&eagle_cache_length, offset, sizeof(int32_t));

        offset = memory->allocate((void**)&d_best, offset, 2 * sizeof(int32_t));
        cudaMallocHost(&h_best, 2 * sizeof(int32_t));
        offset = memory->allocate((void**)&tmp_kvcache, offset, 64 * this->model->kv_caches->num_hidden_layers * 2 * this->model->kv_caches->dim * sizeof(T));
        return offset;
    }

    int init_storage() {
        this->model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);
        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = this->init_output_ptr(this->model->memory, this->model->chunk_length, offset);
        float ratio = float(this->model->num_hidden_layers) / (this->model->num_hidden_layers + this->num_layers);
        if constexpr (std::is_same_v<ModelType, MiniCPM4Impl<T>> || std::is_same_v<ModelType, MiniCPM4W4A16GPTQMarlinModelImpl<T>>) {
            kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, this->model->chunk_length, kv_cache_offset, ratio);
        } else {
            kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        }
        kv_caches->init_output_ptr(this->model->memory, kv_cache_offset);
        return min(kv_caches->budget, this->model->kv_caches->budget);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 5) == "eagle") {
            if (name.substr(0, 9) == "eagle.fc1") {
                fc1->load_to_storage(name, ptr);
            } else if (name.substr(0, 9) == "eagle.fc2") {
                fc2->load_to_storage(name, ptr);
            } else if (name.substr(0, 20) == "eagle.token_id_remap") {
                cudaMemcpy((void*)token_id_remap, ptr, this->frspec_vocab_size * sizeof(int32_t), cudaMemcpyHostToDevice);
            } else if (name.find("eagle.input_norm1") != std::string::npos) {
                if (!use_input_norm) throw std::invalid_argument("norm is not used, but input_norm1 is found");
                input_norm1->load_to_storage(name, ptr);
            } else if (name.find("eagle.input_norm2") != std::string::npos) {
                if (!use_input_norm) throw std::invalid_argument("norm is not used, but input_norm2 is found");
                input_norm2->load_to_storage(name, ptr);
            } else if (name.find("eagle.rotary_emb") != std::string::npos) {
                kv_caches->rotary_embedding->load_to_storage(name, ptr);
            } else {
                std::regex layer_regex("eagle\\.layers\\.(\\d+)\\.(.*)");
                std::smatch matches;
                if (std::regex_search(name, matches, layer_regex)) {
                    int layer_idx = std::stoi(matches[1]);
                    layers[layer_idx]->load_to_storage(matches[2], ptr);
                } else {
                    throw std::invalid_argument("Unsupported name (layer_idx not found): " + name);
                }
            }
        } else {
            this->model->load_to_storage(name, ptr);
            if (name.substr(0, 7) == "lm_head") {
                if (this->frspec_vocab_size != this->model->vocab_size) {
                    remap_copy(calc_stream, this->model->lm_head->weight, this->lm_head->weight, this->model->hidden_size, this->frspec_vocab_size, this->token_id_remap);
                }
            }
        }
    }

    void eagle_prefill(int num_history_tokens) {
        cudaMemcpy(this->prev_embed + (num_prev - 1) * this->model->hidden_size, this->model->embedding->output, this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        if (use_input_norm) {
            this->input_norm1->prefill(calc_stream, num_prev, this->prev_embed, nullptr);
            this->input_norm2->prefill(calc_stream, num_prev, this->prev_hidden_state, nullptr);
            if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
                this->fc1->prefill(calc_stream, num_prev, this->input_norm1->output, this->a_tmp, this->c_tmp);
                this->fc2->prefill(calc_stream, num_prev, this->input_norm2->output, this->a_tmp, this->c_tmp);
            } else {
                this->fc1->prefill(calc_stream, num_prev, this->input_norm1->output);
                this->fc2->prefill(calc_stream, num_prev, this->input_norm2->output);
            }
        } else {
            if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
                this->fc1->prefill(calc_stream, num_prev, this->prev_embed, this->a_tmp, this->c_tmp);
                this->fc2->prefill(calc_stream, num_prev, this->prev_hidden_state, this->a_tmp, this->c_tmp);
            } else {
                this->fc1->prefill(calc_stream, num_prev, this->prev_embed);
                this->fc2->prefill(calc_stream, num_prev, this->prev_hidden_state);
            }
        }
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc1->output, this->fc2->output, this->fc2->output);
        T* layer_output = nullptr;
        for (int i = 0; i < num_layers; i++) {
            this->layers[i]->prefill(num_prev, num_history_tokens, this->fc2->output, layer_output, this->eagle_position_ids, this->kv_caches->caches[i]);
            layer_output = this->layers[i]->output;
        }
        elementwise_scale(calc_stream, num_prev, this->model->hidden_size, layer_output, this->residual_scale);
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc2->output, layer_output, this->fc2->output);
    }

    void eagle_decode(int32_t* cache_length) {
        if (use_input_norm) {
            this->input_norm1->prefill(calc_stream, num_prev, this->prev_embed, nullptr);
            this->input_norm2->prefill(calc_stream, num_prev, this->prev_hidden_state, nullptr);
            if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
                this->fc1->prefill(calc_stream, num_prev, this->input_norm1->output, this->a_tmp, this->c_tmp);
                this->fc2->prefill(calc_stream, num_prev, this->input_norm2->output, this->a_tmp, this->c_tmp);
            } else {
                this->fc1->prefill(calc_stream, num_prev, this->input_norm1->output);
                this->fc2->prefill(calc_stream, num_prev, this->input_norm2->output);
            }
        } else {
            if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
                this->fc1->prefill(calc_stream, num_prev, this->prev_embed, this->a_tmp, this->c_tmp);
                this->fc2->prefill(calc_stream, num_prev, this->prev_hidden_state, this->a_tmp, this->c_tmp);
            } else {
                this->fc1->prefill(calc_stream, num_prev, this->prev_embed);
                this->fc2->prefill(calc_stream, num_prev, this->prev_hidden_state);
            }
        }
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc1->output, this->fc2->output, this->fc2->output);
        T* layer_output = nullptr;
        for (int i = 0; i < num_layers; i++) {
            this->layers[i]->decode(num_prev, this->eagle_padded_length, this->fc2->output, layer_output, this->eagle_position_ids, cache_length, Mask(nullptr), this->kv_caches->caches[i]);
            layer_output = this->layers[i]->output;
        }
        elementwise_scale(calc_stream, num_prev, this->model->hidden_size, layer_output, this->residual_scale);
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc2->output, layer_output, this->fc2->output);
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->model->embedding->prefill(calc_stream, num_tokens, input);
        if (num_history_tokens > 0) {
            this->eagle_prefill(num_history_tokens);
        }

        cudaMemcpy(this->prev_embed, this->model->embedding->output + this->model->hidden_size, (num_tokens - 1) * this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        this->model->prefill_embed(num_tokens, num_history_tokens, this->model->embedding->output, position_ids, output);
        this->prev_hidden_state = this->model->norm->output;
        cudaMemcpy(this->eagle_position_ids, position_ids, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->num_prev = num_tokens;

        this->num_history_tokens = num_history_tokens;
        this->is_first_draft = true;
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->model->decode(num_tokens, padded_length, input, position_ids, cache_length, mask_2d, output);
    }

    void draft(int32_t* tree_draft_ids, int32_t* tree_position_ids, int32_t* cache_length, uint64_t* tree_attn_mask, int32_t* tree_parent) {
        cudaMemcpy(this->eagle_original_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
        this->eagle_padded_length = (this->eagle_original_length[0] + 256 - 1) / 128 * 128;

        // FIX: Initialize tried_history_val to ensure deterministic behavior
        // This prevents uninitialized memory from affecting TopK results
        if (this->is_first_draft) {
            int max_reasonable_iter = 10; // Must match the value in constructor/init_output_ptr
            int max_total_tried = topk_per_iter * topk_per_iter * (max_reasonable_iter - 1) + topk_per_iter;
            // Initialize the entire buffer to negative infinity
            // This ensures that uninitialized entries will never be selected by TopK
            T neg_inf = -std::numeric_limits<T>::infinity();
            T* h_neg_inf_buffer = new T[max_total_tried];
            std::fill(h_neg_inf_buffer, h_neg_inf_buffer + max_total_tried, neg_inf);
            cudaMemcpy(this->tried_history_val, h_neg_inf_buffer, max_total_tried * sizeof(T), cudaMemcpyHostToDevice);
            delete[] h_neg_inf_buffer;
        }


        if (this->is_first_draft) {
            this->model->embedding->prefill(calc_stream, 1, tree_draft_ids);
            this->eagle_prefill(this->num_history_tokens);
        } else {
            this->eagle_decode(cache_length);
        }
        cudaMemcpy(this->eagle_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->eagle_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        repeat(calc_stream, topk_per_iter, 1, 0, this->eagle_position_ids);

        { // d = 0
            lm_head->prefill(calc_stream, 1, this->fc2->output + (num_prev - 1) * this->model->hidden_size, this->eagle_logits);
            log_softmax(calc_stream, 1, this->frspec_vocab_size, this->eagle_logits);
            
            // Log logits for debugging
            if (logits_hook) {
                // Copy logits to CPU for logging (simplified approach)
                cudaDeviceSynchronize();
                logits_hook->log_logits(0, this->frspec_vocab_size, this->eagle_logits);
            }
            
            this->topk_func->prefill(calc_stream, 1, this->eagle_logits);
            cudaMemcpy(this->tried_history_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tried_history_pos, this->topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            if (this->frspec_vocab_size != this->model->vocab_size) {
                remap(calc_stream, topk_per_iter, this->topk_func->topk_pos, this->topk_func_2->topk_pos, this->token_id_remap);
            } else {
                cudaMemcpy(this->topk_func_2->topk_pos, this->topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            }
            cudaMemcpy(this->topk_func_2->topk_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            repeat(calc_stream, topk_per_iter, this->model->hidden_size, num_prev-1, this->fc2->output, this->fc1->output);
            init_tree(calc_stream, topk_per_iter, this->eagle_mask_2d);
        }
        for (int d = 1; d < this->num_iter; ++d) {
            add(calc_stream, 1, this->eagle_cache_length, topk_per_iter);
            this->model->embedding->prefill(calc_stream, topk_per_iter, this->topk_func_2->topk_pos);
            if (use_input_norm) {
                this->input_norm1->prefill(calc_stream, topk_per_iter, this->model->embedding->output, nullptr);
                this->input_norm2->prefill(calc_stream, topk_per_iter, this->fc1->output, nullptr);
                if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
                    this->fc1->prefill(calc_stream, topk_per_iter, this->input_norm1->output, this->a_tmp, this->c_tmp);
                    this->fc2->prefill(calc_stream, topk_per_iter, this->input_norm2->output, this->a_tmp, this->c_tmp);
                } else {
                    this->fc1->prefill(calc_stream, topk_per_iter, this->input_norm1->output);
                    this->fc2->prefill(calc_stream, topk_per_iter, this->input_norm2->output);
                }
            } else {
                if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
                    this->fc2->prefill(calc_stream, topk_per_iter, this->fc1->output, this->a_tmp, this->c_tmp);
                    this->fc1->prefill(calc_stream, topk_per_iter, this->model->embedding->output, this->a_tmp, this->c_tmp);
                } else {
                    this->fc2->prefill(calc_stream, topk_per_iter, this->fc1->output);
                    this->fc1->prefill(calc_stream, topk_per_iter, this->model->embedding->output);
                }
            }
            elementwise_add(calc_stream, topk_per_iter, this->model->hidden_size, this->fc1->output, this->fc2->output, this->fc2->output);
            T* layer_output = nullptr;
            for (int i = 0; i < num_layers; i++) {
                this->layers[i]->decode(topk_per_iter, this->eagle_padded_length, this->fc2->output, layer_output, this->eagle_position_ids, this->eagle_cache_length, Mask(eagle_mask_2d, topk_per_iter, topk_per_iter * d), this->kv_caches->caches[i]);
                layer_output = this->layers[i]->output;
            }
            elementwise_scale(calc_stream, topk_per_iter, this->model->hidden_size, layer_output, this->residual_scale);
            elementwise_add(calc_stream, topk_per_iter, this->model->hidden_size, this->fc2->output, layer_output, this->fc2->output);
            add(calc_stream, topk_per_iter, this->eagle_position_ids, 1);

            lm_head->prefill(calc_stream, topk_per_iter, this->fc2->output, this->eagle_logits);
            log_softmax(calc_stream, topk_per_iter, this->frspec_vocab_size, this->eagle_logits);
            
            // Log logits for debugging (iteration d)
            if (logits_hook) {
                cudaDeviceSynchronize();
                // Log first token's logits in this iteration
                logits_hook->log_logits(d, this->frspec_vocab_size, this->eagle_logits);
            }
            
            this->topk_func->prefill(calc_stream, topk_per_iter, this->eagle_logits);
            cumsum(calc_stream, topk_per_iter, topk_per_iter, this->topk_func->topk_val, this->topk_func_2->topk_val);
            cudaMemcpy(this->tried_history_val + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_val, topk_per_iter * topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tried_history_pos + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_pos, topk_per_iter * topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            this->topk_func_2->prefill(calc_stream, 1, this->topk_func->topk_val, topk_per_iter * topk_per_iter, topk_per_iter);

            cudaMemcpy(this->tmp_mask_2d, this->eagle_mask_2d, topk_per_iter * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            set_parent(calc_stream, topk_per_iter, this->tried_history_parent + (d - 1) * topk_per_iter, this->topk_func_2->topk_pos, topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter);
            update_tree(calc_stream, topk_per_iter, topk_per_iter * d, this->eagle_mask_2d, this->tmp_mask_2d, this->topk_func_2->topk_pos);
            remap_hidden(calc_stream, topk_per_iter, this->model->hidden_size, this->topk_func_2->topk_pos, this->fc2->output, this->fc1->output, topk_per_iter);
            if (this->frspec_vocab_size != this->model->vocab_size) {
                remap_id_fr(calc_stream, topk_per_iter, this->topk_func_2->topk_pos, this->topk_func->topk_pos, this->token_id_remap);
            } else {
                remap_id(calc_stream, topk_per_iter, this->topk_func_2->topk_pos, this->topk_func->topk_pos);
            }
        }

        // Calculate the actual number of valid entries in tried_history_val
        // Only the entries filled during the iterations are valid
        int valid_tried_count = topk_per_iter; // d = 0 contributes topk_per_iter entries
        for (int d = 1; d < this->num_iter; ++d) {
            valid_tried_count += topk_per_iter * topk_per_iter; // each iteration d contributes topk_per_iter^2 entries
        }
        
        // BUG FIX: Only select from actually computed values, not uninitialized memory
        // The original code had a bug where topk_func_2 would read total_tried values,
        // but only valid_tried_count values were actually computed, leading to 
        // non-deterministic behavior due to uninitialized memory access
        int candidates_to_select = min(valid_tried_count, this->tree_size - 1);
        
        this->topk_func_2->prefill(calc_stream, 1, this->tried_history_val, valid_tried_count, candidates_to_select);

        // build tree
        build_dynamic_tree(calc_stream, candidates_to_select + 1, this->eagle_original_length[0], this->topk_per_iter, this->tried_history_parent, this->topk_func_2->topk_pos, tree_position_ids, tree_attn_mask, tree_parent);
        if (this->frspec_vocab_size != this->model->vocab_size) {
            remap_id_fr(calc_stream, candidates_to_select, this->topk_func_2->topk_pos, this->tried_history_pos, this->token_id_remap, tree_draft_ids + 1);
        } else {
            remap_id(calc_stream, candidates_to_select, this->topk_func_2->topk_pos, this->tried_history_pos, tree_draft_ids + 1);
        }

        this->is_first_draft = false;
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, int32_t* tree_parent) {
        verify_draft(calc_stream, num_tokens, pred, gt, position_ids, cache_length, mask_2d, tree_parent, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);

        this->num_prev = h_best[0];
        remap_hidden(calc_stream, this->num_prev, this->model->hidden_size, pred, this->model->norm->output, this->prev_hidden_state);

        fix_kv_cache(calc_stream, h_best[0], this->model->kv_caches->num_hidden_layers * 2, this->model->kv_caches->dim, pred, gt, cache_length, this->model->kv_caches->d_flat_caches, this->tmp_kvcache);

        this->model->embedding->prefill(calc_stream, this->num_prev, pred);
        cudaMemcpy(this->prev_embed, this->model->embedding->output, this->num_prev * this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        make_arange(calc_stream, this->num_prev, cache_length, this->eagle_position_ids);

        if constexpr (std::is_same_v<ModelType, MiniCPM4Impl<T>> || std::is_same_v<ModelType, MiniCPM4W4A16GPTQMarlinModelImpl<T>>) {
            this->model->kv_caches->add_length(h_best[0] - 1);
        }

        return h_best[0];
    }
};