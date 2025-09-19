# CPM.cu

<strong>[中文版本](./README_ZH.md) | English</strong>

CPM.cu is a lightweight, high-performance CUDA implementation for LLMs, optimized for end-device inference and featuring cutting-edge techniques in **sparse architecture**, **speculative sampling** and **quantization**.

<div id="news"></div>

## 🔥 Project Updates

- [2025.06.06] Optimized for [MiniCPM4](https://github.com/openbmb/minicpm).
    - Support InfLLM-v2 attention kernel
    - Support sliding-window for the MTP layer, optimized for long context
    - Support quantization for the MTP layer
- [2025.05.29] Support Quantization at [SpecMQuant](https://github.com/AI9Stars/SpecMQuant).
    - Support Marlin GPTQ kernel for the LLM
    - Support Speculative Sampling for quantized LLM
- [2025.03.01] Release the first version at [FR-Spec](https://github.com/thunlp/FR-Spec).
    - SOTA Speculative Sampling Implementation
    - Support FR-Spec: Frequency-Ranked Speculative Sampling
    - Support Tree-based verification of Speculative Sampling in Flash-Attention
    - Support Static memory management and memory reuse
    - Support Fused kernels
    - Support Chunked prefill
    - Support CUDA Graph

<div id="demo"></div>

## Demo

https://github.com/user-attachments/assets/ab36fd7a-485b-4707-b72f-b80b5c43d024

<div id="getstart"></div>

## Getting Started

- [Installation](#install)
- [Docker Usage](#docker)
- [Model Weights](#modelweights)
- [Command Line Interface (CLI)](#cli)
- [OpenAI API Service](#openai-api)

<div id="install"></div>

## Installation

### Install from source

This library's build depends on torch and ninja. Please install both before installing this library.

Supported Python versions: 3.8–3.12.

```bash
git clone https://github.com/OpenBMB/CPM.cu.git --recursive
cd CPM.cu
pip install .
```

If you encounter installation issues, please follow the error messages to resolve them or create a GitHub issue. You can use `python setup.py --help-config` to view more installation configuration options.

<div id="docker"></div>

## Docker Usage

We provide pre-built Docker images that support out-of-the-box GPU inference environments.

### Docker Images List

| Image | Description | url |
|-------|-------------|-------|
| cpmcu:cuda12.6-release | CUDA 12.6 release image recommended |modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:v1.0.0|
| cpmcu:cuda12.8-release | CUDA 12.8 develop image, add support for RTX 50 series |modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.8:v1.0.0|
| cpmcu:jetpack6.1| Jetpack 6, add support for Jetson Orin, developing |---------|
| cpmcu:cuda11.8-release | CUDA 11.8 release image, developing |---------|

### Quick Start

```bash
# Pull pre-built image
docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:v1.0.0

docker tag modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:v1.0.0 cpmcu:cuda12.6-release

# Run interactive container
docker run --gpus all -it cpmcu:cuda12.6-release /bin/bash

# Start API server(need to login to huggingface or -v mount model)
docker run --gpus all -p 8000:8000 cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py --apply-sparse 
```

### Offline Usage (Recommended)

```bash
# 1. Download model on host
huggingface-cli download openbmb/MiniCPM4-8B-marlin-cpmcu --local-dir model/MiniCPM4-8B-marlin-cpmcu

#    Also download draft model & FRSpec for speculative decoding (optional)
huggingface-cli download openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu --local-dir model/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu

# 2. Mount directories and run
docker run --rm --gpus all \
  -v /path/to/model:/workspace/model \
  cpmcu:cuda12.6-release \
  bash -lc 'cd examples && python3 minicpm4/test_generate.py \
    --model-path /workspace/model/MiniCPM4-8B-marlin-cpmcu \
    --draft-model-path /workspace/model/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu \
    --frspec-path /workspace/model/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu \
    --prompt-text "Hello" --num-generate 128 --use-stream false'
```

**Detailed Documentation**: [Docker User Guide](doc/en/docker_use.md)

<div id="modelweights"></div>

## Prepare Model

Please follow [MiniCPM4's README](https://github.com/openbmb/minicpm) to download the model weights.

<div id="example"></div>

## Quick Start

We provide a simple example to show how to use CPM.cu to generate text.

```bash
cd examples
python3 minicpm4/test_generate.py --prompt-file <your prompt file>
```

If you don't ​​specify​​ the model path, the scripts will load the model from ​​OpenBMB's Hugging Face repository​​.
If you want to use local paths, we recommend keeping all model filenames unchanged and placing them in the same directory. This way, you can run the model by specifying the directory with the -p parameter. Otherwise, we suggest modifying the paths in the code accordingly.
You can use --help to learn more ​​about the script's features​​.

We also provide a script, `examples/long_prompt_gen.py`, to generate ​​long code summarization.
This script ​​automatically collects code from this repository​​ and prompts ​​the model to "Summarize the code."​

```bash
cd examples
python3 long_prompt_gen.py # generate prompt.txt (for more details, use --help)
python3 minicpm4/test_generate.py --prompt-file ../prompt.txt
```

The output should be of the following format:

```bash
Generated text (streaming output):
--------------------------------------------------
Prefilling: 100.0% (106850/106850 tokens) @ 6565.3 tokens/s - Complete!

<Generated Output HERE>
==================================================
Stream Generation Summary:
==================================================
Prefill length: 106850
Prefill time: 16.36 s
Prefill tokens/s: 6530.77
Mean accept length: 2.50
Decode length: 118
Decode time: 0.76 s
Decode tokens/s: 154.59
```

Where:

- the `Prefill` and `Decode` speed are output by (length, time and token/s).
- the `Mean accept length` is the average length of the accepted tokens when using Speculative Sampling.

<div id="cli"></div>

## Command Line Interface (CLI)

For users who need more granular control over inference parameters (e.g., temperature, generation length), we recommend using the `cpmcu.cli` module directly. This is the most flexible way to perform detailed configuration and testing.

You can view all available parameters by running `python -m cpmcu.cli -h`.

**Example Usage:**

The following command shows how to use the CLI and set the `temperature` to `0.7`:

```bash
python -m cpmcu.cli \
    --model-path openbmb/MiniCPM4-8B \
    --prompt-text "Tell me about Tsinghua University" \
    --temperature 0.7 \
    --use-stream true
```

<div id="openai-api"></div>

## OpenAI API Service

CPM.cu can be deployed as a service compatible with the OpenAI API, making it easy to integrate with existing ecosystems.

### 1. Start the Service

We provide a convenient script to load the model and start a FastAPI service.

```bash
python examples/minicpm4/start_server.py --apply-sparse 
# This script provides a simple configuration, allowing you to deploy the model with a single parameter.

MiniCPM4 Configuration:
  --apply-sparse [APPLY_SPARSE], --apply_sparse [APPLY_SPARSE]
                        Enable sparse attention (default: True)
  --apply-quant [APPLY_QUANT], --apply_quant [APPLY_QUANT]
                        Enable quantization for base model (default: True)
  --apply-eagle [APPLY_EAGLE], --apply_eagle [APPLY_EAGLE]
                        Enable Eagle speculative decoding (default: True)
  --apply-eagle-quant [APPLY_EAGLE_QUANT], --apply_eagle_quant [APPLY_EAGLE_QUANT]
                        Enable quantization for Eagle draft model (default: True)
  --minicpm4-yarn [MINICPM4_YARN], --minicpm4_yarn [MINICPM4_YARN]
                        Enable MiniCPM4 YARN for long context support (default: True)
```
After starting, the service listens on `http://localhost:8000` by default. You can change this using the `--host` and `--port` arguments.

For users who need more granular control over inference parameters (e.g., temperature, generation length), we recommend using the `cpmcu.server` module directly. This is the most flexible way to perform detailed configuration and testing.

You can view all available parameters by running `python -m cpmcu.server -h`.

```bash
python -m cpmcu.server [options]
```

### 2. Test the Service

You can use the `examples/test_openai_api.py` script to test the service. It supports both streaming and non-streaming modes, controllable via command-line arguments.

**Basic Usage:**

```bash
python examples/test_openai_api.py
```

**Testing with different temperatures:**

The script also supports the `--temperature` argument, allowing you to test the model's output with different sampling temperatures.

```bash
python examples/test_openai_api.py --temperature 0.5
```

Currently, only the `/v1/chat/completions` endpoint is supported, and the `model` field in requests is ignored.

## Code Structure

```bash
CPM.cu/
├── src/
│   ├── flash_attn/ # attention kernels: sparse, tree-verification, etc.
│   ├── model/
│   │   ├── minicpm4/ # minicpm4 model
│   │   ├── w4a16_gptq_marlin/ # marlin kernel
│   │   └── ... # common layers
│   ├── entry.cu # pybind: bind cuda and python
│   └── ...
├── cpmcu/ # python interface
└── ...
```

## More

### Word Frequency File Generation
We provide a word frequency generation script for FR-Spec, located at "scripts/fr_spec/gen_fr_index.py". You can run it as follows:
```bash
python scripts/fr_spec/gen_fr_index.py --model_path <your_model_path>
```
You can modify the code to use your own dataset. If your task is in a specific vertical domain, constructing word frequencies tailored to that domain can significantly improve processing speed.

### GPTQ to Marlin Conversion
We provide a script to convert GPTQ-quantized model to Marlin format, located at "scripts/model_convert/gptq2marlin.py". You can run it as follows:
```bash
python scripts/model_convert/gptq2marlin.py \
    --src <gptq_model_path> \
    --dst <marlin_model_path>
```
This script supports MiniCPM, Llama and EAGLE format. It will automatically detect the model type and perform the appropriate conversion.

## Acknowledgments

Our `src/flash_attn` folder modified based on [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v2.6.3/csrc/flash_attn).

We have drawn inspiration from the following repositories:

- [EAGLE](https://github.com/SafeAILab/EAGLE)
- [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

## Citation

Please cite our paper if you find our work valuable.

```
@article{zhao2025fr,
  title={FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling},
  author={Zhao, Weilin and Pan, Tengyu and Han, Xu and Zhang, Yudi and Sun, Ao and Huang, Yuxiang and Zhang, Kaihuo and Zhao, Weilun and Li, Yuxuan and Wang, Jianyong and others},
  journal={arXiv preprint arXiv:2502.14856},
  year={2025}
}

@article{zhang2025specmqaunt,
  title={Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design},
  author={Zhang, Yudi and Zhao, Weilin and Han, Xu and Zhao, Tiejun and Xu, Wang and Cao, Hailong and Zhu, Conghui},
  journal={arXiv preprint arXiv:2505.22179},
  year={2025}
}

@article{minicpm4,
  title={MiniCPM4: Ultra-Efficient LLMs on End Devices},
  author={MiniCPM},
  year={2025}
}