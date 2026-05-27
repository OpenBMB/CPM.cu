# CPM.cu

<strong>中文 | [English Version](./README.md)</strong>

CPM.cu 是一个针对端侧大模型推理设计的轻量、高效的 CUDA 推理框架，核心支持 **稀疏架构**、**投机采样** 和 **低位宽量化** 等前沿技术创新。

<div id="news"></div>

## 🔥 项目进展

- [2025.06.06] 为 [MiniCPM4](https://github.com/openbmb/minicpm) 优化。
    - 支持 InfLLM-v2 注意力内核
    - 支持 MTP 层的滑动窗口，优化长上下文处理
    - 支持 MTP 层的量化
- [2025.05.29] 支持 [SpecMQuant](https://github.com/AI9Stars/SpecMQuant) 的量化。
    - 支持 LLM 的 Marlin GPTQ 内核
    - 支持量化 LLM 的投机采样
- [2025.03.01] 在 [FR-Spec](https://github.com/thunlp/FR-Spec) 发布首个版本。
    - 速度最快的投机采样实现
    - 支持 FR-Spec, 基于词频优化的投机采样
    - 支持 Flash-Attention 中的树状投机采样验证
    - 支持静态内存管理和内存复用
    - 支持计算融合内核
    - 支持分块预填充
    - 支持 CUDA Graph

<div id="demo"></div>

## 效果演示

https://github.com/user-attachments/assets/ab36fd7a-485b-4707-b72f-b80b5c43d024

<div id="getstart"></div>

## 快速开始

- [框架安装](#install)
- [Docker 使用](#docker)
- [模型权重](#modelweights)
- [命令行接口 (CLI)](#cli)
- [OpenAI API 服务](#openai-api)

<div id="install"></div>

## 框架安装

### 从源码安装

本库的构建依赖于 PyTorch 和 Ninja，请在安装本库前确保已正确安装这两个依赖。

支持的 Python 版本：3.8–3.12。

```bash
git clone https://github.com/OpenBMB/CPM.cu.git --recursive
cd CPM.cu
pip install .
```

如遇到安装问题，请根据错误提示进行解决，或通过 GitHub Issues 提交问题反馈。你可以使用 `python setup.py --help-config` 查看更多安装配置信息。

<div id="docker"></div>

## Docker 使用

我们提供了预构建的 Docker 镜像，支持开箱即用的 GPU 推理环境。

### 镜像目录

| 镜像 | 描述 | 镜像链接 |
|-------|-------------|-------|
| cpmcu:cuda12.6-release | 建议镜像，支持RTX 30/40系列 和 H100/H800 等主流GPU |modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:v1.0.0|
| cpmcu:cuda12.8-release | CUDA 12.8, 增加了RTX 50系的支持 |modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.8:v1.0.0|
| cpmcu:jetpack6.1| Jetpack 6, 增加了Jetson Orin的支持, 开发中 |----------|
| cpmcu:cuda11.8-release | CUDA 11.8, 开发中 |----------|

### 快速开始

如果你使用的是 RTX 50 系显卡，建议优先使用下面的 CUDA 12.8 镜像。仓库现有 Docker 文档已经将 CUDA 12.8 作为 RTX 50 系支持的推荐路径。

```bash
# 拉取预构建镜像
docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:v1.0.0

docker tag modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:v1.0.0 cpmcu:cuda12.6-release

# 运行交互式容器
docker run --gpus all -it cpmcu:cuda12.6-release /bin/bash

# 启动 API 服务器(需要登录 huggingface 或 -v 挂载模型)
docker run --gpus all -p 8000:8000 cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py --apply-sparse 
```

```bash
# RTX 50 系快速开始
docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.8:v1.0.0

docker tag modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.8:v1.0.0 cpmcu:cuda12.8-release

docker run --gpus all -it cpmcu:cuda12.8-release /bin/bash
```

### 离线使用（推荐）

```bash
# 1. 在宿主机下载模型
huggingface-cli download openbmb/MiniCPM4-8B-marlin-cpmcu --local-dir model/MiniCPM4-8B-marlin-cpmcu

#    同时下载离线草稿模型与 FRSpec（用于开启投机采样，可选）
huggingface-cli download openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu --local-dir model/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu  

# 2. 挂载模型目录运行
docker run --rm --gpus all \
  -v /path/to/model:/workspace/model \
  cpmcu:cuda12.6-release \
  bash -lc 'cd examples && python3 minicpm4/test_generate.py \
    --model-path /workspace/model/MiniCPM4-8B-marlin-cpmcu \
    --draft-model-path /workspace/model/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu \
    --frspec-path /workspace/model/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu \
    --prompt-text "Hello" --num-generate 128 --use-stream false'
```

**详细文档**: [Docker 用户指南](doc/ch/docker_use.md)

<div id="modelweights"></div>

## 准备模型

请按照 [MiniCPM4 的 README](https://github.com/openbmb/minicpm) 的说明下载模型权重。

<div id="example"></div>

## 运行示例

我们提供了一个简单的示例来展示如何使用 CPM.cu。

```bash
cd examples
python3 minicpm4/test_generate.py --prompt-file <输入文件路径>
```

如果您不指定模型路径，脚本将从 OpenBMB 的 Hugging Face 仓库加载模型。
如果你想使用本地路径，我们推荐不修改所有模型文件名并放在同一目录下，这样可以通过-p指定该目录运行模型。否则建议修改代码中的路径。
您可以使用 --help 了解更多关于脚本的功能。

我们还有一个脚本，`examples/long_prompt_gen.py`，用于生成长代码总结。
这个脚本会自动从本仓库中收集代码，并提示模型"Summarize the code."。

```bash
cd examples
python3 long_prompt_gen.py # 生成 prompt.txt (更多细节请见 --help)
python3 minicpm4/test_generate.py --prompt-file ../prompt.txt
```

输出应为如下格式：

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

其中：

- `Prefill` (输入) 和 `Decode` (输出) 速度通过（长度、时间和 token/s）输出。
- `Mean accept length` (平均接受长度) 是使用投机采样时接受 token 的平均长度。

<div id="cli"></div>

## 命令行接口 (CLI)

对于需要更精细化控制推理参数（如温度、生成长度等）的用户，我们推荐直接使用 `cpmcu.cli` 模块。这是进行详细配置和测试最灵活的方式。

你可以通过 `python -m cpmcu.cli -h` 查看所有可用参数。

**使用示例:**

以下命令展示了如何使用 CLI 接口，并设置 `temperature` 为 `0.7`：

```bash
python -m cpmcu.cli \
    --model-path openbmb/MiniCPM-Llama3-V-2_5-int4 \
    --prompt-text "介绍一下清华大学" \
    --temperature 0.7 \
    --use-stream true
```

<div id="openai-api"></div>

## OpenAI API 服务

CPM.cu 支持部署为一个与 OpenAI API 兼容的服务，方便与现有的生态系统集成。

### 1. 启动服务

启动 OpenAI 兼容的 API 服务器（参数与 `examples/minicpm4/test_generate.py` 相同）：

```bash
python examples/minicpm4/start_server.py --apply-sparse 
# 这个脚本提供了简易的配置，通过一个参数就可以一键部署模型

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
服务启动后，默认监听在 `http://localhost:8000`。你可以通过 `--host` 和 `--port` 参数来修改。

对于需要更精细化控制推理参数（如温度、生成长度等）的用户，我们推荐直接使用 `cpmcu.server` 模块。这是进行详细配置和测试最灵活的方式。

你可以通过 `python -m cpmcu.server -h` 查看所有可用参数。

```bash
python -m cpmcu.server [options]
```

### 2. 测试服务

你可以使用 `examples/test_openai_api.py` 脚本来测试服务。该脚本支持流式和非流式两种模式，并可以通过命令行参数控制。

**基础用法:**

```bash
python examples/test_openai_api.py
```

**测试不同温度:**

该脚本也支持 `--temperature` 参数，方便你测试模型在不同温度下的生成效果。

```bash
python examples/test_openai_api.py --temperature 0.5 [--no-stream]
```

目前服务仅支持 `/v1/chat/completions` 接口，请求中的 `model` 字段会被忽略。

## 代码结构

```bash
CPM.cu/
├── src/
│   ├── flash_attn/ # attention: 稀疏, 投机树状验证等
│   ├── model/
│   │   ├── minicpm4/ # minicpm4 模型
│   │   ├── w4a16_gptq_marlin/ # Marlin GPTQ 计算内核
│   │   └── ... # 通用层
│   ├── entry.cu # pybind: 绑定 CUDA 和 Python
│   └── ...
├── cpmcu/ # Python 接口
└── ...
```
## 更多

### 词频文件生成
我们提供了FR-Spec的词频生成脚本，位于"scripts/fr_spec/gen_fr_index.py"，运行方式如下：
```bash
python scripts/fr_spec/gen_fr_index.py --model_path <your modelpath>
```
你可以修改代码使用自己的数据集。如果你的任务是特定垂直领域，根据领域构造词频对速度提升有显著收益。

### GPTQ转Marlin格式
我们提供了GPTQ量化模型转Marlin格式的转换脚本，位于"scripts/model_convert/gptq2marlin.py"，运行方式如下：
```bash
python scripts/model_convert/gptq2marlin.py \
    --src <gptq_model_path> \
    --dst <marlin_model_path>
```
该脚本支持MiniCPM、Llama和EAGLE格式，会自动检测模型类型并进行相应转换。

## 致谢

我们的 `src/flash_attn` 文件夹基于 [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v2.6.3/csrc/flash_attn) 并进行了修改。

我们从以下仓库中获取了实现灵感：

- [EAGLE](https://github.com/SafeAILab/EAGLE)
- [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

## 引用

如果您觉得我们的工作有价值，请引用我们的论文。

```
@article{zhao2025fr,
  title={FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling},
  author={Zhao, Weilin and Pan, Tengyu and Han, Xu and Zhang, Yudi and Sun, Ao and Huang, Yuxiang and Zhang, Kaihuo and Zhao, Weilun and Li, Yuxuan and Wang, Jianyong and others},
  journal={arXiv preprint arXiv:2502.14856},
  year={2025}
}

@article{zhao2025infllm,
  title={InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation},
  author={Zhao, Weilin and Zhou, Zihan and Su, Zhou and Xiao, Chaojun and Li, Yuxuan and Li, Yanghao and Zhang, Yudi and Zhao, Weilun and Li, Zhen and Huang, Yuxiang and others},
  journal={arXiv preprint arXiv:2509.24663},
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
