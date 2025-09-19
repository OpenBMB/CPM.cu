# CPM.cu Docker用户指南

本文档面向**用户**，说明如何使用CPM.cu Docker镜像进行LLM推理。


## 快速开始

### 前提条件

- 安装Docker
- NVIDIA GPU驱动 (推荐版本 >= 525.60.13)
- 当前镜像支持的GPU: A100/A800、RTX 30/40/50系列、H100/H800
- Python 版本：支持 3.8–3.12（使用 Docker 时宿主机无需安装 Python）

### 获取镜像

```bash
# 方式1: 使用预构建镜像（推荐）
docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:v1.0.0
docker tag modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:v1.0.0 cpmcu:cuda12.6-release

# 可选：CUDA 12.8（RTX 50 系推荐）
docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.8:v1.0.0
docker tag modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.8:v1.0.0 cpmcu:cuda12.8-release

# 方式2: 本地构建
git clone https://github.com/OpenBMB/CPM.cu.git
cd CPM.cu/docker/cuda12.6
./build.sh
```

本地构建参考[Docker 构建指南](docker_build.md)

## 基本使用

### 交互式运行

```bash
# 启动交互式容器
docker run --gpus all -it cpmcu:cuda12.6-release

# 在容器内运行
python examples/minicpm4/test_generate.py --help
```

### 文本生成示例

```bash
# 使用默认提示
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py

# 使用自定义提示
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py --prompt-text "解释什么是人工智能"

# 使用提示文件
docker run --gpus all \
  -v /path/to/your/prompt.txt:/workspace/prompt.txt \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py --prompt-file /workspace/prompt.txt
```

### API服务器

```bash
# 启动OpenAI兼容的API服务器
docker run --gpus all -p 8000:8000 cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py

# 后台运行
docker run --gpus all -p 8000:8000 -d --name cpmcu-server \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py
```

### 测试API服务器

```bash
# 在另一个终端测试API
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## 高级用法

### 离线使用（宿主机下载后挂载，推荐稳定）

当容器无法直接访问 Hugging Face 或下载速度慢时，建议在宿主机先把模型下载到本地，再挂载到容器中使用。

```bash
# 使用hf下载模型
huggingface-cli download openbmb/MiniCPM4-8B --local-dir /path/to/model

# 同时下载草稿模型与 FRSpec（用于投机采样，建议离线准备）
huggingface-cli download openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu --local-dir /path/to/draft

# 运行时挂载并使用本地路径
docker run --rm --gpus all \
  -v /path/to/model:/workspace/model \
  -v /path/to/draft:/workspace/draft \
  cpmcu:cuda12.6-release \
  bash -lc 'cd examples && python3 minicpm4/test_generate.py \
    --model-path /workspace/model \
    --draft-model-path /workspace/draft \
    --frspec-path /workspace/draft \
    --prompt-text "你好" --num-generate 128 --use-stream false'
```

需要自定义配置 可以通过cpmcu.cli -h查看所有可用参数。
```bash
docker run --rm --gpus all cpmcu:cuda12.6-release python -m cpmcu.cli -h
```


### 模型配置

```bash
# 指定模型路径
docker run --gpus all \
  -v /path/to/model:/workspace/model \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py --model-path /workspace/model

# 调整生成参数
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --max-tokens 512 \
  --temperature 0.7 \
  --top-p 0.9
```

### 内存优化

```bash
# 限制GPU内存使用
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --memory-limit 0.6  # 使用60%的GPU内存

# 启用内存映射
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --use-mmap
```

### 性能调优

```bash
# 启用推测采样
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --draft-model-path /path/to/draft/model

# 调整批处理大小
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py \
  --max-batch-size 32
```

## 常见用例

### 1. 代码生成

```bash
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --prompt-text "写一个Python函数来计算斐波那契数列" \
  --max-tokens 256
```

### 2. 文档摘要

```bash
# 准备长文档
echo "你的长文档内容..." > document.txt

docker run --gpus all \
  -v $(pwd)/document.txt:/workspace/document.txt \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --prompt-text "请总结以下文档的主要内容：$(cat /workspace/document.txt)" \
  --max-tokens 200
```

### 3. 对话系统

```bash
# 启动对话API服务器
docker run --gpus all -p 8000:8000 -d --name chat-server \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py

# 使用Python客户端
python -c "
import openai
client = openai.OpenAI(base_url='http://localhost:8000/v1', api_key='dummy')
response = client.chat.completions.create(
    model='minicpm4',
    messages=[{'role': 'user', 'content': '你好，请介绍一下自己'}]
)
print(response.choices[0].message.content)
"
```

## 数据持久化

### 模型缓存

```bash
# 创建模型缓存目录
mkdir -p ~/.cache/huggingface

# 挂载缓存目录
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py
```

### 输出保存

```bash
# 保存生成结果
docker run --gpus all \
  -v $(pwd)/output:/workspace/output \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --output-file /workspace/output/result.txt
```

## 故障排除

### GPU相关问题

**检查GPU可用性**:
```bash
docker run --gpus all cpmcu:cuda12.6-release nvidia-smi
```

**GPU内存不足**:
```bash
# 减少内存使用
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py --memory-limit 0.5
```

**CUDA版本不兼容**:
```bash
# 检查CUDA版本
docker run --gpus all cpmcu:cuda12.6-release nvcc --version
```

### 模型加载问题

**模型文件不存在**:
```bash
# 检查模型路径
docker run --gpus all \
  -v /path/to/model:/workspace/model \
  cpmcu:cuda12.6-release \
  ls -la /workspace/model
```

**权限问题**:
```bash
# 修复权限
sudo chown -R $(id -u):$(id -g) /path/to/model
```

### 网络问题

**API服务器无法访问**:
```bash
# 检查端口映射
docker ps | grep cpmcu

# 检查防火墙
sudo ufw status
```

**模型下载失败**:
```bash
# 使用代理
docker run --gpus all \
  -e HTTP_PROXY=http://proxy:port \
  -e HTTPS_PROXY=http://proxy:port \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py
```

## 最佳实践

### 生产部署

1. **使用固定标签**: `cpmcu:cuda12.6-release` 而不是 `latest`
2. **设置资源限制**: 使用 `--memory` 和 `--cpus` 参数
3. **健康检查**: 配置容器健康检查
4. **日志管理**: 使用适当的日志驱动

```bash
# 生产部署示例
docker run -d \
  --name cpmcu-prod \
  --gpus all \
  --memory=32g \
  --cpus=8 \
  --restart=unless-stopped \
  -p 8000:8000 \
  -v /data/models:/workspace/models \
  -v /logs:/workspace/logs \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py \
  --model-path /workspace/models/minicpm4 \
  --log-file /workspace/logs/server.log
```

### 开发调试

```bash
# 开发模式（挂载代码目录）
docker run --gpus all -it \
  -v $(pwd):/workspace/dev \
  cpmcu:cuda12.6-release \
  bash

# 调试模式（保留容器）
docker run --gpus all -it --rm=false \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py --debug
```

## 更新和维护

### 更新镜像

```bash
# 拉取最新版本
docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:latest
docker tag modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:latest cpmcu:cuda12.6-release

# 重启服务
docker stop cpmcu-server
docker rm cpmcu-server
docker run --gpus all -p 8000:8000 -d --name cpmcu-server \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py
```
