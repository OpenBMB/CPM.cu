# MTBench 批量性能数据分析工具

## 概述
这个工具可以批量分析MTBench JSON结果文件，自动搜索`results/logs/`目录下的所有JSON文件，并为每个文件生成对应的性能分析报告。

## 主要功能

### 🔍 智能文件发现
- 自动扫描`results/logs/`目录下的所有`.json`文件
- 显示找到的文件数量和处理进度

### 📊 批量分析处理
- 为每个JSON文件生成对应的CSV和Markdown分析报告
- 保持原始文件名，仅更换扩展名
- 输出文件保存到`results/performance/`目录

### ⚡ 智能跳过机制
- 自动检测已存在的输出文件
- 如果CSV和Markdown文件都已存在，则跳过处理
- 避免重复分析，节省时间

## 使用方法

### 1. 运行批量分析
```bash
cd benchmark
python analyze_performance.py
```

### 2. 输出结构
```
results/
├── logs/                          # 输入JSON文件目录
│   ├── file1.json
│   ├── file2.json
│   └── ...
└── performance/                   # 输出分析文件目录
    ├── file1.csv                 # CSV格式数据
    ├── file1.md                  # Markdown格式表格
    ├── file2.csv
    ├── file2.md
    └── ...
```

## 处理流程

1. **扫描阶段**: 搜索`results/logs/`目录下的所有JSON文件
2. **检查阶段**: 对每个文件检查对应的输出文件是否已存在
3. **处理阶段**: 分析新文件或缺失输出文件的JSON文件
4. **汇总阶段**: 显示处理统计信息

## 输出示例

```
在 results/logs 中找到 3 个JSON文件

开始批量处理JSON文件...
============================================================

处理文件: experiment_001.json
✓ 成功处理: experiment_001.json
  - CSV: results/performance/experiment_001.csv
  - MD:  results/performance/experiment_001.md

处理文件: experiment_002.json
跳过 experiment_002.json：输出文件已存在

处理文件: experiment_003.json
✓ 成功处理: experiment_003.json
  - CSV: results/performance/experiment_003.csv
  - MD:  results/performance/experiment_003.md

============================================================
批量处理完成!
总文件数: 3
成功处理: 2
跳过文件: 1
处理失败: 0
```

## 性能优势

1. **批量处理**: 一次处理所有JSON文件，无需手动指定
2. **智能跳过**: 避免重复分析已处理的文件
3. **错误隔离**: 单个文件失败不影响其他文件处理
4. **进度跟踪**: 实时显示处理进度和结果统计

