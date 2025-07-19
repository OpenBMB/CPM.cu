#!/usr/bin/env python3
"""
MTBench 性能数据分析脚本
分析JSON文件中的性能指标并生成CSV文件用于观测
"""

import json
import csv
import sys
import statistics
import glob
import os
from pathlib import Path

def calculate_performance_metrics(record):
    """
    计算单条记录的性能指标
    """
    # 基础数据提取
    record_id = record['id']
    category = record.get('category', 'unknown')
    
    # 时间数据
    timing = record['timing']
    prefill_time = timing['prefill_time']
    decode_time = timing['decode_time']
    total_time = timing['total_time']
    
    # Token数据
    tokens = record['tokens']
    input_length = tokens['input_length']  # Prefill Length
    output_length = tokens['output_length']  # Decode Length
    
    # Accept lengths数据
    accept_lengths = record['accept_lengths']
    mean_accept_length = statistics.mean(accept_lengths) if accept_lengths else 0
    
    # 计算速度指标
    prefill_speed = input_length / prefill_time if prefill_time > 0 else 0
    decode_speed = output_length / decode_time if decode_time > 0 else 0
    
    return {
        'id': record_id,
        'category': category,
        'prefill_length_tokens': input_length,
        'prefill_time_s': prefill_time,
        'prefill_speed_tokens_per_s': prefill_speed,
        'mean_accept_length_tokens': mean_accept_length,
        'decode_length_tokens': output_length,
        'decode_time_s': decode_time,
        'decode_speed_tokens_per_s': decode_speed,
        'total_time_s': total_time
    }

def calculate_average_metrics(performance_data):
    """
    计算所有记录的平均值指标
    """
    if not performance_data:
        return {}
    
    # 计算各项指标的平均值
    avg_prefill_length = statistics.mean(d['prefill_length_tokens'] for d in performance_data)
    avg_prefill_time = statistics.mean(d['prefill_time_s'] for d in performance_data)
    avg_prefill_speed = statistics.mean(d['prefill_speed_tokens_per_s'] for d in performance_data)
    avg_mean_accept_length = statistics.mean(d['mean_accept_length_tokens'] for d in performance_data)
    avg_decode_length = statistics.mean(d['decode_length_tokens'] for d in performance_data)
    avg_decode_time = statistics.mean(d['decode_time_s'] for d in performance_data)
    avg_decode_speed = statistics.mean(d['decode_speed_tokens_per_s'] for d in performance_data)
    avg_total_time = statistics.mean(d['total_time_s'] for d in performance_data)
    
    return {
        'id': 'AVERAGE',
        'category': 'average',
        'prefill_length_tokens': round(avg_prefill_length, 2),
        'prefill_time_s': round(avg_prefill_time, 4),
        'prefill_speed_tokens_per_s': round(avg_prefill_speed, 2),
        'mean_accept_length_tokens': round(avg_mean_accept_length, 4),
        'decode_length_tokens': round(avg_decode_length, 2),
        'decode_time_s': round(avg_decode_time, 4),
        'decode_speed_tokens_per_s': round(avg_decode_speed, 2),
        'total_time_s': round(avg_total_time, 4)
    }

def analyze_json_file(json_file_path, output_csv_path):
    """
    分析JSON文件并生成CSV报告
    """
    print(f"正在分析文件: {json_file_path}")
    
    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误：无法读取JSON文件 - {e}")
        return False
    
    # 提取结果数据
    results = data.get('results', [])
    print(f"找到 {len(results)} 条记录")
    
    # 计算每条记录的性能指标
    performance_data = []
    for record in results:
        try:
            metrics = calculate_performance_metrics(record)
            performance_data.append(metrics)
        except Exception as e:
            print(f"警告：处理记录 ID {record.get('id', 'unknown')} 时出错 - {e}")
            continue
    
    # 生成CSV文件
    if performance_data:
        fieldnames = [
            'id', 'category', 
            'prefill_length_tokens', 'prefill_time_s', 'prefill_speed_tokens_per_s',
            'mean_accept_length_tokens', 
            'decode_length_tokens', 'decode_time_s', 'decode_speed_tokens_per_s',
            'total_time_s'
        ]
        
        # 计算平均值记录
        avg_record = calculate_average_metrics(performance_data)
        
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(performance_data)
                # 添加平均值行
                writer.writerow(avg_record)
            
            print(f"成功生成CSV文件: {output_csv_path}")
            print(f"共处理 {len(performance_data)} 条原始记录 + 1条平均值记录 = {len(performance_data) + 1} 条总记录")
            
            # 生成Markdown表格文件
            md_output_path = output_csv_path.replace('.csv', '.md')
            generate_markdown_table(performance_data, avg_record, md_output_path)
            
            # 打印汇总统计
            print_summary_statistics(performance_data)
            
            return True
        except Exception as e:
            print(f"错误：无法写入CSV文件 - {e}")
            return False
    else:
        print("错误：没有有效的性能数据")
        return False

def generate_markdown_table(performance_data, avg_record, output_md_path):
    """
    生成Markdown格式的表格文件
    """
    try:
        with open(output_md_path, 'w', encoding='utf-8') as f:
            # 写入表格标题
            f.write("# MTBench 性能数据分析表格\n\n")
            f.write("## 详细性能数据\n\n")
            
            # 写入表头
            f.write("| ID | Category | Prefill Length (tokens) | Prefill Time (s) | Prefill Speed (tokens/s) | Mean Accept Length (tokens) | Decode Length (tokens) | Decode Time (s) | Decode Speed (tokens/s) | Total Time (s) |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            
            # 写入数据行
            for record in performance_data:
                f.write(f"| {record['id']} | {record['category']} | {record['prefill_length_tokens']} | {record['prefill_time_s']:.4f} | {record['prefill_speed_tokens_per_s']:.2f} | {record['mean_accept_length_tokens']:.4f} | {record['decode_length_tokens']} | {record['decode_time_s']:.4f} | {record['decode_speed_tokens_per_s']:.2f} | {record['total_time_s']:.4f} |\n")
            
            # 写入平均值行（加粗显示）
            f.write(f"| **{avg_record['id']}** | **{avg_record['category']}** | **{avg_record['prefill_length_tokens']}** | **{avg_record['prefill_time_s']}** | **{avg_record['prefill_speed_tokens_per_s']}** | **{avg_record['mean_accept_length_tokens']}** | **{avg_record['decode_length_tokens']}** | **{avg_record['decode_time_s']}** | **{avg_record['decode_speed_tokens_per_s']}** | **{avg_record['total_time_s']}** |\n")
            
            # 添加汇总信息
            f.write("\n## 数据汇总\n\n")
            f.write(f"- **总记录数**: {len(performance_data)} 条原始数据 + 1 条平均值\n")
            f.write(f"- **ID范围**: {min(d['id'] for d in performance_data)} - {max(d['id'] for d in performance_data)}\n")
            f.write(f"- **平均Prefill速度**: {avg_record['prefill_speed_tokens_per_s']} tokens/s\n")
            f.write(f"- **平均Decode速度**: {avg_record['decode_speed_tokens_per_s']} tokens/s\n")
            f.write(f"- **平均Accept长度**: {avg_record['mean_accept_length_tokens']} tokens\n")
            f.write(f"- **平均总时间**: {avg_record['total_time_s']} s\n")
            
        print(f"成功生成Markdown文件: {output_md_path}")
        return True
    except Exception as e:
        print(f"错误：无法写入Markdown文件 - {e}")
        return False

def print_summary_statistics(performance_data):
    """
    打印汇总统计信息
    """
    print("\n" + "="*60)
    print("性能数据汇总统计")
    print("="*60)
    
    if not performance_data:
        print("没有数据可以统计")
        return
    
    # 计算统计指标
    prefill_speeds = [d['prefill_speed_tokens_per_s'] for d in performance_data]
    decode_speeds = [d['decode_speed_tokens_per_s'] for d in performance_data]
    mean_accept_lengths = [d['mean_accept_length_tokens'] for d in performance_data]
    total_times = [d['total_time_s'] for d in performance_data]
    
    print(f"记录总数: {len(performance_data)}")
    print(f"ID范围: {min(d['id'] for d in performance_data)} - {max(d['id'] for d in performance_data)}")
    print(f"Prefill速度 - 平均: {statistics.mean(prefill_speeds):.1f} tokens/s, 最大: {max(prefill_speeds):.1f} tokens/s")
    print(f"Decode速度 - 平均: {statistics.mean(decode_speeds):.1f} tokens/s, 最大: {max(decode_speeds):.1f} tokens/s")
    print(f"平均Accept长度 - 平均: {statistics.mean(mean_accept_lengths):.2f} tokens")
    print(f"总时间 - 平均: {statistics.mean(total_times):.2f}s, 最大: {max(total_times):.2f}s")

def find_json_files(logs_dir):
    """
    搜索logs目录下的所有JSON文件
    """
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"错误：logs目录不存在 {logs_dir}")
        return []
    
    json_files = list(logs_path.glob("*.json"))
    print(f"在 {logs_dir} 中找到 {len(json_files)} 个JSON文件")
    return json_files

def get_output_paths(json_file_path, performance_dir):
    """
    根据输入JSON文件路径生成输出CSV和Markdown文件路径
    """
    json_filename = Path(json_file_path).stem  # 获取不带扩展名的文件名
    csv_path = Path(performance_dir) / f"{json_filename}.csv"
    md_path = Path(performance_dir) / f"{json_filename}.md"
    return csv_path, md_path

def should_skip_file(json_file_path, performance_dir):
    """
    检查是否应该跳过这个文件（输出文件已存在）
    """
    csv_path, md_path = get_output_paths(json_file_path, performance_dir)
    if csv_path.exists() and md_path.exists():
        print(f"跳过 {Path(json_file_path).name}：输出文件已存在")
        return True
    return False

def main():
    # 设置目录路径
    logs_dir = "results/logs"
    performance_dir = "results/performance"
    
    # 确保performance目录存在
    Path(performance_dir).mkdir(parents=True, exist_ok=True)
    
    # 搜索所有JSON文件
    json_files = find_json_files(logs_dir)
    
    if not json_files:
        print("未找到任何JSON文件")
        sys.exit(1)
    
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    print("\n开始批量处理JSON文件...")
    print("=" * 60)
    
    for json_file in json_files:
        print(f"\n处理文件: {json_file.name}")
        
        # 检查是否应该跳过
        if should_skip_file(json_file, performance_dir):
            skipped_count += 1
            continue
        
        # 生成输出路径
        csv_path, md_path = get_output_paths(json_file, performance_dir)
        
        # 执行分析
        success = analyze_json_file(str(json_file), str(csv_path))
        
        if success:
            processed_count += 1
            print(f"✓ 成功处理: {json_file.name}")
            print(f"  - CSV: {csv_path}")
            print(f"  - MD:  {md_path}")
        else:
            failed_count += 1
            print(f"✗ 处理失败: {json_file.name}")
    
    # 打印汇总结果
    print("\n" + "=" * 60)
    print("批量处理完成!")
    print(f"总文件数: {len(json_files)}")
    print(f"成功处理: {processed_count}")
    print(f"跳过文件: {skipped_count}")
    print(f"处理失败: {failed_count}")
    
    if failed_count > 0:
        print("\n注意：有文件处理失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main() 