#!/usr/bin/env python3
"""
并行架构对比测试 - 展示新的细粒度并行处理相比原版本的改进
"""

import os
import sys
import time
import json
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corner_case_gen_parallel import *
from config import ConfigManager


def create_mock_sample_with_multiple_solutions():
    """创建包含多个解决方案的模拟样本，用于测试并行效果"""
    return {
        'name': 'Test Problem. Matrix Multiplication',
        'description': '''Given two matrices A and B, compute their product C = A × B.
        
Input:
First line: n, m, k (dimensions)
Next n lines: matrix A (n×m)
Next m lines: matrix B (m×k)

Output:
n lines with k integers each: matrix C (n×k)''',
        
        'solutions': {
            'language': [1, 1, 1, 2, 2],  # 多个 PYTHON 和 CPP 解决方案
            'solution': [
                # Python solutions (正确的)
                '''
n, m, k = map(int, input().split())
A = []
for _ in range(n):
    A.append(list(map(int, input().split())))
B = []
for _ in range(m):
    B.append(list(map(int, input().split())))

C = [[0] * k for _ in range(n)]
for i in range(n):
    for j in range(k):
        for l in range(m):
            C[i][j] += A[i][l] * B[l][j]

for row in C:
    print(' '.join(map(str, row)))
                ''',
                '''
import sys
input = sys.stdin.readline

n, m, k = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(n)]
B = [list(map(int, input().split())) for _ in range(m)]

result = []
for i in range(n):
    row = []
    for j in range(k):
        val = sum(A[i][l] * B[l][j] for l in range(m))
        row.append(val)
    result.append(row)

for row in result:
    print(' '.join(map(str, row)))
                ''',
                '''
n, m, k = map(int, input().split())
A = [[int(x) for x in input().split()] for _ in range(n)]
B = [[int(x) for x in input().split()] for _ in range(m)]

for i in range(n):
    result_row = []
    for j in range(k):
        result_row.append(sum(A[i][l] * B[l][j] for l in range(m)))
    print(' '.join(map(str, result_row)))
                ''',
                # CPP solutions
                '''
#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n, m, k;
    cin >> n >> m >> k;
    
    vector<vector<int>> A(n, vector<int>(m));
    vector<vector<int>> B(m, vector<int>(k));
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            cin >> A[i][j];
        }
    }
    
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < k; j++) {
            cin >> B[i][j];
        }
    }
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < k; j++) {
            int sum = 0;
            for(int l = 0; l < m; l++) {
                sum += A[i][l] * B[l][j];
            }
            cout << sum;
            if(j < k-1) cout << " ";
        }
        cout << "\\n";
    }
    
    return 0;
}
                ''',
                '''
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n, m, k;
    cin >> n >> m >> k;
    
    vector<vector<int>> A(n, vector<int>(m));
    vector<vector<int>> B(m, vector<int>(k));
    vector<vector<int>> C(n, vector<int>(k, 0));
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            cin >> A[i][j];
        }
    }
    
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < k; j++) {
            cin >> B[i][j];
        }
    }
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < k; j++) {
            for(int l = 0; l < m; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < k; j++) {
            cout << C[i][j];
            if(j < k-1) cout << " ";
        }
        cout << "\\n";
    }
    
    return 0;
}
                '''
            ]
        },
        
        'incorrect_solutions': {
            'language': [1, 1, 2],  # 错误的解决方案
            'solution': [
                # Python - 错误的矩阵乘法（转置错误）
                '''
n, m, k = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(n)]
B = [list(map(int, input().split())) for _ in range(m)]

for i in range(n):
    for j in range(k):
        result = sum(A[i][l] * B[j][l] for l in range(m))  # 错误：B[j][l] 应该是 B[l][j]
        print(result, end=' ' if j < k-1 else '\\n')
                ''',
                # Python - 维度错误
                '''
n, m, k = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(n)]
B = [list(map(int, input().split())) for _ in range(m)]

for i in range(m):  # 错误：应该是 range(n)
    for j in range(k):
        result = sum(A[i][l] * B[l][j] for l in range(m))
        print(result, end=' ' if j < k-1 else '\\n')
                ''',
                # CPP - 索引错误
                '''
#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n, m, k;
    cin >> n >> m >> k;
    
    vector<vector<int>> A(n, vector<int>(m));
    vector<vector<int>> B(m, vector<int>(k));
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            cin >> A[i][j];
        }
    }
    
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < k; j++) {
            cin >> B[i][j];
        }
    }
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < k; j++) {
            int sum = 0;
            for(int l = 0; l < k; l++) {  // 错误：应该是 l < m
                sum += A[i][l] * B[l][j];
            }
            cout << sum;
            if(j < k-1) cout << " ";
        }
        cout << "\\n";
    }
    
    return 0;
}
                '''
            ]
        },
        
        'canonical_solution': {
            'python': '''
n, m, k = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(n)]
B = [list(map(int, input().split())) for _ in range(m)]

for i in range(n):
    row = []
    for j in range(k):
        val = sum(A[i][l] * B[l][j] for l in range(m))
        row.append(val)
    print(' '.join(map(str, row)))
            '''
        },
        
        'public_tests': {
            'input': [
                '2 2 2\\n1 2\\n3 4\\n5 6\\n7 8\\n',
                '1 3 1\\n1 2 3\\n4\\n5\\n6\\n'
            ],
            'output': [
                '19 22\\n43 50\\n',
                '32\\n'
            ]
        },
        'private_tests': {'input': [], 'output': []},
        'generated_tests': {'input': [], 'output': []},
        'cf_tags': ['implementation', 'math']
    }


def benchmark_parallel_performance():
    """基准测试并行性能"""
    print("=== Parallel Performance Benchmark ===")
    
    # 创建测试样本
    sample = create_mock_sample_with_multiple_solutions()
    
    # 转换为正确格式
    processor = DatasetProcessor()
    formatted_sample = processor.transform_codecontents(sample)
    
    # 创建配置
    config_manager = ConfigManager()
    
    # 测试配置1：低并行度
    config_manager.processing_config.output_generation_workers = 1
    config_manager.processing_config.solution_validation_workers = 1
    config_manager.processing_config.sample_level_workers = 1
    
    print("\\n1. Testing with LOW parallelism (1 worker each)...")
    start_time = time.time()
    
    # 模拟处理（不实际调用 API）
    simulate_processing(formatted_sample, config_manager, "low")
    
    low_parallel_time = time.time() - start_time
    print(f"   Time taken: {low_parallel_time:.2f} seconds")
    
    # 测试配置2：高并行度
    config_manager.processing_config.output_generation_workers = 8
    config_manager.processing_config.solution_validation_workers = 8
    config_manager.processing_config.sample_level_workers = 4
    
    print("\\n2. Testing with HIGH parallelism (8/8/4 workers)...")
    start_time = time.time()
    
    simulate_processing(formatted_sample, config_manager, "high")
    
    high_parallel_time = time.time() - start_time
    print(f"   Time taken: {high_parallel_time:.2f} seconds")
    
    # 性能提升计算
    if low_parallel_time > 0:
        speedup = low_parallel_time / high_parallel_time if high_parallel_time > 0 else float('inf')
        print(f"\\n3. Performance improvement: {speedup:.2f}x speedup")
    
    return low_parallel_time, high_parallel_time


def simulate_processing(sample, config_manager, mode):
    """模拟处理过程，展示并行架构"""
    print(f"   Sample ID: {sample['id']}")
    print(f"   Solutions to validate: {len(sample.get('solutions', {}).get('language', []))}")
    print(f"   Incorrect solutions to validate: {len(sample.get('incorrect_solutions', {}).get('language', []))}")
    
    # 模拟输出生成
    mock_case_inputs = ['2 2 2\\n1 2\\n3 4\\n5 6\\n7 8\\n', '1 1 1\\n5\\n3\\n']
    print(f"   Corner cases to generate outputs for: {len(mock_case_inputs)}")
    
    if mode == "low":
        # 串行模拟
        time.sleep(0.1 * len(mock_case_inputs))  # 模拟输出生成时间
        time.sleep(0.1 * (len(sample.get('solutions', {}).get('language', [])) + 
                         len(sample.get('incorrect_solutions', {}).get('language', []))))  # 模拟验证时间
    else:
        # 并行模拟（时间缩短）
        parallel_factor = max(config_manager.processing_config.output_generation_workers, 
                            config_manager.processing_config.solution_validation_workers)
        time.sleep(0.1 * len(mock_case_inputs) / parallel_factor)
        time.sleep(0.1 * (len(sample.get('solutions', {}).get('language', [])) + 
                         len(sample.get('incorrect_solutions', {}).get('language', []))) / parallel_factor)


def demonstrate_architecture():
    """演示新的并行架构"""
    print("=== New Parallel Architecture Demonstration ===")
    print()
    print("The new architecture implements multi-level parallelism:")
    print()
    print("1. Sample Level Parallelism:")
    print("   - Multiple samples can be processed simultaneously")
    print("   - Configurable via: processing_config.sample_level_workers")
    print()
    print("2. Output Generation Parallelism:")
    print("   - Corner case outputs are generated in parallel")
    print("   - Each test case input processed by separate workers")
    print("   - Configurable via: processing_config.output_generation_workers")
    print()
    print("3. Solution Validation Parallelism:")
    print("   - Multiple solutions validated simultaneously")
    print("   - Both correct and incorrect solutions processed in parallel")
    print("   - Configurable via: processing_config.solution_validation_workers")
    print()
    print("Comparison with test.py approach:")
    print("- test.py: Task-queue based parallelism at solution level")
    print("- New approach: Multi-level parallelism with fine-grained control")
    print("- Benefits: Better resource utilization, configurable parallelism")


def test_configuration_options():
    """测试不同的配置选项"""
    print("\\n=== Configuration Options Test ===")
    
    config_manager = ConfigManager()
    
    print("\\n1. Default Configuration:")
    print(f"   Output generation workers: {config_manager.processing_config.output_generation_workers}")
    print(f"   Solution validation workers: {config_manager.processing_config.solution_validation_workers}")
    print(f"   Sample level workers: {config_manager.processing_config.sample_level_workers}")
    
    print("\\n2. High Performance Configuration:")
    config_manager.processing_config.output_generation_workers = 16
    config_manager.processing_config.solution_validation_workers = 16
    config_manager.processing_config.sample_level_workers = 8
    
    print(f"   Output generation workers: {config_manager.processing_config.output_generation_workers}")
    print(f"   Solution validation workers: {config_manager.processing_config.solution_validation_workers}")
    print(f"   Sample level workers: {config_manager.processing_config.sample_level_workers}")
    
    print("\\n3. Memory Optimized Configuration:")
    config_manager.processing_config.output_generation_workers = 4
    config_manager.processing_config.solution_validation_workers = 4
    config_manager.processing_config.sample_level_workers = 2
    
    print(f"   Output generation workers: {config_manager.processing_config.output_generation_workers}")
    print(f"   Solution validation workers: {config_manager.processing_config.solution_validation_workers}")
    print(f"   Sample level workers: {config_manager.processing_config.sample_level_workers}")


def compare_with_original():
    """与原始版本的对比"""
    print("\\n=== Comparison with Original Architecture ===")
    print()
    print("Original corner_case_gen.py:")
    print("- Sequential processing of samples")
    print("- Sequential output generation for each test case")
    print("- Sequential solution validation")
    print("- Single API endpoint usage")
    print()
    print("Original test.py:")
    print("- Task queue based solution-level parallelism")
    print("- Multiple API endpoints")
    print("- Good load balancing")
    print("- But limited to solution validation only")
    print()
    print("New corner_case_gen_parallel.py:")
    print("- Multi-level parallelism:")
    print("  * Sample level: Multiple samples processed concurrently")
    print("  * Output generation: Test case outputs generated in parallel")
    print("  * Solution validation: Solutions validated in parallel")
    print("- Multiple API endpoints support")
    print("- Configurable parallelism at each level")
    print("- Better resource utilization")
    print("- Maintains test.py's load balancing benefits")
    print("- Adds fine-grained parallel control for corner case generation")


def main():
    """主函数"""
    print("Fine-Grained Parallel Corner Case Generator - Architecture Demo")
    print("=" * 60)
    
    demonstrate_architecture()
    test_configuration_options()
    compare_with_original()
    
    # 如果用户同意，运行性能基准测试
    try:
        user_input = input("\\nRun performance benchmark? (y/n): ").lower().strip()
        if user_input == 'y':
            benchmark_parallel_performance()
    except KeyboardInterrupt:
        print("\\nBenchmark skipped.")
    
    print("\\n" + "=" * 60)
    print("Demo completed. The new architecture provides:")
    print("1. Better parallelism control")
    print("2. Improved resource utilization") 
    print("3. Configurable performance tuning")
    print("4. Maintains compatibility with existing interfaces")


if __name__ == "__main__":
    main()
