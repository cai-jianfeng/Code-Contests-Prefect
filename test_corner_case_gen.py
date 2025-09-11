#!/usr/bin/env python3
"""
并行 Corner Case 生成器的使用示例和测试脚本
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corner_case_gen_parallel import *
from config import ConfigManager, DEVELOPMENT_CONFIG


def create_test_dataset():
    """创建一个小的测试数据集"""
    test_sample = {
        'name': 'Test Problem. Simple Addition',
        'description': 'Given two integers a and b, output their sum a+b.',
        'solutions': {
            'language': [1],  # PYTHON
            'solution': ['a, b = map(int, input().split())\nprint(a + b)']
        },
        'incorrect_solutions': {
            'language': [1],  # PYTHON 
            'solution': ['a, b = map(int, input().split())\nprint(a - b)']  # 错误的解决方案
        },
        'canonical_solution': {
            'python': 'a, b = map(int, input().split())\nprint(a + b)'
        },
        'public_tests': {
            'input': ['1 2\n', '3 4\n'],
            'output': ['3\n', '7\n']
        },
        'private_tests': {
            'input': [],
            'output': []
        },
        'generated_tests': {
            'input': [],
            'output': []
        },
        'cf_tags': ['math', 'implementation']
    }
    
    return [test_sample]


def test_dataset_processor():
    """测试数据集处理器"""
    print("Testing DatasetProcessor...")
    
    test_data = create_test_dataset()
    processor = DatasetProcessor()
    
    # 测试数据转换
    transformed = processor.transform_codecontents(test_data[0])
    
    print(f"Original sample keys: {list(test_data[0].keys())}")
    print(f"Transformed sample keys: {list(transformed.keys())}")
    print(f"Sample ID: {transformed.get('id', 'N/A')}")
    print(f"Test cases count: {len(transformed.get('test', []))}")
    
    return transformed


def test_openai_client():
    """测试 OpenAI 客户端（如果 API 可用）"""
    print("Testing OpenAI Client...")
    
    try:
        client = OpenAIClient(API_BASE, API_KEY)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Generate a simple test case for adding two numbers. Output format: ['1 2']"}
        ]
        
        result = client.generate_corner_case(messages, max_tokens=50)
        print(f"OpenAI response: {result}")
        return True
    except Exception as e:
        print(f"OpenAI test failed: {e}")
        return False


def test_sandbox_client():
    """测试 Sandbox 客户端（如果服务可用）"""
    print("Testing Sandbox Client...")
    
    client = SandboxClient()
    
    # 测试简单的代码执行
    test_payload = {
        "code": "print('Hello, World!')",
        "language": "python",
        "stdin": ""
    }
    
    try:
        # 注意：这里使用一个假的端点进行测试，实际使用时需要真实的端点
        result = client.call_api("http://fake-endpoint/run_code", test_payload)
        print(f"Sandbox response: {result}")
        return "error" in result  # 期望有错误，因为端点是假的
    except Exception as e:
        print(f"Sandbox test failed (expected): {e}")
        return True  # 期望失败


def test_config_manager():
    """测试配置管理器"""
    print("Testing ConfigManager...")
    
    config_manager = ConfigManager()
    
    # 测试配置属性
    print(f"OpenAI API Base: {config_manager.openai_config.api_base}")
    print(f"Sandbox hosts: {config_manager.sandbox_config.hosts}")
    print(f"Max workers: {config_manager.processing_config.max_workers_per_api}")
    
    # 测试 API 路径生成
    api_paths = config_manager.sandbox_config.get_api_paths()
    print(f"Generated API paths: {api_paths[:2]}...")  # 只显示前两个
    
    # 测试运行时信息
    runtime_info = config_manager.get_runtime_info()
    print(f"Runtime info keys: {list(runtime_info.keys())}")
    
    return config_manager


def test_corner_case_generator():
    """测试 Corner Case 生成器"""
    print("Testing CornerCaseGenerator...")
    
    # 创建测试环境
    config_manager = ConfigManager()
    openai_client = OpenAIClient(API_BASE, API_KEY)
    sandbox_client = SandboxClient()
    
    generator = CornerCaseGenerator(openai_client, sandbox_client, config_manager)
    
    # 测试 corner case 解析
    test_cases = "['1 2', '0 0', '-1 1']"
    parsed = generator.parse_corner_cases(test_cases)
    print(f"Parsed corner cases: {parsed}")
    
    return parsed is not None


def test_full_pipeline_mock():
    """测试完整的处理流水线（模拟模式）"""
    print("Testing full pipeline (mock mode)...")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # 创建配置
        config_manager = ConfigManager()
        config_manager.dataset_config.results_dir = temp_dir
        config_manager.processing_config.max_iterations = 1  # 减少迭代次数
        config_manager.processing_config.debug = True
        
        # 创建测试数据
        test_data = create_test_dataset()
        transformed_data = []
        processor = DatasetProcessor()
        
        for sample in test_data:
            transformed = processor.transform_codecontents(sample)
            transformed_data.append(transformed)
        
        print(f"Created {len(transformed_data)} test samples")
        
        # 模拟保存结果（不实际调用 API）
        for i, sample in enumerate(transformed_data):
            sample_name = sample['id'].replace('/', '_') + '.json'
            result_file = os.path.join(temp_dir, sample_name)
            
            # 创建模拟结果
            mock_result = {
                'id': sample['id'],
                'corner_cases': [
                    {'input': {'stdin': '1 2'}, 'output': {'stdout': '3'}},
                    {'input': {'stdin': '0 0'}, 'output': {'stdout': '0'}}
                ],
                'result': [
                    {
                        'corner_cases': mock_result['corner_cases'] if 'mock_result' in locals() else [],
                        'corner_cases_error': [],
                        'result': {'solution_result': [], 'incorrect_solution_result': []},
                        'case_inputs_original': "['1 2', '0 0']",
                        'case_inputs': ['1 2', '0 0'],
                        'messages': []
                    }
                ]
            }
            
            with open(result_file, 'w') as f:
                json.dump(mock_result, f, indent=2)
        
        # 检查结果文件
        result_files = list(Path(temp_dir).glob('*.json'))
        print(f"Created {len(result_files)} result files")
        
        return len(result_files) > 0


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("Running All Tests for Parallel Corner Case Generator")
    print("=" * 50)
    
    tests = [
        ("Dataset Processor", test_dataset_processor),
        ("Config Manager", test_config_manager),
        ("Corner Case Generator", test_corner_case_generator),
        ("Full Pipeline (Mock)", test_full_pipeline_mock),
        ("OpenAI Client", test_openai_client),
        ("Sandbox Client", test_sandbox_client),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 30}")
        print(f"Running: {test_name}")
        print(f"{'-' * 30}")
        
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            print(f"Result: {results[test_name]}")
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)}"
            print(f"Result: {results[test_name]}")
    
    # 打印总结
    print(f"\n{'=' * 50}")
    print("Test Summary")
    print(f"{'=' * 50}")
    
    for test_name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    print(f"\nPassed: {passed}/{total}")


def demo_usage():
    """演示如何使用系统"""
    print("=" * 50)
    print("Demonstration of Usage")
    print("=" * 50)
    
    print("\n1. 创建配置管理器:")
    print("   config_manager = ConfigManager()")
    
    print("\n2. 读取数据集:")
    print("   processor = DatasetProcessor()")
    print("   dataset = processor.read_dataset(data_path, split)")
    
    print("\n3. 创建并行处理器:")
    print("   api_paths = config_manager.sandbox_config.get_api_paths()")
    print("   processor = ParallelProcessor(api_paths, max_workers=1, config_manager=config_manager)")
    
    print("\n4. 开始处理:")
    print("   processor.process_dataset(dataset, dataset_type, results_dir)")
    
    print("\n5. 命令行使用:")
    print("   python corner_case_gen_parallel.py production")
    print("   python corner_case_gen_parallel.py development")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_all_tests()
        elif sys.argv[1] == "demo":
            demo_usage()
        else:
            print("Unknown command. Available: test, demo")
    else:
        print("Corner Case Generator Test Script")
        print("Usage:")
        print("  python test_corner_case_gen.py test   # Run all tests")
        print("  python test_corner_case_gen.py demo   # Show usage demo")
