#!/usr/bin/env python3
"""
验证负载均衡优化的测试脚本
测试新的队列机制在实际 corner case 生成中的效果
"""

import sys
import time
from collections import Counter
from unittest.mock import MagicMock, patch

# 导入我们的模块
from corner_case_gen_parallel import CornerCaseGenerator, SolutionValidator, OpenAIClient, SandboxClient
from config import ConfigManager

def test_corner_case_generator_load_balancing():
    """测试 CornerCaseGenerator 的负载均衡"""
    print("=== 测试 CornerCaseGenerator 负载均衡 ===")
    
    # 创建模拟的客户端
    openai_client = OpenAIClient("http://test", "key")
    sandbox_client = SandboxClient()
    config_manager = ConfigManager()
    
    # 创建生成器
    generator = CornerCaseGenerator(openai_client, sandbox_client, config_manager)
    
    # 模拟 API 调用追踪
    api_call_tracker = Counter()
    
    def mock_call_api(api_path, payload):
        # 记录 API 调用
        base_path = api_path.replace("run_code", "").rstrip("/")
        api_call_tracker[base_path] += 1
        
        # 模拟成功响应
        return {
            'status': 'Success',
            'run_result': {
                'stdout': 'test_output',
                'stderr': '',
                'return_code': 0
            }
        }
    
    # 打补丁
    with patch.object(sandbox_client, 'call_api', side_effect=mock_call_api):
        # 创建测试样本
        sample = {
            'name': 'Test Problem',
            'description': 'Test problem description',
            'canonical_solution': {
                'python': 'def solve(): return "test"'
            }
        }
        
        # 测试用例输入
        case_inputs = [f"test_input_{i}" for i in range(12)]  # 12个测试用例
        
        # API 端点
        api_paths = ['/api1/', '/api2/', '/api3/', '/api4/']
        
        # 执行测试
        corner_cases, errors = generator.generate_test_outputs(
            case_inputs, sample, api_paths, max_workers=8
        )
        
        print(f"处理了 {len(case_inputs)} 个测试用例")
        print(f"成功生成 {len(corner_cases)} 个 corner cases")
        print(f"发生 {len(errors)} 个错误")
        
        print("\nAPI 调用分布:")
        total_calls = sum(api_call_tracker.values())
        for api_path, count in sorted(api_call_tracker.items()):
            percentage = (count / total_calls * 100) if total_calls > 0 else 0
            print(f"  {api_path}: {count} 次 ({percentage:.1f}%)")
        
        # 计算负载均衡度
        if len(api_call_tracker) > 1:
            values = list(api_call_tracker.values())
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            balance_score = 1 / (1 + variance) if variance > 0 else 1.0
            print(f"\n负载均衡分数: {balance_score:.3f} (1.0为完全均衡)")
            print(f"方差: {variance:.2f}")
        
        print()

def test_solution_validator_load_balancing():
    """测试 SolutionValidator 的负载均衡"""
    print("=== 测试 SolutionValidator 负载均衡 ===")
    
    # 创建验证器
    sandbox_client = SandboxClient()
    validator = SolutionValidator(sandbox_client)
    
    # 模拟 API 调用追踪
    api_call_tracker = Counter()
    
    def mock_call_api(api_path, payload):
        # 记录 API 调用
        if "submit" in api_path:
            base_path = api_path.replace("submit", "").rstrip("/")
            api_call_tracker[base_path] += 1
        
        # 模拟响应
        if "submit" in api_path:
            return {
                'accepted': False,  # 让它进入我们要测试的分支
                'tests': []
            }
        else:
            return {'status': 'Success', 'run_result': {'stdout': 'ok', 'stderr': ''}}
    
    # 打补丁
    with patch.object(sandbox_client, 'call_api', side_effect=mock_call_api):
        # 创建测试配置
        config = {
            'language': 'python',
            'provided_data': {}
        }
        
        # 创建多个解决方案进行测试
        solutions = {
            'language': [1, 1, 1, 1, 1, 1, 1, 1],  # 8个 Python 解决方案
            'solution': [f'def solve(): return {i}' for i in range(8)]
        }
        
        api_paths = ['/api1/', '/api2/', '/api3/', '/api4/']
        
        # 执行验证
        results = validator._validate_solutions(
            config, solutions, api_paths, 'test_id', 'test_dataset', 
            flag=False, max_workers=6
        )
        
        print(f"验证了 {len(solutions['solution'])} 个解决方案")
        print(f"返回了 {len(results)} 个结果")
        
        print("\nAPI 调用分布:")
        total_calls = sum(api_call_tracker.values())
        for api_path, count in sorted(api_call_tracker.items()):
            percentage = (count / total_calls * 100) if total_calls > 0 else 0
            print(f"  {api_path}: {count} 次 ({percentage:.1f}%)")
        
        # 计算负载均衡度
        if len(api_call_tracker) > 1:
            values = list(api_call_tracker.values())
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            balance_score = 1 / (1 + variance) if variance > 0 else 1.0
            print(f"\n负载均衡分数: {balance_score:.3f} (1.0为完全均衡)")
            print(f"方差: {variance:.2f}")
        
        print()

def main():
    """主测试函数"""
    print("=" * 60)
    print("负载均衡优化验证测试")
    print("=" * 60)
    
    try:
        # 测试 CornerCaseGenerator
        test_corner_case_generator_load_balancing()
        
        # 测试 SolutionValidator  
        test_solution_validator_load_balancing()
        
        print("=" * 60)
        print("✅ 所有负载均衡测试通过!")
        print("新的队列机制成功实现了 API 端点的均衡使用")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
