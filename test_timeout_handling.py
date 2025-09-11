#!/usr/bin/env python3
"""
测试 TimeLimitExceeded 分类和处理逻辑的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solutions_eval_original_test import is_time_limit_exceeded

def test_timeout_classification():
    """测试超时类型分类功能"""
    
    print("Testing TimeLimitExceeded classification:")
    
    # 测试案例 1: Sandbox 内部阻塞，无输出
    sandbox_blocked_response = {
        "tests": [
            {
                "exec_info": {
                    "status": "Failed",
                    "run_result": {
                        "status": "TimeLimitExceeded",
                        "stdout": ""  # 无输出
                    }
                }
            }
        ]
    }
    
    result_1 = is_time_limit_exceeded(sandbox_blocked_response)
    print(f"Test 1 (Sandbox blocked): {result_1} (expected: 'sandbox_blocked')")
    
    # 测试案例 2: 代码真实运行超时，有部分输出
    real_timeout_response = {
        "tests": [
            {
                "exec_info": {
                    "status": "Failed",
                    "run_result": {
                        "status": "TimeLimitExceeded",
                        "stdout": "1 2 3 4 5"  # 有部分输出
                    }
                }
            }
        ]
    }
    
    result_2 = is_time_limit_exceeded(real_timeout_response)
    print(f"Test 2 (Real timeout): {result_2} (expected: 'real_timeout')")
    
    # 测试案例 3: 正常成功执行
    normal_response = {
        "tests": [
            {
                "exec_info": {
                    "status": "Success",
                    "run_result": {
                        "status": "Success",
                        "stdout": "Hello World"
                    }
                }
            }
        ]
    }
    
    result_3 = is_time_limit_exceeded(normal_response)
    print(f"Test 3 (Normal execution): {result_3} (expected: False)")
    
    # 测试案例 4: 编译超时
    compile_timeout_response = {
        "tests": [
            {
                "exec_info": {
                    "status": "Failed",
                    "compile_result": {
                        "status": "TimeLimitExceeded"
                    },
                    "run_result": None
                }
            }
        ]
    }
    
    result_4 = is_time_limit_exceeded(compile_timeout_response)
    print(f"Test 4 (Compile timeout): {result_4} (expected: 'sandbox_blocked')")
    
    # 测试案例 5: stdout 为 None 的情况
    stdout_none_response = {
        "tests": [
            {
                "exec_info": {
                    "status": "Failed",
                    "run_result": {
                        "status": "TimeLimitExceeded",
                        "stdout": None
                    }
                }
            }
        ]
    }
    
    result_5 = is_time_limit_exceeded(stdout_none_response)
    print(f"Test 5 (stdout is None): {result_5} (expected: 'sandbox_blocked')")
    
    # 验证结果
    expected_results = ['sandbox_blocked', 'real_timeout', False, 'sandbox_blocked', 'sandbox_blocked']
    actual_results = [result_1, result_2, result_3, result_4, result_5]
    
    if actual_results == expected_results:
        print("✅ All timeout classification tests passed!")
    else:
        print("❌ Some tests failed!")
        print(f"Expected: {expected_results}")
        print(f"Actual: {actual_results}")

def test_config_timeout_logic():
    """测试配置超时逻辑"""
    
    print("\nTesting config timeout logic:")
    
    # 模拟配置修改逻辑
    def simulate_timeout_handling(tle_type, config, has_increased_timeout):
        if tle_type == "real_timeout" and not has_increased_timeout:
            config['compile_timeout'] = 1000
            config['run_timeout'] = 1000
            return True, "Increased timeout to 1000s"
        elif tle_type == "sandbox_blocked":
            return False, "Sandbox blocked, normal retry"
        else:
            return False, "Other timeout case"
    
    # 测试真实超时情况
    config1 = {'compile_timeout': 20, 'run_timeout': 20}
    increased1, message1 = simulate_timeout_handling("real_timeout", config1, False)
    print(f"Real timeout handling: {message1}")
    print(f"Config after: {config1}")
    print(f"Timeout increased: {increased1}")
    
    # 测试 sandbox 阻塞情况
    config2 = {'compile_timeout': 20, 'run_timeout': 20}
    increased2, message2 = simulate_timeout_handling("sandbox_blocked", config2, False)
    print(f"Sandbox blocked handling: {message2}")
    print(f"Config after: {config2}")
    print(f"Timeout increased: {increased2}")
    
    # 验证结果
    if config1['run_timeout'] == 1000 and config2['run_timeout'] == 20:
        print("✅ Config timeout logic works correctly!")
    else:
        print("❌ Config timeout logic failed!")

if __name__ == "__main__":
    test_timeout_classification()
    test_config_timeout_logic()
