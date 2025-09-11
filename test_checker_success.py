#!/usr/bin/env python3
"""
测试 checker_success 函数的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from results_summary import checker_success

def test_checker_success_function():
    """测试 checker_success 函数的各种情况"""
    
    print("Testing checker_success function:")
    
    # 测试案例 1: 没有 checker_info
    test_case_1 = {
        'result': {
            'tests': [
                {
                    'exec_info': {'status': 'Success'}
                    # 没有 checker_info
                }
            ]
        }
    }
    result_1 = checker_success(test_case_1)
    print(f"Test 1 (No checker_info): {result_1} (expected: 3)")
    
    # 测试案例 2: checker_info 中有 TimeLimitExceeded
    test_case_2 = {
        'result': {
            'tests': [
                {
                    'exec_info': {'status': 'Success'},
                    'checker_info': {
                        'tests': [
                            {
                                'exec_info': {
                                    'status': 'Failed',
                                    'run_result': {
                                        'status': 'TimeLimitExceeded'
                                    }
                                }
                            }
                        ]
                    }
                }
            ]
        }
    }
    result_2 = checker_success(test_case_2)
    print(f"Test 2 (Checker TLE): {result_2} (expected: 0)")
    
    # 测试案例 3: checker_info 中有其他错误
    test_case_3 = {
        'result': {
            'tests': [
                {
                    'exec_info': {'status': 'Success'},
                    'checker_info': {
                        'tests': [
                            {
                                'exec_info': {
                                    'status': 'Failed',
                                    'run_result': {
                                        'status': 'RuntimeError'
                                    }
                                }
                            }
                        ]
                    }
                }
            ]
        }
    }
    result_3 = checker_success(test_case_3)
    print(f"Test 3 (Checker Other Error): {result_3} (expected: 1)")
    
    # 测试案例 4: checker_info 成功
    test_case_4 = {
        'result': {
            'tests': [
                {
                    'exec_info': {'status': 'Success'},
                    'checker_info': {
                        'tests': [
                            {
                                'exec_info': {
                                    'status': 'Success',
                                    'run_result': {
                                        'status': 'Success'
                                    }
                                }
                            }
                        ]
                    }
                }
            ]
        }
    }
    result_4 = checker_success(test_case_4)
    print(f"Test 4 (Checker Success): {result_4} (expected: 2)")
    
    # 测试案例 5: 有 checker_error
    test_case_5 = {
        'result': {
            'tests': [
                {
                    'exec_info': {'status': 'Success'},
                    'checker_error': {
                        'reason': 'compilation failed',
                        'status': 'failed'
                    }
                }
            ]
        }
    }
    result_5 = checker_success(test_case_5)
    print(f"Test 5 (Checker Error): {result_5} (expected: 1)")
    
    # 验证结果
    expected_results = [3, 0, 1, 2, 1]
    actual_results = [result_1, result_2, result_3, result_4, result_5]
    
    if actual_results == expected_results:
        print("✅ All tests passed! checker_success function works correctly.")
    else:
        print("❌ Some tests failed!")
        print(f"Expected: {expected_results}")
        print(f"Actual: {actual_results}")

if __name__ == "__main__":
    test_checker_success_function()
