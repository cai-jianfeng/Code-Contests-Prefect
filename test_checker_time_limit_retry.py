#!/usr/bin/env python3
"""
测试 checker TimeLimitExceeded 重试功能的脚本
"""

import json
import os
import tempfile
from result_refine_parallel_original_test import has_time_limit_exceeded_in_checker_results

def test_checker_time_limit_detection():
    """测试 checker TimeLimitExceeded 检测功能"""
    
    # 创建一个包含 checker TimeLimitExceeded 的模拟响应
    mock_checker_response_with_tle = {
        "tests": [
            {
                "exec_info": {
                    "status": "Failed",
                    "run_result": {
                        "status": "TimeLimitExceeded"
                    }
                }
            }
        ]
    }
    
    # 创建一个正常的 checker 响应
    mock_checker_response_normal = {
        "tests": [
            {
                "exec_info": {
                    "status": "Passed",
                    "run_result": {
                        "status": "Passed"
                    }
                }
            }
        ]
    }
    
    # 创建一个模拟的结果文件（checker 版本）
    mock_result_data = {
        "id": "test_sample",
        "solution_result": [
            {
                "language": "PYTHON",
                "solution": "print('hello')",
                "result": {
                    "accepted": False,
                    "tests": [
                        {
                            "passed": False,
                            "exec_info": {"status": "Success"},
                            "checker_info": mock_checker_response_normal  # 正常的 checker 结果
                        },
                        {
                            "passed": False,
                            "exec_info": {"status": "Success"},
                            "checker_info": mock_checker_response_with_tle  # TLE 的 checker 结果
                        }
                    ]
                }
            }
        ],
        "incorrect_solution_result": [
            {
                "language": "PYTHON",
                "solution": "print('wrong')",
                "result": {
                    "accepted": False,
                    "tests": [
                        {
                            "passed": False,
                            "exec_info": {"status": "Success"},
                            "checker_info": mock_checker_response_with_tle  # TLE 的 checker 结果
                        }
                    ]
                }
            }
        ]
    }
    
    # 创建临时文件并测试
    with tempfile.NamedTemporaryFile(mode='w', suffix='_checker.json', delete=False) as f:
        json.dump(mock_result_data, f, indent=2)
        temp_file_path = f.name
    
    try:
        print(f"Testing has_time_limit_exceeded_in_checker_results function:")
        retry_solutions, retry_incorrect_solutions = has_time_limit_exceeded_in_checker_results(temp_file_path)
        print(f"Solutions to retry: {retry_solutions}")
        print(f"Incorrect solutions to retry: {retry_incorrect_solutions}")
        
        # 验证结果
        # 预期：solution 0 的 test 1 有 TLE，incorrect_solution 0 的 test 0 有 TLE
        expected_retry_solutions = [(0, [1])]  # solution 0 的 test 1
        expected_retry_incorrect_solutions = [(0, [0])]  # incorrect_solution 0 的 test 0
        
        if retry_solutions == expected_retry_solutions and retry_incorrect_solutions == expected_retry_incorrect_solutions:
            print("✅ Test passed! Checker TimeLimitExceeded detection works correctly.")
        else:
            print(f"❌ Test failed!")
            print(f"Expected retry_solutions={expected_retry_solutions}, retry_incorrect_solutions={expected_retry_incorrect_solutions}")
            print(f"Got retry_solutions={retry_solutions}, retry_incorrect_solutions={retry_incorrect_solutions}")
    
    finally:
        # 清理临时文件
        os.unlink(temp_file_path)

def test_empty_checker_info():
    """测试空的 checker_info 情况"""
    
    mock_result_data = {
        "id": "test_sample_empty",
        "solution_result": [
            {
                "language": "PYTHON",
                "solution": "print('hello')",
                "result": {
                    "accepted": False,
                    "tests": [
                        {
                            "passed": False,
                            "exec_info": {"status": "Success"}
                            # 没有 checker_info
                        }
                    ]
                }
            }
        ],
        "incorrect_solution_result": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_checker.json', delete=False) as f:
        json.dump(mock_result_data, f, indent=2)
        temp_file_path = f.name
    
    try:
        print(f"\nTesting empty checker_info case:")
        retry_solutions, retry_incorrect_solutions = has_time_limit_exceeded_in_checker_results(temp_file_path)
        print(f"Solutions to retry: {retry_solutions}")
        print(f"Incorrect solutions to retry: {retry_incorrect_solutions}")
        
        # 预期：没有 checker_info，所以不需要重试
        if retry_solutions == [] and retry_incorrect_solutions == []:
            print("✅ Test passed! Empty checker_info handled correctly.")
        else:
            print("❌ Test failed! Should not retry when no checker_info exists.")
    
    finally:
        os.unlink(temp_file_path)

if __name__ == "__main__":
    test_checker_time_limit_detection()
    test_empty_checker_info()
