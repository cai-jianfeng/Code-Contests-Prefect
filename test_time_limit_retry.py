#!/usr/bin/env python3
"""
测试 TimeLimitExceeded 重试功能的脚本
"""

import json
import os
import tempfile
from solutions_eval_original_test import has_time_limit_exceeded_in_results, is_time_limit_exceeded

def test_time_limit_detection():
    """测试 TimeLimitExceeded 检测功能"""
    
    # 创建一个包含 TimeLimitExceeded 的模拟响应
    mock_response_with_tle = {
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
    
    # 创建一个正常的模拟响应
    mock_response_normal = {
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
    
    print("Testing is_time_limit_exceeded function:")
    print(f"Response with TLE: {is_time_limit_exceeded(mock_response_with_tle)}")
    print(f"Normal response: {is_time_limit_exceeded(mock_response_normal)}")
    
    # 创建一个模拟的结果文件
    mock_result_data = {
        "id": "test_sample",
        "solution_result": [
            {
                "language": "PYTHON",
                "solution": "print('hello')",
                "result": mock_response_normal  # 正常结果
            },
            {
                "language": "CPP",
                "solution": "while(true);",
                "result": mock_response_with_tle  # TLE 结果
            }
        ],
        "incorrect_solution_result": [
            {
                "language": "PYTHON",
                "solution": "print('wrong')",
                "result": mock_response_with_tle  # TLE 结果
            }
        ]
    }
    
    # 创建临时文件并测试
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_result_data, f, indent=2)
        temp_file_path = f.name
    
    try:
        print(f"\nTesting has_time_limit_exceeded_in_results function:")
        retry_solutions, retry_incorrect_solutions = has_time_limit_exceeded_in_results(temp_file_path)
        print(f"Solutions to retry (indexes): {retry_solutions}")
        print(f"Incorrect solutions to retry (indexes): {retry_incorrect_solutions}")
        
        # 验证结果
        expected_retry_solutions = [1]  # 第2个solution (索引1) 有TLE
        expected_retry_incorrect_solutions = [0]  # 第1个incorrect_solution (索引0) 有TLE
        
        if retry_solutions == expected_retry_solutions and retry_incorrect_solutions == expected_retry_incorrect_solutions:
            print("✅ Test passed! TimeLimitExceeded detection works correctly.")
        else:
            print(f"❌ Test failed! Expected retry_solutions={expected_retry_solutions}, retry_incorrect_solutions={expected_retry_incorrect_solutions}")
            print(f"Got retry_solutions={retry_solutions}, retry_incorrect_solutions={retry_incorrect_solutions}")
    
    finally:
        # 清理临时文件
        os.unlink(temp_file_path)

if __name__ == "__main__":
    test_time_limit_detection()
