#!/usr/bin/env python3
"""
集成测试脚本 - 验证完整的超时处理和重试流程
"""

import sys
import os
import json
import tempfile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solutions_eval_original_test import has_time_limit_exceeded_in_results

def create_test_result_file(results_data):
    """创建临时测试结果文件"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(results_data, temp_file, indent=2)
    temp_file.close()
    return temp_file.name

def test_complete_timeout_workflow():
    """测试完整的超时处理工作流"""
    
    print("Testing complete timeout handling workflow:")
    
    # 测试案例 1: 包含 sandbox blocked 的结果文件
    sandbox_blocked_results = {
        "solution_result": [
            {
                "result": {
                    "tests": [
                        {
                            "exec_info": {
                                "status": "Failed",
                                "run_result": {
                                    "status": "TimeLimitExceeded",
                                    "stdout": ""  # 无输出，sandbox阻塞
                                }
                            }
                        }
                    ]
                }
            },
            {
                "result": {
                    "tests": [
                        {
                            "exec_info": {
                                "status": "Success",
                                "run_result": {
                                    "status": "Success",
                                    "stdout": "42"
                                }
                            }
                        }
                    ]
                }
            }
        ],
        "incorrect_solution_result": []
    }
    
    # 测试案例 2: 包含真实超时的结果文件
    real_timeout_results = {
        "solution_result": [],
        "incorrect_solution_result": [
            {
                "result": {
                    "tests": [
                        {
                            "exec_info": {
                                "status": "Failed",
                                "run_result": {
                                    "status": "TimeLimitExceeded",
                                    "stdout": "1 2 3 4"  # 有部分输出，真实超时
                                }
                            }
                        }
                    ]
                }
            },
            {
                "result": {
                    "tests": [
                        {
                            "exec_info": {
                                "status": "Success",
                                "run_result": {
                                    "status": "Success",
                                    "stdout": "Result: 100"
                                }
                            }
                        }
                    ]
                }
            }
        ]
    }
    
    # 测试案例 3: 所有测试都成功的结果文件
    all_success_results = {
        "solution_result": [
            {
                "result": {
                    "tests": [
                        {
                            "exec_info": {
                                "status": "Success",
                                "run_result": {
                                    "status": "Success",
                                    "stdout": "Perfect!"
                                }
                            }
                        }
                    ]
                }
            }
        ],
        "incorrect_solution_result": []
    }
    
    try:
        # 创建测试文件
        file1 = create_test_result_file(sandbox_blocked_results)
        file2 = create_test_result_file(real_timeout_results)
        file3 = create_test_result_file(all_success_results)
        
        # 测试 sandbox blocked 检测
        retry_solutions_1, retry_incorrect_1 = has_time_limit_exceeded_in_results(file1)
        print(f"Test 1 (Sandbox blocked file): Solutions: {retry_solutions_1}, Incorrect: {retry_incorrect_1}")
        print(f"Expected: solutions=[0], incorrect=[] (should contain sandbox blocked solution)")
        
        # 测试真实超时检测
        retry_solutions_2, retry_incorrect_2 = has_time_limit_exceeded_in_results(file2)
        print(f"Test 2 (Real timeout file): Solutions: {retry_solutions_2}, Incorrect: {retry_incorrect_2}")
        print(f"Expected: solutions=[], incorrect=[0] (should contain real timeout incorrect solution)")
        
        # 测试无超时情况
        retry_solutions_3, retry_incorrect_3 = has_time_limit_exceeded_in_results(file3)
        print(f"Test 3 (All success file): Solutions: {retry_solutions_3}, Incorrect: {retry_incorrect_3}")
        print(f"Expected: solutions=[], incorrect=[] (no TLE solutions)")
        
        # 验证结果
        success = (retry_solutions_1 == [0] and retry_incorrect_1 == [] and
                  retry_solutions_2 == [] and retry_incorrect_2 == [0] and
                  retry_solutions_3 == [] and retry_incorrect_3 == [])
        
        if success:
            print("✅ Complete timeout workflow tests passed!")
        else:
            print("❌ Some workflow tests failed!")
            
    finally:
        # 清理临时文件
        for temp_file in [file1, file2, file3]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

def test_mixed_timeout_scenarios():
    """测试混合超时场景"""
    
    print("\nTesting mixed timeout scenarios:")
    
    # 混合场景：同时包含 sandbox blocked 和 real timeout
    mixed_results = {
        "solution_result": [
            {
                "result": {
                    "tests": [
                        {
                            "exec_info": {
                                "status": "Failed",
                                "run_result": {
                                    "status": "TimeLimitExceeded",
                                    "stdout": ""  # sandbox blocked
                                }
                            }
                        }
                    ]
                }
            },
            {
                "result": {
                    "tests": [
                        {
                            "exec_info": {
                                "status": "Success",
                                "run_result": {
                                    "status": "Success",
                                    "stdout": "Done!"
                                }
                            }
                        }
                    ]
                }
            }
        ],
        "incorrect_solution_result": [
            {
                "result": {
                    "tests": [
                        {
                            "exec_info": {
                                "status": "Failed",
                                "run_result": {
                                    "status": "TimeLimitExceeded",
                                    "stdout": "Processing... 50%"  # real timeout
                                }
                            }
                        }
                    ]
                }
            },
            {
                "result": {
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
            }
        ]
    }
    
    try:
        mixed_file = create_test_result_file(mixed_results)
        retry_solutions, retry_incorrect = has_time_limit_exceeded_in_results(mixed_file)
        
        print(f"Mixed scenario - Solutions: {retry_solutions}, Incorrect: {retry_incorrect}")
        expected_solutions = [0]  # solution with sandbox blocked
        expected_incorrect = [0, 1]  # incorrect solutions with real timeout and compile timeout
        
        if retry_solutions == expected_solutions and retry_incorrect == expected_incorrect:
            print("✅ Mixed timeout scenario test passed!")
        else:
            print("❌ Mixed timeout scenario test failed!")
            print(f"Expected solutions: {expected_solutions}, incorrect: {expected_incorrect}")
            print(f"Got solutions: {retry_solutions}, incorrect: {retry_incorrect}")
            
    finally:
        if os.path.exists(mixed_file):
            os.unlink(mixed_file)

if __name__ == "__main__":
    test_complete_timeout_workflow()
    test_mixed_timeout_scenarios()
    print("\n🎉 All integration tests completed!")
