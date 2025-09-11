#!/usr/bin/env python3
"""
测试不同API类型的TimeLimitExceeded处理
"""

from corner_case_gen_parallel import SandboxClient

def test_api_type_detection():
    """测试不同API类型的超时检测"""
    print("Testing API type-specific timeout detection...")
    
    client = SandboxClient()
    
    # 测试submit API的响应格式
    submit_response = {
        "tests": [{
            "exec_info": {
                "status": "Failed",
                "run_result": {
                    "status": "TimeLimitExceeded",
                    "stdout": "Partial output"
                }
            },
            "test_info": {
                "output": {"stdout": "Expected full output"}
            }
        }]
    }
    
    # 测试run_code API的响应格式
    run_code_response = {
        "status": "Failed",
        "compile_result": {
            "status": "Success",
            "stdout": "",
            "stderr": ""
        },
        "run_result": {
            "status": "TimeLimitExceeded",
            "stdout": "Some output",
            "stderr": ""
        }
    }
    
    # 测试run_code API无输出的情况
    run_code_blocked_response = {
        "status": "Failed",
        "compile_result": {
            "status": "Success"
        },
        "run_result": {
            "status": "TimeLimitExceeded",
            "stdout": "",
            "stderr": ""
        }
    }
    
    # 测试submit API检测
    result = client._detect_timeout_by_api_type(submit_response, True, False)
    print(f"✓ Submit API timeout detection: {result}")
    
    # 测试run_code API检测
    result = client._detect_timeout_by_api_type(run_code_response, False, True)
    print(f"✓ Run code API timeout detection: {result}")
    
    # 测试run_code API阻塞检测
    result = client._detect_timeout_by_api_type(run_code_blocked_response, False, True)
    print(f"✓ Run code API blocked detection: {result}")
    
    print("All API type detection tests passed!")

def test_timeout_analysis():
    """测试超时分析功能"""
    print("Testing timeout analysis...")
    
    client = SandboxClient()
    
    # 模拟submit API的json_data
    submit_json_data = {
        "config": {
            "provided_data": {
                "test": [
                    {"output": {"stdout": "Expected full output here"}}
                ]
            }
        }
    }
    
    submit_response = {
        "tests": [{
            "exec_info": {
                "run_result": {
                    "status": "TimeLimitExceeded",
                    "stdout": "Expected full"
                }
            }
        }]
    }
    
    # 测试submit API的超时分析
    should_retry, config = client._analyze_timeout_and_adjust(
        submit_response, submit_json_data, 20, 20, True, False, 1000
    )
    
    print(f"✓ Submit API timeout analysis: should_retry={should_retry}, config={config}")
    
    # 模拟run_code API
    run_code_response = {
        "run_result": {
            "status": "TimeLimitExceeded",
            "stdout": "Some output"
        }
    }
    
    should_retry, config = client._analyze_timeout_and_adjust(
        run_code_response, {}, 20, 20, False, True, 1000
    )
    
    print(f"✓ Run code API timeout analysis: should_retry={should_retry}, config={config}")
    
    print("All timeout analysis tests passed!")

if __name__ == "__main__":
    print("Testing enhanced SandboxClient...")
    
    try:
        test_api_type_detection()
        test_timeout_analysis()
        
        print("\n🎉 All enhanced SandboxClient tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
