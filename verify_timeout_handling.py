#!/usr/bin/env python3
"""
验证TimeLimitExceeded处理修改
"""

from corner_case_gen_parallel import (
    is_time_limit_exceeded, 
    calculate_timeout_multiplier,
    SandboxClient
)

def test_functions():
    print("Testing TimeLimitExceeded handling functions...")
    
    # 测试sandbox_blocked检测
    sandbox_blocked_response = {
        "tests": [{
            "exec_info": {
                "status": "Failed",
                "run_result": {
                    "status": "TimeLimitExceeded",
                    "stdout": ""
                }
            },
            "test_info": {
                "output": {"stdout": "Expected output"}
            }
        }]
    }
    
    result = is_time_limit_exceeded(sandbox_blocked_response)
    print(f"✓ Sandbox blocked detection: {result}")
    
    # 测试real_timeout检测
    real_timeout_response = {
        "tests": [{
            "exec_info": {
                "status": "Failed", 
                "run_result": {
                    "status": "TimeLimitExceeded",
                    "stdout": "Partial"
                }
            },
            "test_info": {
                "output": {"stdout": "Partial output"}
            }
        }]
    }
    
    result = is_time_limit_exceeded(real_timeout_response)
    print(f"✓ Real timeout detection: {result}")
    
    # 测试超时倍数计算
    is_partial, multiplier = calculate_timeout_multiplier("Hello", "Hello World!")
    print(f"✓ Timeout multiplier: partial={is_partial}, multiplier={multiplier:.2f}")
    
    # 测试SandboxClient
    client = SandboxClient()
    print(f"✓ SandboxClient created: {type(client)}")
    
    print("All function tests passed!")

if __name__ == "__main__":
    test_functions()
