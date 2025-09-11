#!/usr/bin/env python3
"""
超时处理系统综合演示脚本
展示新的 TimeLimitExceeded 分类和处理机制
"""

import sys
import os
import json
import tempfile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solutions_eval_original_test import (
    is_time_limit_exceeded, 
    has_time_limit_exceeded_in_results
)

def create_demo_scenarios():
    """创建演示场景"""
    
    print("🚀 TimeLimitExceeded 处理系统演示")
    print("=" * 60)
    
    # 场景1: Sandbox 内部阻塞
    print("\n📌 场景1: Sandbox 内部阻塞 (无输出)")
    sandbox_blocked = {
        "tests": [{
            "exec_info": {
                "status": "Failed",
                "run_result": {
                    "status": "TimeLimitExceeded",
                    "stdout": ""
                }
            }
        }]
    }
    
    result1 = is_time_limit_exceeded(sandbox_blocked)
    print(f"   检测结果: {result1}")
    print(f"   处理策略: 正常重试，保持原始超时配置 (20s)")
    
    # 场景2: 代码真实超时
    print("\n📌 场景2: 代码真实超时 (有部分输出)")
    real_timeout = {
        "tests": [{
            "exec_info": {
                "status": "Failed", 
                "run_result": {
                    "status": "TimeLimitExceeded",
                    "stdout": "计算中... 已处理 1000 条数据"
                }
            }
        }]
    }
    
    result2 = is_time_limit_exceeded(real_timeout)
    print(f"   检测结果: {result2}")
    print(f"   处理策略: 提高超时配置到 1000s 后重试")
    
    # 场景3: 编译超时
    print("\n📌 场景3: 编译超时")
    compile_timeout = {
        "tests": [{
            "exec_info": {
                "status": "Failed",
                "compile_result": {
                    "status": "TimeLimitExceeded"
                },
                "run_result": None
            }
        }]
    }
    
    result3 = is_time_limit_exceeded(compile_timeout)
    print(f"   检测结果: {result3}")
    print(f"   处理策略: 正常重试，保持原始超时配置")

def simulate_retry_workflow():
    """模拟完整的重试工作流"""
    
    print("\n🔄 重试工作流模拟")
    print("=" * 40)
    
    # 创建包含不同超时类型的结果文件
    demo_results = {
        "solution_result": [
            {
                "result": {
                    "tests": [{
                        "exec_info": {
                            "status": "Failed",
                            "run_result": {
                                "status": "TimeLimitExceeded",
                                "stdout": ""  # sandbox blocked
                            }
                        }
                    }]
                }
            },
            {
                "result": {
                    "tests": [{
                        "exec_info": {
                            "status": "Failed",
                            "run_result": {
                                "status": "TimeLimitExceeded", 
                                "stdout": "正在计算第 500 步..."  # real timeout
                            }
                        }
                    }]
                }
            },
            {
                "result": {
                    "tests": [{
                        "exec_info": {
                            "status": "Success",
                            "run_result": {
                                "status": "Success",
                                "stdout": "42"
                            }
                        }
                    }]
                }
            }
        ],
        "incorrect_solution_result": [
            {
                "result": {
                    "tests": [{
                        "exec_info": {
                            "status": "Failed",
                            "compile_result": {
                                "status": "TimeLimitExceeded"
                            },
                            "run_result": None
                        }
                    }]
                }
            }
        ]
    }
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(demo_results, temp_file, indent=2)
    temp_file.close()
    
    try:
        # 分析结果文件
        retry_solutions, retry_incorrect = has_time_limit_exceeded_in_results(temp_file.name)
        
        print(f"📄 分析结果文件:")
        print(f"   - 需要重试的 solutions: {retry_solutions}")
        print(f"   - 需要重试的 incorrect_solutions: {retry_incorrect}")
        
        print(f"\n📋 具体处理计划:")
        
        # 分析每个需要重试的solution
        for idx in retry_solutions:
            solution_data = demo_results["solution_result"][idx]["result"]
            tle_type = is_time_limit_exceeded(solution_data)
            
            if tle_type == "sandbox_blocked":
                print(f"   - Solution {idx}: Sandbox阻塞 → 正常重试 (20s)")
            elif tle_type == "real_timeout":
                print(f"   - Solution {idx}: 真实超时 → 提高超时重试 (1000s)")
        
        # 分析每个需要重试的incorrect_solution  
        for idx in retry_incorrect:
            incorrect_data = demo_results["incorrect_solution_result"][idx]["result"]
            tle_type = is_time_limit_exceeded(incorrect_data)
            
            if tle_type == "sandbox_blocked":
                print(f"   - Incorrect Solution {idx}: Sandbox阻塞 → 正常重试 (20s)")
            elif tle_type == "real_timeout":
                print(f"   - Incorrect Solution {idx}: 真实超时 → 提高超时重试 (1000s)")
                
    finally:
        os.unlink(temp_file.name)

def demonstrate_load_balancing():
    """演示负载均衡优化"""
    
    print("\n⚖️ 负载均衡优化")
    print("=" * 30)
    print("🔀 队列随机化功能:")
    print("   - 防止重试任务在队列末尾循环")
    print("   - 确保其他API可以在中间获取任务")
    print("   - 提高整体并行处理效率")
    
    print("\n🔒 线程安全保障:")
    print("   - task_queue_lock 保护队列操作")
    print("   - shuffle_queue_safely 安全打乱队列")
    print("   - 避免并发访问冲突")

def show_configuration_strategy():
    """展示配置策略"""
    
    print("\n⚙️ 超时配置策略")
    print("=" * 30)
    
    scenarios = [
        ("Sandbox 阻塞", "sandbox_blocked", "20s", "可能是sandbox内部问题，保持原配置重试"),
        ("真实超时", "real_timeout", "20s → 1000s", "代码需要更多时间，大幅提高超时限制"),
        ("编译超时", "compile timeout", "20s", "编译环境问题，保持原配置重试")
    ]
    
    for scenario, tle_type, timeout_change, reason in scenarios:
        print(f"📊 {scenario}:")
        print(f"   类型: {tle_type}")
        print(f"   超时调整: {timeout_change}")
        print(f"   原因: {reason}\n")

if __name__ == "__main__":
    create_demo_scenarios()
    simulate_retry_workflow() 
    demonstrate_load_balancing()
    show_configuration_strategy()
    
    print("\n✨ 总结")
    print("=" * 20)
    print("🎯 新系统特点:")
    print("   ✅ 智能超时类型识别")
    print("   ✅ 差异化重试策略") 
    print("   ✅ 负载均衡优化")
    print("   ✅ 线程安全保障")
    print("   ✅ 全面统计支持")
    print("\n🚀 系统已经准备就绪，可以投入生产使用！")
