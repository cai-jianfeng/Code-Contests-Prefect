#!/usr/bin/env python3
"""
智能超时处理系统完整演示
展示从结果分析到智能重试的完整流程
"""

import sys
import os
import json
import tempfile
import shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solutions_eval_plus_test import (
    calculate_timeout_multiplier,
    has_time_limit_exceeded_in_results,
    is_time_limit_exceeded
)

def create_comprehensive_test_scenario():
    """创建综合测试场景"""
    
    print("🏗️ 创建综合测试场景")
    print("=" * 40)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"使用临时目录: {temp_dir}")
    
    # 模拟各种超时情况的结果文件
    comprehensive_result = {
        "solution_result": [
            {
                "language": "PYTHON",
                "solution": "for i in range(1000): print(i)",
                "result": {
                    "tests": [{
                        "exec_info": {
                            "status": "Failed",
                            "run_result": {
                                "status": "TimeLimitExceeded",
                                "stdout": "0\n1\n2\n3\n4\n5\n"  # 部分输出，应该重试
                            }
                        }
                    }]
                }
            },
            {
                "language": "CPP",
                "solution": "int main() { while(1); return 0; }",
                "result": {
                    "tests": [{
                        "exec_info": {
                            "status": "Failed",
                            "run_result": {
                                "status": "TimeLimitExceeded",
                                "stdout": ""  # 无输出，可能是死循环，不重试
                            }
                        }
                    }]
                }
            },
            {
                "language": "JAVA",
                "solution": "public class Main { }",
                "result": {
                    "tests": [{
                        "exec_info": {
                            "status": "Failed",
                            "compile_result": {
                                "status": "TimeLimitExceeded"  # 编译超时，sandbox问题，应该重试
                            }
                        }
                    }]
                }
            },
            {
                "language": "PYTHON",
                "solution": "print('success')",
                "result": {
                    "tests": [{
                        "exec_info": {
                            "status": "Success",
                            "run_result": {
                                "status": "Success",
                                "stdout": "success"  # 正常成功，不需要重试
                            }
                        }
                    }]
                }
            }
        ],
        "incorrect_solution_result": [
            {
                "language": "PYTHON", 
                "solution": "import time; [print(f'processing {i}') for i in range(100)]",
                "result": {
                    "tests": [{
                        "exec_info": {
                            "status": "Failed",
                            "run_result": {
                                "status": "TimeLimitExceeded",
                                "stdout": "processing 0\nprocessing 1\n" + "processing 2\nprocessing 3\n" * 50  # 长输出，需要更多时间
                            }
                        }
                    }]
                }
            },
            {
                "language": "CPP",
                "solution": "int main() { printf(\"wrong\"); return 0; }",
                "result": {
                    "tests": [{
                        "exec_info": {
                            "status": "Failed",
                            "run_result": {
                                "status": "TimeLimitExceeded",
                                "stdout": "wrong answer format"  # 有输出但格式错误，还是应该给机会重试
                            }
                        }
                    }]
                }
            }
        ]
    }
    
    # 保存结果文件
    result_file = os.path.join(temp_dir, "comprehensive_test.json")
    with open(result_file, 'w') as f:
        json.dump(comprehensive_result, f, indent=2)
    
    return temp_dir, result_file, comprehensive_result

def analyze_and_demonstrate(temp_dir, result_file, original_data):
    """分析结果并演示智能处理"""
    
    print("\n🔍 智能分析结果文件")
    print("=" * 40)
    
    # 执行智能分析
    retry_solutions, retry_incorrect, timeout_configs = has_time_limit_exceeded_in_results(result_file)
    
    print(f"📊 分析结果:")
    print(f"  需要重试的 solutions: {retry_solutions}")
    print(f"  需要重试的 incorrect_solutions: {retry_incorrect}")
    print(f"  生成的超时配置: {len(timeout_configs)} 项")
    
    # 详细解释每个决策
    print(f"\n📋 详细决策解释:")
    
    solution_results = original_data["solution_result"]
    for idx, result in enumerate(solution_results):
        language = result["language"]
        tle_type = is_time_limit_exceeded(result["result"])
        
        print(f"\n  Solution {idx} ({language}):")
        if tle_type:
            if idx in retry_solutions:
                reason = "有输出的real_timeout" if tle_type == "real_timeout" else "sandbox_blocked"
                config_key = f'solution_{idx}'
                timeout_info = timeout_configs.get(config_key, "使用默认超时")
                print(f"    决策: 🔄 重试 (原因: {reason})")
                print(f"    超时配置: {timeout_info}")
            else:
                print(f"    决策: ⏭️ 跳过 (原因: 无输出的{tle_type}，可能是逻辑问题)")
        else:
            print(f"    决策: ✅ 成功 (无需重试)")
    
    incorrect_results = original_data["incorrect_solution_result"]
    for idx, result in enumerate(incorrect_results):
        language = result["language"]
        tle_type = is_time_limit_exceeded(result["result"])
        
        print(f"\n  Incorrect Solution {idx} ({language}):")
        if tle_type:
            if idx in retry_incorrect:
                config_key = f'incorrect_solution_{idx}'
                timeout_info = timeout_configs.get(config_key, "使用默认超时")
                print(f"    决策: 🔄 重试 (原因: 有输出的{tle_type})")
                print(f"    超时配置: {timeout_info}")
            else:
                print(f"    决策: ⏭️ 跳过 (原因: 无输出的{tle_type})")
        else:
            print(f"    决策: ✅ 成功 (无需重试)")

def demonstrate_runtime_intelligence():
    """演示运行时智能处理"""
    
    print(f"\n⚡ 运行时智能处理演示")
    print(f"=" * 40)
    
    scenarios = [
        {
            "name": "场景1: 部分完成的计算任务",
            "actual_output": "计算完成: 1/100\n计算完成: 2/100\n计算完成: 3/100\n",
            "expected_output": "计算完成: 1/100\n计算完成: 2/100\n...\n计算完成: 100/100\n完成",
            "description": "程序正在正常执行，但需要更多时间"
        },
        {
            "name": "场景2: 错误的输出格式",
            "actual_output": "Error: Invalid input format",
            "expected_output": "Result: 42",
            "description": "程序产生了输出，但格式不正确"
        },
        {
            "name": "场景3: 接近完成的任务",
            "actual_output": "Processing data... 95% complete",
            "expected_output": "Processing data... 95% complete\nProcessing data... 100% complete\nDone",
            "description": "程序几乎完成，只需要少量额外时间"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  描述: {scenario['description']}")
        print(f"  实际输出: '{scenario['actual_output'][:50]}{'...' if len(scenario['actual_output']) > 50 else ''}'")
        print(f"  期望输出: '{scenario['expected_output'][:50]}{'...' if len(scenario['expected_output']) > 50 else ''}'")
        
        is_partial, multiplier = calculate_timeout_multiplier(
            scenario['actual_output'], 
            scenario['expected_output']
        )
        
        if is_partial:
            print(f"  🎯 智能决策: 增加 {multiplier}x 超时时间重试")
            print(f"  💡 原因: 输出显示程序正在正常执行，需要更多时间")
        else:
            print(f"  🛑 智能决策: 不重试")
            print(f"  💡 原因: 输出格式不匹配，可能是程序逻辑问题")

def show_performance_comparison():
    """展示性能对比"""
    
    print(f"\n📈 性能对比")
    print(f"=" * 30)
    
    print(f"🔄 原有方案:")
    print(f"  • 固定提升到1000s超时")
    print(f"  • 所有TimeLimitExceeded都重试")
    print(f"  • 无法区分问题类型")
    print(f"  • 浪费时间在无效重试上")
    
    print(f"\n⚡ 智能方案:")
    print(f"  • 根据输出分析动态调整超时(2x-10x)")
    print(f"  • 只重试有价值的情况")
    print(f"  • 区分程序问题vs环境问题")
    print(f"  • 预设优化配置，避免重复分析")
    
    print(f"\n💰 效益分析:")
    print(f"  • 时间节省: 避免无效的长时间等待")
    print(f"  • 资源优化: 减少不必要的重试占用")
    print(f"  • 成功率提升: 更精准的超时时间调整")
    print(f"  • 用户体验: 更快的响应和更高的成功率")

if __name__ == "__main__":
    try:
        # 创建测试场景
        temp_dir, result_file, original_data = create_comprehensive_test_scenario()
        
        # 分析和演示
        analyze_and_demonstrate(temp_dir, result_file, original_data)
        
        # 演示运行时智能处理
        demonstrate_runtime_intelligence()
        
        # 性能对比
        show_performance_comparison()
        
        print(f"\n🎉 智能超时处理系统演示完成")
        print(f"✨ 系统现在能够:")
        print(f"   • 智能分析超时原因")
        print(f"   • 精确调整超时时间")  
        print(f"   • 避免无效重试")
        print(f"   • 预设优化配置")
        
    finally:
        # 清理临时文件
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
            print(f"\n🧹 清理完成")
