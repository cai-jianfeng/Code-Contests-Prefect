#!/usr/bin/env python3
"""
使用修改后的 solutions_eval_original_test.py 的示例说明

修改后的功能说明：
1. 程序会自动检查结果文件夹中已存在的结果文件
2. 对于已存在的结果文件，会检查其中是否包含 TimeLimitExceeded 状态
3. 如果发现 TimeLimitExceeded，会重新测试相应的 solution/incorrect_solution
4. 重新测试的结果会替换原有的结果，而不是追加

主要修改点：
1. 添加了 has_time_limit_exceeded_in_results() 函数来检查结果文件中的 TLE 状态
2. 修改了样本筛选逻辑，不再简单跳过已存在的结果文件
3. 修改了任务队列创建逻辑，支持部分重新测试
4. 修改了结果更新逻辑，支持替换特定索引的结果

使用方法：
直接运行原有的 main 函数即可，程序会自动检测和处理 TimeLimitExceeded 的情况。
"""

def example_usage():
    """展示如何使用修改后的功能"""
    
    print("修改后的代码使用方法：")
    print("1. 直接运行原有的 main 代码")
    print("2. 程序会自动检查 results_path 文件夹中的已有结果")
    print("3. 对于包含 TimeLimitExceeded 的结果，会自动重新测试")
    print("4. 重新测试的结果会更新到原文件中")
    print()
    print("输出示例：")
    print("Found 100 samples to process out of 1000 total.")
    print("New samples: 80")
    print("Samples with TimeLimitExceeded to retry: 20")
    print("  - Solutions to retry: 15")
    print("  - Incorrect solutions to retry: 8") 
    print("Total solutions/incorrect_solutions to process: 245")
    print("Total parallel workers: 512")
    print()
    print("主要改进：")
    print("- ✅ 支持检测和重试 TimeLimitExceeded 的测试用例")
    print("- ✅ 保留已成功测试的结果，只重新测试有问题的部分")
    print("- ✅ 自动替换有问题的结果，不影响其他正常结果")
    print("- ✅ 提供详细的处理统计信息")

if __name__ == "__main__":
    example_usage()
