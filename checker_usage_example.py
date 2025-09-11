#!/usr/bin/env python3
"""
result_refine_parallel_original_test.py 改进功能说明

改进后的功能说明：
1. 程序会自动检查结果文件夹中已存在的 *_checker.json 结果文件
2. 对于已存在的结果文件，会检查其中的 checker_info 是否包含 TimeLimitExceeded 状态
3. 如果发现 TimeLimitExceeded，会重新测试相应的 test case
4. 重新测试的结果会替换原有的 checker_info，而不是追加

主要修改点：
1. 添加了 has_time_limit_exceeded_in_checker_results() 函数来检查 checker_info 中的 TLE 状态
2. 修改了 filter_existing_results() 函数，支持检测和处理需要重新测试的情况
3. 修改了 parallel_checker_validation() 函数，支持部分重新测试
4. 修改了 apply_checker_results() 函数，支持替换特定 test case 的结果

使用方法：
直接运行原有的 main 函数即可，程序会自动检测和处理 checker_info 中的 TimeLimitExceeded 情况。

输出示例：
"""

def example_usage():
    """展示如何使用改进后的功能"""
    
    print("修改后的代码使用方法：")
    print("1. 直接运行原有的 main 代码")
    print("2. 程序会自动检查 result_folder 中的 *_checker.json 文件")
    print("3. 对于包含 checker_info TimeLimitExceeded 的结果，会自动重新测试")
    print("4. 重新测试的结果会更新到原文件中")
    print()
    print("输出示例：")
    print("Filtering already processed results and checking for TimeLimitExceeded...")
    print("Found TimeLimitExceeded in /path/to/result_checker.json, will retry affected test cases.")
    print("Found 50 samples to process out of 1000 total.")
    print("New samples: 30")
    print("Samples with TimeLimitExceeded to retry: 20")
    print("  - Total test cases to retry: 35")
    print("Total test cases to validate: 245")
    print("Using 128 API endpoints with 1 workers per endpoint")
    print()
    print("主要改进：")
    print("- ✅ 支持检测和重试 checker_info 中 TimeLimitExceeded 的测试用例")
    print("- ✅ 保留已成功检测的 checker 结果，只重新测试有问题的部分")
    print("- ✅ 自动替换有问题的 checker_info，不影响其他正常结果")
    print("- ✅ 提供详细的处理统计信息，包括重试的 test case 数量")
    print("- ✅ 支持 test case 级别的精确重试，提高效率")
    print()
    print("检测逻辑：")
    print("- 检查每个 test 的 checker_info 中是否有 TimeLimitExceeded")
    print("- 返回格式：[(solution_idx, [test_indices]), ...]")
    print("- 只重新测试有问题的特定 test case，而不是整个 solution")

if __name__ == "__main__":
    example_usage()
