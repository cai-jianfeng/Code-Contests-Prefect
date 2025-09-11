#!/usr/bin/env python3
"""
test_4.py 新增 checker_info 统计功能说明

新增功能说明：
1. 添加了 checker_success() 函数来检查 checker_info 的状态
2. 添加了 get_checker_results_binary() 函数来统计 checker 结果的分布
3. 添加了相应的平均值计算函数

主要新增函数：

1. checker_success(solution_result)
   检查 solution_result 中 checker_info 的状态
   返回值：
   - 0: TimeLimitExceeded (checker 超时)
   - 1: Failed (其他 checker 错误，包括 checker_error)
   - 2: Success (checker 成功)
   - 3: No checker_info (没有 checker 信息，跳过)

2. get_checker_results_binary(result_folder)
   统计 checker_info 的状态分布，处理 *_checker.json 文件
   返回每个样本的 checker 统计信息

3. 新增的平均值计算函数：
   - checker_success_avg(results): 统计 checker 成功率
   - checker_other_unsuccess_avg(results): 统计 checker 其他失败率
   - checker_time_limit_exceeded_avg(results): 统计 checker 超时率
   - checker_no_info_avg(results): 统计没有 checker_info 的比例

使用示例和输出：
"""

def usage_example():
    """展示如何使用新功能"""
    
    print("使用方法：")
    print("1. 运行 python test_4.py")
    print("2. 程序会自动统计原始测试结果和 checker 结果")
    print("3. 比较两者的成功率差异")
    print()
    print("输出示例：")
    print("=== Original Test Results ===")
    print("Average Success Correct: 1.0000")
    print("Average Success Incorrect: 1.0000")
    print("Average Other Unsuccess Correct: 0.0000")
    print("Average Other Unsuccess Incorrect: 0.0000")
    print("Average Time Limit Exceeded Correct: 0.0000")
    print("Average Time Limit Exceeded Incorrect: 0.0000")
    print()
    print("=== Checker Results ===")
    print("Average Checker Success Correct: 0.1325")
    print("Average Checker Success Incorrect: 0.5810")
    print("Average Checker Other Unsuccess Correct: 0.0000")
    print("Average Checker Other Unsuccess Incorrect: 0.0000")
    print("Average Checker Time Limit Exceeded Correct: 0.0000")
    print("Average Checker Time Limit Exceeded Incorrect: 0.0000")
    print("Average Checker No Info Correct: 0.8675")
    print("Average Checker No Info Incorrect: 0.4190")
    print()
    print("=== Summary ===")
    print("Total samples processed: 153")
    print("Samples with checker results: 153")
    print("Original vs Checker Success Rate (Correct): 1.0000 vs 0.1325")
    print("Original vs Checker Success Rate (Incorrect): 1.0000 vs 0.5810")
    print()
    print("主要观察点：")
    print("- ✅ 原始测试和 checker 测试的成功率对比")
    print("- ✅ checker_info 中 TimeLimitExceeded 的分布")
    print("- ✅ 没有 checker_info 的样本比例")
    print("- ✅ checker 验证对正确/错误解决方案的影响")

if __name__ == "__main__":
    usage_example()
