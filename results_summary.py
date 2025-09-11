# %% setup
import json
import os
from tqdm import tqdm

TIME_LIMIT_EXCEEDED_STDOUT_ORIGINAL_NUM = 0
TIME_LIMIT_EXCEEDED_STDOUT_PLUS_NUM = 0
TIME_LIMIT_EXCEEDED_STDOUT_PLUS_NUM_FALSE = 0
SAMPLES_NUM = 0
TEST_FAILED = set()
TESTS = set()

def solution_success(solution_result):
    tests = solution_result['result'].get("tests", [])
    for test in tests:
        global TESTS
        TESTS.add(test['test_info']['input']['stdin'])
        if test['exec_info'].get('status') == "Failed":
            global TEST_FAILED
            TEST_FAILED.add(test['test_info']['input']['stdin'])
            compile_result = test['exec_info'].get('compile_result')
            run_result = test['exec_info'].get('run_result')
            if (isinstance(compile_result, dict) and compile_result.get('status') == "TimeLimitExceeded") or \
                (isinstance(run_result, dict) and run_result.get('status') == "TimeLimitExceeded"):
                if isinstance(run_result, dict) and run_result['stdout']:
                    if run_result['stdout'] not in test['test_info']['output']['stdout']:
                        global TIME_LIMIT_EXCEEDED_STDOUT_PLUS_NUM_FALSE
                        TIME_LIMIT_EXCEEDED_STDOUT_PLUS_NUM_FALSE += 1
                        return -1.1
                    elif run_result['execution_time'] <= 100:
                        global TIME_LIMIT_EXCEEDED_STDOUT_ORIGINAL_NUM
                        TIME_LIMIT_EXCEEDED_STDOUT_ORIGINAL_NUM += 1
                        return -1.2
                    else:
                        global TIME_LIMIT_EXCEEDED_STDOUT_PLUS_NUM
                        TIME_LIMIT_EXCEEDED_STDOUT_PLUS_NUM += 1
                        return -1.2
                        
                return 0
            elif isinstance(run_result, dict) and run_result.get('return_code') == -11:
                return -2
            else:
                return 1
    return 2

def checker_success(solution_result):
    """
    检查 solution_result 中 checker_info 的状态
    返回值：
    0 - TimeLimitExceeded
    1 - Failed (其他错误)
    2 - Success
    3 - No checker_info (跳过)
    """
    tests = solution_result['result'].get("tests", [])
    has_checker_info = False
    has_checker_error = False
    
    for test in tests:
        # 首先检查是否有 checker_error
        checker_error = test.get('checker_error')
        if checker_error is not None:
            has_checker_error = True
            return 1  # Failed
        
        checker_info = test.get('checker_info')
        if checker_info is None:
            continue
            
        has_checker_info = True
        
        # 检查 checker_info 中的 tests
        if isinstance(checker_info, dict):
            checker_tests = checker_info.get("tests", [])
            for checker_test in checker_tests:
                if checker_test.get('exec_info', {}).get('status') == "Failed":
                    compile_result = checker_test.get('exec_info', {}).get('compile_result')
                    run_result = checker_test.get('exec_info', {}).get('run_result')
                    if (isinstance(compile_result, dict) and compile_result.get('status') == "TimeLimitExceeded") or \
                       (isinstance(run_result, dict) and run_result.get('status') == "TimeLimitExceeded"):
                        return 0  # TimeLimitExceeded
                    else:
                        return 1  # Other failure
    
    if has_checker_error:
        return 1  # Failed
    elif not has_checker_info:
        return 3  # No checker_info
    else:
        return 2  # Success

def get_original_results_binary(result_folder, result_checker=False):
    if result_checker:
        result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" in file and "result_list" not in file]
    else:
        result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" not in file and "result_list" not in file]
    result_files = sorted(result_files)
    results_checker = []

    for result_file in tqdm(result_files):
        if result_file.endswith(".json"):
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                # try:
                data = json.load(file)
                solution_results = data.get("solution_result", [])
                success = 0
                other_unsuccess = 0
                time_limit_exceeded = 0
                time_limit_exceeded_stdout = 0
                time_limit_exceeded_stdout_false = 0
                seg_fault = 0
                total_solutions = len(solution_results)
                success_sample_level = True
                for result in solution_results:
                    status = solution_success(result)
                    if status == 2:
                        success += 1
                    elif status == 1:
                        other_unsuccess += 1
                    elif status == 0:
                        time_limit_exceeded += 1
                    elif status == -1.1:
                        time_limit_exceeded_stdout_false += 1
                    elif status == -1.2:
                        time_limit_exceeded_stdout += 1
                    elif status == -2:
                        seg_fault += 1

                    if status != 2:
                        success_sample_level = False
                
                success_correct = success / total_solutions if total_solutions > 0 else 0
                other_unsuccess_rate = other_unsuccess / total_solutions if total_solutions > 0 else 0
                time_limit_exceeded_rate = time_limit_exceeded / total_solutions if total_solutions > 0 else 0
                time_limit_exceeded_stdout_false_rate = time_limit_exceeded_stdout_false / total_solutions if total_solutions > 0 else 0
                time_limit_exceeded_stdout_rate = time_limit_exceeded_stdout / total_solutions if total_solutions > 0 else 0
                seg_fault_rate = seg_fault / total_solutions if total_solutions > 0 else 0
                incorrect_solutions_results = data.get("incorrect_solution_result", [])
                success = 0
                other_unsuccess = 0
                time_limit_exceeded = 0
                time_limit_exceeded_stdout = 0
                time_limit_exceeded_stdout_false = 0
                seg_fault = 0
                total_incorrect_solutions = len(incorrect_solutions_results)
                for incorrect_result in incorrect_solutions_results:
                    status = solution_success(incorrect_result)
                    if status == 2:
                        success += 1
                    elif status == 1:
                        other_unsuccess += 1
                    elif status == 0:
                        time_limit_exceeded += 1
                    elif status == -1.1:
                        time_limit_exceeded_stdout_false += 1
                    elif status == -1.2:
                        time_limit_exceeded_stdout += 1
                    elif status == -2:
                        seg_fault += 1

                    if status != 2:
                        success_sample_level = False

                global SAMPLES_NUM
                SAMPLES_NUM += 1 if success_sample_level else 0

                success_incorrect = success / total_incorrect_solutions if total_incorrect_solutions else 0
                other_unsuccess_incorrect = other_unsuccess / total_incorrect_solutions if total_incorrect_solutions else 0
                time_limit_exceeded_incorrect = time_limit_exceeded / total_incorrect_solutions if total_incorrect_solutions else 0
                time_limit_exceeded_incorrect_stdout = time_limit_exceeded_stdout / total_incorrect_solutions if total_incorrect_solutions else 0
                time_limit_exceeded_incorrect_stdout_false = time_limit_exceeded_stdout_false / total_incorrect_solutions if total_incorrect_solutions else 0
                seg_fault_incorrect = seg_fault / total_incorrect_solutions if total_incorrect_solutions else 0
                results_checker.append({
                    "id": data.get("id", ""),
                    "Correct": {
                        "Success Correct": success_correct,
                        "Other Unsuccess Rate": other_unsuccess_rate,
                        "Time Limit Exceeded Rate": time_limit_exceeded_rate,
                        "Time Limit Exceeded Stdout Rate": time_limit_exceeded_stdout_rate,
                        "Time Limit Exceeded Stdout False Rate": time_limit_exceeded_stdout_false_rate,
                        "Seg Fault Rate": seg_fault_rate
                    },
                    "Incorrect": {
                        "Success Incorrect": success_incorrect,
                        "Other Unsuccess Incorrect": other_unsuccess_incorrect,
                        "Time Limit Exceeded Incorrect": time_limit_exceeded_incorrect,
                        "Time Limit Exceeded Incorrect Stdout": time_limit_exceeded_incorrect_stdout,
                        "Time Limit Exceeded Incorrect Stdout False": time_limit_exceeded_incorrect_stdout_false,
                        "Seg Fault Incorrect": seg_fault_incorrect
                    }
                })

                # except json.JSONDecodeError as e:
                #     print(f"Error decoding JSON from {result_file}: {e}")
                # except Exception as e:
                #     print(f"An error occurred while processing {result_file}: {e}")
        else:
            print(f"Skipping non-JSON file: {result_file}")

    return results_checker

def get_checker_results_binary(result_folder):
    """
    统计 checker_info 的状态分布
    """
    result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" in file]
    result_files = sorted(result_files)
    results_checker = []

    for result_file in tqdm(result_files, desc="Processing checker results"):
        if result_file.endswith(".json"):
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    solution_results = data.get("solution_result", [])
                    success = 0
                    other_unsuccess = 0
                    time_limit_exceeded = 0
                    no_checker_info = 0
                    total_solutions = len(solution_results)
                    
                    for result in solution_results:
                        status = checker_success(result)
                        if status == 2:
                            success += 1
                        elif status == 1:
                            other_unsuccess += 1
                        elif status == 0:
                            time_limit_exceeded += 1
                        elif status == 3:
                            no_checker_info += 1
                    
                    # 计算正确解决方案的 checker 统计
                    success_correct = success / total_solutions if total_solutions > 0 else 0
                    other_unsuccess_rate = other_unsuccess / total_solutions if total_solutions > 0 else 0
                    time_limit_exceeded_rate = time_limit_exceeded / total_solutions if total_solutions > 0 else 0
                    no_checker_info_rate = no_checker_info / total_solutions if total_solutions > 0 else 0
                    
                    # 处理错误解决方案
                    incorrect_solutions_results = data.get("incorrect_solution_result", [])
                    success = 0
                    other_unsuccess = 0
                    time_limit_exceeded = 0
                    no_checker_info = 0
                    total_incorrect_solutions = len(incorrect_solutions_results)
                    
                    for incorrect_result in incorrect_solutions_results:
                        status = checker_success(incorrect_result)
                        if status == 2:
                            success += 1
                        elif status == 1:
                            other_unsuccess += 1
                        elif status == 0:
                            time_limit_exceeded += 1
                        elif status == 3:
                            no_checker_info += 1

                    success_incorrect = success / total_incorrect_solutions if total_incorrect_solutions else 0
                    other_unsuccess_incorrect = other_unsuccess / total_incorrect_solutions if total_incorrect_solutions else 0
                    time_limit_exceeded_incorrect = time_limit_exceeded / total_incorrect_solutions if total_incorrect_solutions else 0
                    no_checker_info_incorrect = no_checker_info / total_incorrect_solutions if total_incorrect_solutions else 0
                    
                    results_checker.append({
                        "id": data.get("id", ""),
                        "Correct": {
                            "Checker Success Correct": success_correct,
                            "Checker Other Unsuccess Rate": other_unsuccess_rate,
                            "Checker Time Limit Exceeded Rate": time_limit_exceeded_rate,
                            "Checker No Info Rate": no_checker_info_rate
                        },
                        "Incorrect": {
                            "Checker Success Incorrect": success_incorrect,
                            "Checker Other Unsuccess Incorrect": other_unsuccess_incorrect,
                            "Checker Time Limit Exceeded Incorrect": time_limit_exceeded_incorrect,
                            "Checker No Info Incorrect": no_checker_info_incorrect
                        }
                    })

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {result_file}: {e}")
                except Exception as e:
                    print(f"An error occurred while processing {result_file}: {e}")
        else:
            print(f"Skipping non-JSON file: {result_file}")

    return results_checker

def success_avg(results):
    success_correct = [sample['Correct']['Success Correct'] for sample in results]
    success_incorrect = [sample['Incorrect']['Success Incorrect'] for sample in results]
    success_correct_avg = sum(success_correct) / len(success_correct) if success_correct else 0
    success_incorrect_avg = sum(success_incorrect) / len(success_incorrect) if success_incorrect else 0
    print(f"Average Success Correct: {success_correct_avg:.8f}")
    print(f"Average Success Incorrect: {success_incorrect_avg:.8f}")
    return success_correct_avg, success_incorrect_avg

def other_unsuccess_avg(results):
    other_unsuccess_correct = [sample['Correct']['Other Unsuccess Rate'] for sample in results]
    other_unsuccess_incorrect = [sample['Incorrect']['Other Unsuccess Incorrect'] for sample in results]
    other_unsuccess_correct_avg = sum(other_unsuccess_correct) / len(other_unsuccess_correct) if other_unsuccess_correct else 0
    other_unsuccess_incorrect_avg = sum(other_unsuccess_incorrect) / len(other_unsuccess_incorrect) if other_unsuccess_incorrect else 0
    print(f"Average Other Unsuccess Correct: {other_unsuccess_correct_avg:.8f}")
    print(f"Average Other Unsuccess Incorrect: {other_unsuccess_incorrect_avg:.8f}")
    return other_unsuccess_correct_avg, other_unsuccess_incorrect_avg

def time_limit_exceeded_avg(results):
    time_limit_exceeded_correct = [sample['Correct']['Time Limit Exceeded Rate'] for sample in results]
    time_limit_exceeded_incorrect = [sample['Incorrect']['Time Limit Exceeded Incorrect'] for sample in results]
    time_limit_exceeded_correct_avg = sum(time_limit_exceeded_correct) / len(time_limit_exceeded_correct) if time_limit_exceeded_correct else 0
    time_limit_exceeded_incorrect_avg = sum(time_limit_exceeded_incorrect) / len(time_limit_exceeded_incorrect) if time_limit_exceeded_incorrect else 0
    print(f"Average Time Limit Exceeded Correct: {time_limit_exceeded_correct_avg:.8f}")
    print(f"Average Time Limit Exceeded Incorrect: {time_limit_exceeded_incorrect_avg:.8f}")
    return time_limit_exceeded_correct_avg, time_limit_exceeded_incorrect_avg

def time_limit_exceeded_stdout_avg(results):
    time_limit_exceeded_stdout_correct = [sample['Correct']['Time Limit Exceeded Stdout Rate'] for sample in results]
    time_limit_exceeded_stdout_incorrect = [sample['Incorrect']['Time Limit Exceeded Incorrect Stdout'] for sample in results]
    time_limit_exceeded_stdout_correct_avg = sum(time_limit_exceeded_stdout_correct) / len(time_limit_exceeded_stdout_correct) if time_limit_exceeded_stdout_correct else 0
    time_limit_exceeded_stdout_incorrect_avg = sum(time_limit_exceeded_stdout_incorrect) / len(time_limit_exceeded_stdout_incorrect) if time_limit_exceeded_stdout_incorrect else 0
    print(f"Average Time Limit Exceeded Stdout Correct: {time_limit_exceeded_stdout_correct_avg:.8f}")
    print(f"Average Time Limit Exceeded Stdout Incorrect: {time_limit_exceeded_stdout_incorrect_avg:.8f}")
    return time_limit_exceeded_stdout_correct_avg, time_limit_exceeded_stdout_incorrect_avg

def time_limit_exceeded_stdout_false_avg(results):
    time_limit_exceeded_stdout_false_correct = [sample['Correct']['Time Limit Exceeded Stdout False Rate'] for sample in results]
    time_limit_exceeded_stdout_false_incorrect = [sample['Incorrect']['Time Limit Exceeded Incorrect Stdout False'] for sample in results]
    time_limit_exceeded_stdout_false_correct_avg = sum(time_limit_exceeded_stdout_false_correct) / len(time_limit_exceeded_stdout_false_correct) if time_limit_exceeded_stdout_false_correct else 0
    time_limit_exceeded_stdout_false_incorrect_avg = sum(time_limit_exceeded_stdout_false_incorrect) / len(time_limit_exceeded_stdout_false_incorrect) if time_limit_exceeded_stdout_false_incorrect else 0
    print(f"Average Time Limit Exceeded Stdout False Correct: {time_limit_exceeded_stdout_false_correct_avg:.8f}")
    print(f"Average Time Limit Exceeded Stdout False Incorrect: {time_limit_exceeded_stdout_false_incorrect_avg:.8f}")
    return time_limit_exceeded_stdout_false_correct_avg, time_limit_exceeded_stdout_false_incorrect_avg

def seg_fault_avg(results):
    seg_fault_correct = [sample['Correct']['Seg Fault Rate'] for sample in results]
    seg_fault_incorrect = [sample['Incorrect']['Seg Fault Incorrect'] for sample in results]
    seg_fault_correct_avg = sum(seg_fault_correct) / len(seg_fault_correct) if seg_fault_correct else 0
    seg_fault_incorrect_avg = sum(seg_fault_incorrect) / len(seg_fault_incorrect) if seg_fault_incorrect else 0
    print(f"Average Seg Fault Correct: {seg_fault_correct_avg:.8f}")
    print(f"Average Seg Fault Incorrect: {seg_fault_incorrect_avg:.8f}")
    return seg_fault_correct_avg, seg_fault_incorrect_avg

def checker_success_avg(results):
    """统计 checker_info 的成功率平均值"""
    success_correct = [sample['Correct']['Checker Success Correct'] for sample in results]
    success_incorrect = [sample['Incorrect']['Checker Success Incorrect'] for sample in results]
    success_correct_avg = sum(success_correct) / len(success_correct) if success_correct else 0
    success_incorrect_avg = sum(success_incorrect) / len(success_incorrect) if success_incorrect else 0
    print(f"Average Checker Success Correct: {success_correct_avg:.8f}")
    print(f"Average Checker Success Incorrect: {success_incorrect_avg:.8f}")
    return success_correct_avg, success_incorrect_avg

def checker_other_unsuccess_avg(results):
    """统计 checker_info 的其他失败率平均值"""
    other_unsuccess_correct = [sample['Correct']['Checker Other Unsuccess Rate'] for sample in results]
    other_unsuccess_incorrect = [sample['Incorrect']['Checker Other Unsuccess Incorrect'] for sample in results]
    other_unsuccess_correct_avg = sum(other_unsuccess_correct) / len(other_unsuccess_correct) if other_unsuccess_correct else 0
    other_unsuccess_incorrect_avg = sum(other_unsuccess_incorrect) / len(other_unsuccess_incorrect) if other_unsuccess_incorrect else 0
    print(f"Average Checker Other Unsuccess Correct: {other_unsuccess_correct_avg:.8f}")
    print(f"Average Checker Other Unsuccess Incorrect: {other_unsuccess_incorrect_avg:.8f}")
    return other_unsuccess_correct_avg, other_unsuccess_incorrect_avg

def checker_time_limit_exceeded_avg(results):
    """统计 checker_info 的超时率平均值"""
    time_limit_exceeded_correct = [sample['Correct']['Checker Time Limit Exceeded Rate'] for sample in results]
    time_limit_exceeded_incorrect = [sample['Incorrect']['Checker Time Limit Exceeded Incorrect'] for sample in results]
    time_limit_exceeded_correct_avg = sum(time_limit_exceeded_correct) / len(time_limit_exceeded_correct) if time_limit_exceeded_correct else 0
    time_limit_exceeded_incorrect_avg = sum(time_limit_exceeded_incorrect) / len(time_limit_exceeded_incorrect) if time_limit_exceeded_incorrect else 0
    print(f"Average Checker Time Limit Exceeded Correct: {time_limit_exceeded_correct_avg:.8f}")
    print(f"Average Checker Time Limit Exceeded Incorrect: {time_limit_exceeded_incorrect_avg:.8f}")
    return time_limit_exceeded_correct_avg, time_limit_exceeded_incorrect_avg

def checker_no_info_avg(results):
    """统计没有 checker_info 的比例平均值"""
    no_info_correct = [sample['Correct']['Checker No Info Rate'] for sample in results]
    no_info_incorrect = [sample['Incorrect']['Checker No Info Incorrect'] for sample in results]
    no_info_correct_avg = sum(no_info_correct) / len(no_info_correct) if no_info_correct else 0
    no_info_incorrect_avg = sum(no_info_incorrect) / len(no_info_incorrect) if no_info_incorrect else 0
    print(f"Average Checker No Info Correct: {no_info_correct_avg:.8f}")
    print(f"Average Checker No Info Incorrect: {no_info_incorrect_avg:.8f}")
    return no_info_correct_avg, no_info_incorrect_avg

result_folder = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_new_replace_add_feedback_gen/"

print("=== Original Test Results ===")
result = get_original_results_binary(result_folder, result_checker=False)

# print("\nTotal Results:")
# print(result)

sc, sic = success_avg(result)
ousc, ousic = other_unsuccess_avg(result)
tlec, tleic = time_limit_exceeded_avg(result)
tle_stdout_c, tle_stdout_ic = time_limit_exceeded_stdout_avg(result)
tle_stdout_false_c, tle_stdout_false_ic = time_limit_exceeded_stdout_false_avg(result)
sf_c, sf_ic = seg_fault_avg(result)

print("time false: ", TIME_LIMIT_EXCEEDED_STDOUT_PLUS_NUM_FALSE)
print("execute 20 second: ", TIME_LIMIT_EXCEEDED_STDOUT_ORIGINAL_NUM)
print("execute 1000 second: ", TIME_LIMIT_EXCEEDED_STDOUT_PLUS_NUM)

print(SAMPLES_NUM)

print('test case failed: ', len(TEST_FAILED))
print('test case: ', len(TESTS))

# print("\n=== Checker Results ===")
# checker_result = get_checker_results_binary(result_folder)

# csc, csic = checker_success_avg(checker_result)
# cousc, cousic = checker_other_unsuccess_avg(checker_result)
# ctlec, ctleic = checker_time_limit_exceeded_avg(checker_result)
# cnic, cniic = checker_no_info_avg(checker_result)

# print("\n=== Summary ===")
# print(f"Total samples processed: {len(result)}")
# print(f"Samples with checker results: {len(checker_result)}")
# print(f"Original vs Checker Success Rate (Correct): {sc:.8f} vs {csc:.8f}")
# print(f"Original vs Checker Success Rate (Incorrect): {sic:.8f} vs {csic:.8f}")

# %% 读取 json 文件

import json
import os
import numpy as np

dir_path = "/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_o4mini_new_replace_add_feedback_gen_command_replace_new_new/"

file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.json')]

# %%
from tqdm import tqdm
solutions = [0, 0, 0]
sample_num = [0, 0, 0]
incorrect_solutions = [0, 0, 0]
incorrect_sample_num = [0, 0, 0]

# 用于存储每个 turn 的 solutions 和 incorrect_solutions 数量
solutions_list = [[], [], []]
incorrect_solutions_list = [[], [], []]

for file_path in tqdm(file_paths):
    with open(file_path, 'r') as file:
        data = json.load(file)

    results = data['result']

    for i, result in enumerate(results):
        result = result['result']
        sol_len = len(result['solution_result'])
        inc_sol_len = len(result['incorrect_solution_result'])
        solutions[i] += sol_len
        # if sol_len > 0 else 0
        sample_num[i] += 1
        # if sol_len > 0 else 0
        incorrect_solutions[i] += inc_sol_len
        # if inc_sol_len > 0 else 0
        incorrect_sample_num[i] += 1
        # if inc_sol_len > 0 else 0
        # if sol_len > 0:
        solutions_list[i].append(sol_len)
        # if inc_sol_len > 0:
        incorrect_solutions_list[i].append(inc_sol_len)
# 计算 average
average_solutions = [s / sample_num[i] if sample_num[i] > 0 else 0 for i, s in enumerate(solutions)]
average_incorrect_solutions = [s / incorrect_sample_num[i] if incorrect_sample_num[i] > 0 else 0 for i, s in enumerate(incorrect_solutions)]
print(f"Solutions: {solutions}")
print(f"Average Solutions: {average_solutions}")
print(f"Incorrect Solutions: {incorrect_solutions}")
print(f"Average Incorrect Solutions: {average_incorrect_solutions}")

# 计算标准差
solutions_var = [np.std(solutions_list[i]) if len(solutions_list[i]) > 0 else 0 for i in range(3)]
incorrect_solutions_var = [np.std(incorrect_solutions_list[i]) if len(incorrect_solutions_list[i]) > 0 else 0 for i in range(3)]
print(f"Solutions Variance: {solutions_var}")
print(f"Incorrect Solutions Variance: {incorrect_solutions_var}")

# %% 绘制折线图
import matplotlib.pyplot as plt

turns = ['turn 1', 'turn 2', 'turn 3']
x = np.arange(len(turns))

plt.figure()
plt.plot(turns, average_solutions, label='Solutions', marker='o')
plt.plot(turns, average_incorrect_solutions, label='Incorrect Solutions', marker='o')
plt.xlabel('Turns')
plt.ylabel('Number of Solutions')
# plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
for i, v in enumerate(average_solutions):
    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
for i, v in enumerate(average_incorrect_solutions):
    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom', alpha=0.5)
plt.title('Average Solutions and Incorrect Solutions per Turn')
plt.legend()
plt.show()

# %% 绘制标准差为上下界的折线图
plt.figure()
# Solutions
avg = np.array(average_solutions)
std = np.array(solutions_var)
plt.plot(turns, avg, marker='o', label='Solutions')
plt.fill_between(turns, avg - std, avg + std, color='blue', alpha=0.2, label='Solutions ± Std')
# Incorrect Solutions
avg_inc = np.array(average_incorrect_solutions)
std_inc = np.array(incorrect_solutions_var)
plt.plot(turns, avg_inc, marker='o', label='Incorrect Solutions')
plt.fill_between(turns, avg_inc - std_inc, avg_inc + std_inc, color='orange', alpha=0.2, label='Incorrect Solutions ± Std')
# 显示每个点的值
for i, v in enumerate(avg):
    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
for i, v in enumerate(avg_inc):
    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')

# 设置 y 轴为对数尺度
# plt.yscale('log')
# # 限制 y 轴范围
# plt.ylim(0, 30)
plt.xlabel('Turns')
plt.ylabel('Average Number of Solutions')
plt.title('Average Solutions and Incorrect Solutions per Turn (with Std)')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# %%
import json
import os
from tqdm import tqdm
result_folder = "/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_o4mini_new/"

files_list = os.listdir(result_folder)
files_list_filter = [file for file in files_list if file.endswith('.json')]

corner_cases_num_step = [[], [], []]

corner_cases_num_step_replace = [[], [], []]

corner_cases_num_step_add = [[], [], []]

solution_results = [[], [], []]
incorrect_solution_results = [[], [], []]

for file_name in tqdm(files_list_filter):
    file_path = os.path.join(result_folder, file_name)
    with open(file_path, 'r') as file:
        data = json.load(file)

    corner_cases_num = len(data['corner_cases'])

    results = data['result']

    for i, result in enumerate(results):
        # if len(result['case_inputs']) != len(result['corner_cases']):
        #     print(f"File: {file_name}, Turn: {i+1}, {len(result['corner_cases'])} is not equal to {len(result['case_inputs'])}")
        # assert len(result['corner_cases']) + len(result['corner_cases_error']) == len(result['case_inputs']), f"File: {file_name}, Turn: {i+1}, {len(result['corner_cases'])} + {len(result['corner_cases_error'])} is not equal to {len(result['case_inputs'])}"
        # corner_cases_num_step[i].append(len(result['corner_cases']))
        # corner_cases_num_step_replace[i].append(len(result['case_inputs_replace']))
        # corner_cases_num_step_add[i].append(len(result['case_inputs_add']))
        solution_results[i].append(len(result['result']['solution_result']))
        incorrect_solution_results[i].append(len(result['result']['incorrect_solution_result']))

    # assert len(result['corner_cases']) == corner_cases_num
# %%
corner_cases_num_step_avg = [sum(corner_cases_num_step[i]) / len(corner_cases_num_step[i]) if len(corner_cases_num_step[i]) > 0 else 0 for i in range(3)]

corner_cases_num_step_replace_avg = [sum(corner_cases_num_step_replace[i]) / len(corner_cases_num_step_replace[i]) if len(corner_cases_num_step_replace[i]) > 0 else 0 for i in range(3)]

corner_cases_num_step_add_avg = [sum(corner_cases_num_step_add[i]) / len(corner_cases_num_step_add[i]) if len(corner_cases_num_step_add[i]) > 0 else 0 for i in range(3)]

solution_results_avg = [sum(solution_results[i]) / len(solution_results[i]) if len(solution_results[i]) > 0 else 0 for i in range(3)]

incorrect_solution_results_avg = [sum(incorrect_solution_results[i]) / len(incorrect_solution_results[i]) if len(incorrect_solution_results[i]) > 0 else 0 for i in range(3)]

# %%
