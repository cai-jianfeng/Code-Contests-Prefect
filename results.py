# %% setup
import json
import os
from tqdm import tqdm


result_folder_original = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_test_results_512sandbox_512_1concur_1runconcur_512_collocate_queue_dir_165"

result_folder_original = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests/"

result_folder_original_retry = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_retry/"

result_folder_original_only_generate = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_only_generate/"

result_folder_original_only_public_private = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_only_public_private/"

result_folder_original_only_public_private_generate = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_only_public_private_generate/"

result_folder_original_only_public_private_generate_retry = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_only_public_private_generate_retry/"

# result_folder_plus = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_plus_test_results_190sandbox_190_1concur_1runconcur_190_collocate_queue_dir_161"
result_folder_plus = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_plus/"

result_folder_plus_3x = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_plus_3x/"

result_folder_plus_5x = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_plus_5x/"

result_folder_ours = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_ours_test_results_512sandbox_512_1concur_1runconcur_512_collocate_queue_dir_153_0_153/"

result_folder_ours_repeat = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_ours_repeat_test_results_512sandbox_512_1concur_1runconcur_512_collocate_queue_dir_152_0_152/"

result_folder_ours_repeat_2 = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_ours_repeat_2_test_results_512sandbox_512_1concur_1runconcur_512_collocate_queue_dir_109_0_109/"

result_folder_ours_all_solutions = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_ours_all_solutions_test_results_512sandbox_512_1concur_1runconcur_512_collocate_queue_dir_152_0_152/"

result_folder_ours_all_solutions_repeat = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_ours_all_solutions_repeat_test_results_512sandbox_512_1concur_1runconcur_512_collocate_queue_dir_152_0_152/"

result_folder_ours_all_solutions_repeat_2 = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_ours_all_solutions_repeat_2_test_results_512sandbox_512_1concur_1runconcur_512_collocate_queue_dir_153_0_153/"

result_folder_ours_all_solutions_10maxsample = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_10maxsample/"

result_folder_ours_all_solutions_10maxsample_repeat = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_10maxsample_repeat/"

result_folder_ours_all_solutions_10maxsample_repeat_2 = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_10maxsample_repeat_2/"

result_folder_ours_all_solutions_o4mini = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini/"

result_folder_ours_all_solutions_o4mini_repeat = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_repeat/"

result_folder_ours_all_solutions_o4mini_repeat_2 = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_repeat_2/"

result_folder_ours_all_solutions_o4mini_10maxsample = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_10maxsample/"

result_folder_ours_all_solutions_o4mini_10maxsample_repeat = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_10maxsample_repeat/"

result_folder_ours_all_solutions_o4mini_10maxsample_repeat_2 = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_10maxsample_repeat_2/"
# %% load results (original)
def get_results(result_folder):
    result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" not in file]

    results = []

    for result_file in tqdm(result_files):
        if result_file.endswith(".json"):
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    solution_results = data.get("solution_result", [])
                    accepted = 0
                    for result in solution_results:
                        accepted += 1 if result['result'].get("accepted", False) else 0
                    TPR = accepted / len(solution_results) if solution_results else 0
                    incorrect_solutions_results = data.get("incorrect_solution_result", [])
                    unaccepted = 0
                    for incorrect_result in incorrect_solutions_results:
                        unaccepted += 1 if not incorrect_result['result'].get("accepted", True) else 0
                    TNR = unaccepted / len(incorrect_solutions_results) if incorrect_solutions_results else 0
                    results.append({
                        "id": data.get("id", ""),
                        "True Positive Rate": TPR,
                        "True Negative Rate": TNR,
                    })

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {result_file}: {e}")
                except Exception as e:
                    print(f"An error occurred while processing {result_file}: {e}")
        else:
            print(f"Skipping non-JSON file: {result_file}")

    return results

results_original = get_results(result_folder_original)
# results_plus = get_results(result_folder_plus)
results_ours = get_results(result_folder_ours)
results_ours_all_solutions = get_results(result_folder_ours_all_solutions)

# %% load results (checker)

checker_results = set()
TIME_LIMIT_EXCEEDED_NUM = 0

def solution_check(solution_result, checker=True):
    if solution_result['result'].get("accepted", False):
        return True
    tests = solution_result['result'].get("tests", [])
    result = True
    for test in tests:
        if test['passed']:
            continue
        # test['passed'] = False
        # 如果 exec_info 的 status 是 Failed，则直接返回 False
        if test['exec_info'].get('status') == "Failed":
            compile_result = test['exec_info'].get('compile_result')
            run_result = test['exec_info'].get('run_result')
            if (isinstance(compile_result, dict) and compile_result.get('status') == "TimeLimitExceeded") or \
                (isinstance(run_result, dict) and run_result.get('status') == "TimeLimitExceeded"):
                global TIME_LIMIT_EXCEEDED_NUM
                TIME_LIMIT_EXCEEDED_NUM += 1
                if isinstance(run_result, dict) and run_result['stdout'] and run_result['stdout'] in test['test_info']['output']['stdout']:
                    return True
                return False
            return False
        
        # if test['checker_info'].get('status') == "Success":
        #     if "ok" not in test['checker_info']['run_result'].get("stderr", ""):
        #         print(f"{test['checker_info']['run_result'].get('stderr', '')}")
        #         result = False
        #         return False
        #     else:
        #         checker_results.add(test['checker_info']['run_result'].get("stderr", ""))
        # else:  # 整个 solution 代码运行成功，但是 checker 代码失败，只能以 test['passed'] 为准
        #     # print(f"Checker failed for test case {test['checker_info'].get('status', 'Unknown')}")
        #     return False
        if checker:
            if test['checker_info'].get('status') != "Success":
                result = False
                return False
            else:
                checker_results.add(test['checker_info']['run_result'].get("stderr", ""))
        else:
            result = False
            return False
    return result

def solution_success(solution_result):
    tests = solution_result['result'].get("tests", [])
    for test in tests:
        if test['exec_info'].get('status') == "Failed":
            return False
    return True


def get_original_results(result_folder, result_checker=False):
    if result_checker:
        result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" in file]
    else:
        result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" not in file]
    result_files = sorted(result_files)
    results_checker = []

    for result_file in tqdm(result_files):
        if result_file.endswith(".json"):
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    solution_results = data.get("solution_result", [])
                    accepted = 0
                    total_solutions = len(solution_results)
                    for result in solution_results:
                        status = solution_check(result, checker=result_checker)
                        if status is None:
                            total_solutions -= 1
                            continue
                        if status:
                            accepted += 1
                    TPR = accepted / total_solutions if total_solutions > 0 else 0
                    incorrect_solutions_results = data.get("incorrect_solution_result", [])
                    unaccepted = 0
                    total_incorrect_solutions = len(incorrect_solutions_results)
                    for incorrect_result in incorrect_solutions_results:
                        status = solution_check(incorrect_result, checker=result_checker)
                        if status is None:
                            total_incorrect_solutions -= 1
                            continue
                        if not status:
                            unaccepted += 1
                        
                    TNR = unaccepted / total_incorrect_solutions if total_incorrect_solutions else 0
                    results_checker.append({
                        "id": data.get("id", ""),
                        "True Positive Rate": TPR,
                        "True Negative Rate": TNR,
                    })

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {result_file}: {e}")
                except Exception as e:
                    print(f"An error occurred while processing {result_file}: {e}")
        else:
            print(f"Skipping non-JSON file: {result_file}")

    return results_checker

def get_original_results_binary(result_folder, result_checker=False):
    if result_checker:
        result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" in file]
    else:
        result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" not in file]
    result_files = sorted(result_files)
    results_checker = []

    for result_file in tqdm(result_files):
        if result_file.endswith(".json"):
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    solution_results = data.get("solution_result", [])
                    success = 0
                    total_solutions = len(solution_results)
                    for result in solution_results:
                        status = solution_success(result)
                        if status:
                            success += 1
                    success_correct = success / total_solutions if total_solutions > 0 else 0
                    incorrect_solutions_results = data.get("incorrect_solution_result", [])
                    success = 0
                    total_incorrect_solutions = len(incorrect_solutions_results)
                    for incorrect_result in incorrect_solutions_results:
                        status = solution_success(incorrect_result)
                        if status:
                            success += 1

                    success_incorrect = success / total_incorrect_solutions if total_incorrect_solutions else 0
                    results_checker.append({
                        "id": data.get("id", ""),
                        "Success Correct": success_correct,
                        "Success Incorrect": success_incorrect,
                    })

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {result_file}: {e}")
                except Exception as e:
                    print(f"An error occurred while processing {result_file}: {e}")
        else:
            print(f"Skipping non-JSON file: {result_file}")

    return results_checker

def get_original_results_bak(result_folder, result_checker=None):
    result_files = [file for file in os.listdir(result_folder) if file.endswith(".json")]
    result_files = sorted(result_files)
    results_checker = []

    for result_file in tqdm(result_files):
        if result_file.endswith(".json"):
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    solution_results = data.get("solution_result", [])
                    accepted = 0
                    total_solutions = len(solution_results)
                    for result in solution_results:
                        accepted += 1 if result['result'].get("accepted", False) else 0
                    TPR = accepted / total_solutions if total_solutions > 0 else 0
                    incorrect_solutions_results = data.get("incorrect_solution_result", [])
                    unaccepted = 0
                    total_incorrect_solutions = len(incorrect_solutions_results)
                    for incorrect_result in incorrect_solutions_results:
                        unaccepted += 1 if not incorrect_result['result'].get("accepted", True) else 0
                        
                    TNR = unaccepted / total_incorrect_solutions if total_incorrect_solutions else 0
                    results_checker.append({
                        "id": data.get("id", ""),
                        "True Positive Rate": TPR,
                        "True Negative Rate": TNR,
                    })

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {result_file}: {e}")
                except Exception as e:
                    print(f"An error occurred while processing {result_file}: {e}")
        else:
            print(f"Skipping non-JSON file: {result_file}")

    return results_checker

def get_correct_checker_results(result_folder, result_checker=None):
    result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" in file and "incorrect" not in file]
    result_files = sorted(result_files)
    results_checker = []

    for result_file in tqdm(result_files):
        if result_file.endswith(".json"):
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    solution_results = data.get("solution_result", [])
                    accepted = 0
                    total_solutions = len(solution_results)
                    for result in solution_results:
                        status = solution_check(result)
                        if status is None:
                            total_solutions -= 1
                            continue
                        if status:
                            accepted += 1
                    TPR = accepted / total_solutions if total_solutions > 0 else 0
                    results_checker.append({
                        "id": data.get("id", ""),
                        "True Positive Rate": TPR,
                    })

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {result_file}: {e}")
                except Exception as e:
                    print(f"An error occurred while processing {result_file}: {e}")
        else:
            print(f"Skipping non-JSON file: {result_file}")

    return results_checker

def get_incorrect_checker_results(result_folder, results_checker=None):
    result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" in file and "incorrect" in file]
    result_files = sorted(result_files)

    for i, result_file in enumerate(tqdm(result_files)):
        if result_file.endswith(".json"):
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    solution_results = data.get("incorrect_solution_result", [])
                    unaccepted = 0
                    total_solutions = len(solution_results)
                    for result in solution_results:
                        status = solution_check(result)
                        if status is None:
                            total_solutions -= 1
                            continue
                        if not status:
                            unaccepted += 1
                    TNR = unaccepted / total_solutions if total_solutions > 0 else 0
                    assert results_checker[i]["id"] == data.get("id", ""), f"ID mismatch: {results_checker[i]['id']} != {data.get('id', '')}"
                    results_checker[i]["True Negative Rate"] = TNR

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {result_file}: {e}")
                except Exception as e:
                    print(f"An error occurred while processing {result_file}: {e}")
        else:
            print(f"Skipping non-JSON file: {result_file}")

    return results_checker

def get_checker_results(result_folder, result_checker=True):
    if result_checker:
        result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" in file]
    else:
        result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" not in file]
    result_files = sorted(result_files)
    results_checker = []

    for result_file in tqdm(result_files):
        if result_file.endswith(".json"):
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    solution_results = data.get("solution_result", [])
                    accepted = 0
                    total_solutions = len(solution_results)
                    for result in solution_results:
                        status = solution_check(result, result_checker)
                        if status is None:
                            total_solutions -= 1
                            continue
                        if status:
                            accepted += 1
                    TPR = accepted / total_solutions if total_solutions > 0 else 0
                    incorrect_solutions_results = data.get("incorrect_solution_result", [])
                    unaccepted = 0
                    total_incorrect_solutions = len(incorrect_solutions_results)
                    for incorrect_result in incorrect_solutions_results:
                        status = solution_check(incorrect_result, result_checker)
                        if status is None:
                            total_incorrect_solutions -= 1
                            continue
                        if not status:
                            unaccepted += 1
                    TNR = unaccepted / total_incorrect_solutions if total_incorrect_solutions else 0
                    results_checker.append({
                        "id": data.get("id", ""),
                        "True Positive Rate": TPR,
                        "True Negative Rate": TNR,
                    })

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {result_file}: {e}")
                except Exception as e:
                    print(f"(checker) An error occurred while processing {result_file}: {e}")
        else:
            print(f"Skipping non-JSON file: {result_file}")

    return results_checker

# %% 

# results_checker_original = get_original_results(result_folder_original)

# results_checker_plus = get_correct_checker_results(result_folder_plus)

# results_checker_plus = get_incorrect_checker_results(result_folder_plus, results_checker_plus)

results_checker_ours = get_checker_results(result_folder_ours)

results_checker_ours_repeat = get_checker_results(result_folder_ours_repeat)

results_checker_ours_repeat_2 = get_checker_results(result_folder_ours_repeat_2)

results_checker_ours_all_solutions = get_checker_results(result_folder_ours_all_solutions)

results_checker_ours_all_solutions_10maxsample = get_checker_results(result_folder_ours_all_solutions_10maxsample)

results_checker_ours_all_solutions_10maxsample_repeat = get_checker_results(result_folder_ours_all_solutions_10maxsample_repeat)

results_checker_ours_all_solutions_10maxsample_repeat_2 = get_checker_results(result_folder_ours_all_solutions_10maxsample_repeat_2)

results_checker_ours_all_solutions_o4mini = get_checker_results(result_folder_ours_all_solutions_o4mini)

results_checker_ours_all_solutions_o4mini_repeat = get_checker_results(result_folder_ours_all_solutions_o4mini_repeat)

results_checker_ours_all_solutions_o4mini_repeat_2 = get_checker_results(result_folder_ours_all_solutions_o4mini_repeat_2)

# ids = [item["id"].replace("Codeforces/", "").replace("/", "_") for item in results_checker_ours_all_solutions]

# ids_original = [item['id'] for item in results_checker_ours_all_solutions]

# results_checker_original_select = [sample for sample in results_checker_original if sample["id"] in ids_original]

# results_checker_plus_select = [sample for sample in results_checker_plus if sample["id"] in ids]

# results_checker_ours_select = [sample for sample in results_checker_ours if sample["id"] in ids_original]

# %% plot results

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_3d(results_plot, subplot, z_max=50):
    bins = np.arange(0, 1.1, 0.1)
    hist = np.zeros((10, 10), dtype=int)

    for item in results_plot:
        tpr = item["True Positive Rate"]
        TNR = item["True Negative Rate"]
        # 修正 bin 计算，确保边界值被正确处理
        tpr_bin = min(int(tpr * 10), 9)  # 直接乘以10而不是除以0.1
        TNR_bin = min(int(TNR * 10), 9)  # 直接乘以10而不是除以0.1
        hist[TNR_bin, tpr_bin] += 1

    # 使用 bin 的左边界作为位置
    xpos, ypos = np.meshgrid(bins[:-1], bins[:-1])
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.1 * np.ones_like(xpos)  # 恢复完整的 bin 宽度
    dz = hist.flatten()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(subplot, projection='3d')
    # 增加 edgecolor 和 linewidth 参数
    ax.bar3d(
        xpos, ypos, zpos, dx, dy, dz,
        shade=True, edgecolor='k', linewidth=0.5,
        zsort='max'      # 关键行：按每根柱子距相机的最远深度排序
    )
    ax.set_xlabel('True Positive Rate')
    ax.set_ylabel('True Negative Rate')
    ax.set_zlabel('Problem Count')
    # 设置 z 轴的范围
    # ax.set_zlim(0, z_max)
    ax.invert_xaxis()  # 反转 x 轴
    # ax.invert_yaxis()  # 反转 y 轴
    return hist

def plot_heatmap(results_plot, title="TPR vs TNR Heatmap"):
    bins = np.arange(0, 1.1, 0.1)
    hist = np.zeros((10, 10), dtype=int)

    for item in results_plot:
        tpr = item["True Positive Rate"]
        TNR = item["True Negative Rate"]
        tpr_bin = min(int(tpr * 10), 9)
        TNR_bin = min(int(TNR * 10), 9)
        hist[TNR_bin, tpr_bin] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    # 热力图，x轴为TPR，y轴为TNR
    im = ax.imshow(hist, origin='lower', cmap='YlOrRd', aspect='auto')

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels([f"{b:.1f}" for b in bins[:-1]])
    ax.set_yticklabels([f"{b:.1f}" for b in bins[:-1]])
    ax.set_xlabel('True Positive Rate')
    ax.set_ylabel('True Negative Rate')
    ax.set_title(title)

    # 添加色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Problem Count')

    # 在每个格子中标注数量
    for i in range(10):
        for j in range(10):
            count = hist[i, j]
            if count > 0:
                ax.text(j, i, str(count), ha='center', va='center', color='black', fontsize=8)

    plt.tight_layout()
    return hist

# %% 

hist_original = plot_3d(results_checker_original_select, 111)
# TNR: [ 1,  4,  4,  6,  7,  8,  8, 11, 17, 87]
# TPR: [ 27, 2,  2,  0,  1,  1,  0, 2,  4, 114]
# hist_plus = plot_3d(results_checker_plus_select, 111)
# TNR: [ 2,  7,  9,  8, 14, 13, 20, 21, 23, 36]
# TPR: [ 14, 1,  2,  2,  2,  8,  3,  6,  7, 108]
hist_ours = plot_3d(results_checker_ours, 111)
# TNR: [ 5,  2,  4, 16, 11, 16, 11, 22, 18, 47]
# TPR: [21,  4,  4,  5,  2,  5,  6,  6,  7, 92]
# TNR: [ 1,  3,  4, 12, 10,  8, 21, 22, 24, 47]
# TPR: [ 10, 3,  2,  8,  4,  3,  4,  2,  9, 107]
# hist_ours_all_solutions = plot_3d(results_checker_ours_all_solutions, 111)

plt.show()


# %% Calculate average TPR and TNR for all solutions
def TPR_TNR_avg(results):
    TPRs = [sample['True Positive Rate'] for sample in results]
    TNRs = [sample['True Negative Rate'] for sample in results]
    TPR_avg = sum(TPRs) / len(TPRs) if TPRs else 0
    TNR_avg = sum(TNRs) / len(TNRs) if TNRs else 0
    print(f"Average True Positive Rate: {TPR_avg:.4f}")
    print(f"Average True Negative Rate: {TNR_avg:.4f}")
    return TPR_avg, TNR_avg

def success_avg(results):
    success_correct = [sample['Success Correct'] for sample in results]
    success_incorrect = [sample['Success Incorrect'] for sample in results]
    success_correct_avg = sum(success_correct) / len(success_correct) if success_correct else 0
    success_incorrect_avg = sum(success_incorrect) / len(success_incorrect) if success_incorrect else 0
    print(f"Average Success Correct: {success_correct_avg:.4f}")
    print(f"Average Success Incorrect: {success_incorrect_avg:.4f}")
    return success_correct_avg, success_incorrect_avg
# %%
def test(result_folder):
    result_files = [file for file in os.listdir(result_folder) if file.endswith(".json") and "checker" in file]
    result_files = sorted(result_files)
    num = 0
    for result_file in tqdm(result_files):
        if result_file.endswith(".json"):
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    incorrect_solutions_results = data.get("incorrect_solution_result", [])
                    for incorrect_result in incorrect_solutions_results:
                        status = solution_check(incorrect_result)

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {result_file}: {e}")
                except Exception as e:
                    print(f"(checker) An error occurred while processing {result_file}: {e}")
                    num += 1
        else:
            print(f"Skipping non-JSON file: {result_file}")

    return num
# %%

data_original_no_checker = [
    [85.65, 82.58],
]

data_original = [
    [97.77, 72.73],
]

data_original_only_generate_no_checker = [
    [86.10, 83.11],
]

data_original_only_generate = [
    [95.68, 76.68],
]

data_original_only_public_private_no_checker = [
    [86.75, 58.10],
]

data_original_only_public_private = [
    [98.25, 48.78],
]

data_original_only_public_private_generate_no_checker = [
    [85.67, 82.38],
]

data_original_only_public_private_generate = [
    [95.76, 75.63],
]


data_plus_no_checker = [
    [58.75, 91.35],
]

data_plus = [
    [73.19, 82.16],
]

data_plus_correct_no_checker = [
    [71.32, 89.76],
]

data_plus_correct = [
    [85.76, 80.58],
]

data_plus_3x_no_checker = [
    [57.96, 92.60],
]

data_plus_3x = [
    [70.42, 84.49],
]

data_plus_3x_correct_no_checker = [
    [73.80, 87.68],
]

data_plus_3x_correct = [
    [86.26, 79.57],
]

data_plus_5x_no_checker = [
    [57.94, 93.22],
]

data_plus_5x = [
    [69.06, 86.98],
]

data_plus_5x_correct_no_checker = [
    [74.91, 87.52],
]

data_plus_5x_correct = [
    [86.03, 81.28],
]

data_ours_no_checker = [
    [78.99, 68.54],
]

data_ours = [
    [91.78, 59.02],
]

data_ours_max_no_checker = [
    [82.38, 71.62],
]

data_ours_max = [
    [94.83, 62.20],
]

data_ours_10sample_no_checker = [
    [79.95, 68.85],
]

data_ours_10sample = [
    [90.21, 60.88],
]

data_ours_10sample_max_no_checker = [
    [83.13, 71.38],
]

data_ours_10sample_max = [
    [94.02, 62.87],
]

data_ours_replace_add_no_checker = [
    [67.12, 81.12],
]

data_ours_replace_add = [
    [78.06, 72.95],
]

data_ours_replace_add_feedback_no_checker = [
    [63.72, 80.36],
]

data_ours_replace_add_feedback = [
    [74.52, 72.47],
]

data_ours_replace_add_max_no_checker = [
    [78.90, 76.64],
]

data_ours_replace_add_max = [
    [89.87, 68.41],
]

data_ours_replace_add_feedback_new_no_checker = [
    [67.08, 78.56],
]

data_ours_replace_add_feedback_new = [
    [78.72, 69.18],
]

data_ours_replace_add_feedback_new_new_no_checker = [
    [66.81, 79.52],
]

data_ours_replace_add_feedback_new_new = [
    [79.11, 70.84]
]

data_ours_replace_add_feedback_new_new_gpt4o_no_checker = [
    [58.71, 75.09],
]

data_ours_replace_add_feedback_new_new_gpt4o = [
    [67.63, 68.48],
]

data_ours_replace_add_feedback_with_gen_no_checker = [
    [76.55, 85.73],
]

data_ours_replace_add_feedback_with_gen = [
    [89.51, 76.45],
]

data_ours_replace_add_feedback_with_gen_first_no_checker = [
    [75.85, 84.84],
]

data_ours_replace_add_feedback_with_gen_first = [
    [88.80, 75.58],
]

data_ours_replace_add_feedback_with_gen_command_replace_no_checker = [
    [74.90, 86.32],
]

data_ours_replace_add_feedback_with_gen_command_replace = [
    [88.39, 77.08],
]

data_ours_replace_add_feedback_with_gen_command_replace_new_no_checker = [
    [76.15, 86.43],
]

data_ours_replace_add_feedback_with_gen_command_replace_new = [
    [89.41, 77.13],
]

data_ours_replace_add_feedback_with_gen_command_replace_new_first_no_checker = [
    [75.89, 84.73],
]

data_ours_replace_add_feedback_with_gen_command_replace_new_first = [
    [88.96, 75.18],
]

data_ours_replace_add_feedback_with_gen_command_replace_new_max_no_checker = [
    [77.42, 86.67],
]

data_ours_replace_add_feedback_with_gen_command_replace_new_max = [
    [90.58, 77.10],
]

data_ours_replace_add_feedback_with_gen_command_replace_new_all_no_checker = [
    [76.14, 85.55],
]

data_ours_replace_add_feedback_with_gen_command_replace_new_all = [
    [89.56, 75.76],
]

data_ours_replace_add_feedback_with_gen_command_replace_new_new_no_checker = [
    [78.39, 85.66],
]

data_ours_replace_add_feedback_with_gen_command_replace_new_new = [
    [92.17, 76.51],
]

# 绘制散点图，同一组内的使用相同颜色
import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(data, labels, colors, title):
    plt.figure(figsize=(10, 6))
    all_x = []
    all_y = []
    scatter_points = []
    # 与箭头颜色同步
    scatter_colors = [
        'blue',           # Original
        'cyan',           # Original (No Checker)
        'purple',         # Original (Only Generate)
        'violet',         # Original (Only Generate No Checker)
        'brown',          # Original (Only Public and Private)
        'peru',           # Original (Only Public and Private No Checker)
        'darkgreen',      # Original (Only Public and Private Generate)
        'darkslategray',  # Original (Only Public and Private Generate No Checker)
        'orange',         # Plus
        'gold',           # Plus (No Checker)
        'crimson',        # Plus (Correct)
        'red',            # Plus (Correct No Checker)
        'teal',           # Plus (3x)
        'skyblue',        # Plus (3x No Checker)
        'olive',          # Plus (3x Correct)
        'limegreen',      # Plus (3x Correct No Checker)
        'deepskyblue',    # Plus (5x)
        'darkblue',       # Plus (5x No Checker)
        'saddlebrown',    # Plus (5x Correct)
        'darkred',        # Plus (5x Correct No Checker)
        'darkslategray',  # Ours
        'darkgray',       # Ours (No Checker)
        'slateblue',      # Ours (Max)
        'darkolivegreen', # Ours (Max No Checker)
        'darkviolet',     # Ours (10 Sample)
        'darkmagenta',    # Ours (10 Sample No Checker)
        'indigo',         # Ours (10 Sample Max)
        'darkred',        # Ours (10 Sample Max No Checker)
        'goldenrod',      # Ours (Replace Add)
        'darkviolet',     # Ours (Replace Add No Checker)
        'firebrick',      # Ours (Replace Add Feedback)
        'darkred',        # Ours (Replace Add Feedback No Checker)
        'seagreen',       # Ours (Replace Add Max)
        'darkred',        # Ours (Replace Add Max No Checker)
        'navy',           # Ours (Replace Add Feedback New)
        'darkred',        # Ours (Replace Add Feedback New No Checker)
        'magenta',        # Ours (Replace Add Feedback New New)
        'darkred',        # Ours (Replace Add Feedback New New No Checker)
        'darkorange',    # Ours (Replace Add Feedback New New GPT4O)
        'darkred',        # Ours (Replace Add Feedback New New GPT4O No Checker)
        'darkorange',     # Ours (Replace Add Feedback With Gen)
        'darkred',        # Ours (Replace Add Feedback With Gen No Checker)
        'lightcoral',     # Ours (Replace Add Feedback With Gen First)
        'darkred',        # Ours (Replace Add Feedback With Gen First No Checker)
        'purple',         # Ours (Replace Add Feedback With Gen Command Replace)
        'lightgreen',        # Ours (Replace Add Feedback With Gen Command Replace No Checker)
        'brown',          # Ours (Replace Add Feedback With Gen Command Replace New All)
        'peru',           # Ours (Replace Add Feedback With Gen Command Replace New All No Checker)
        'goldenrod',      # Ours (Replace Add Feedback With Gen Command Replace New)
        'darkviolet',     # Ours (Replace Add Feedback With Gen Command Replace New No Checker)
        'purple',         # Ours (Replace Add Feedback With Gen Command Replace New First)
        'violet',         # Ours (Replace Add Feedback With Gen Command Replace New First No Checker)
        'teal',           # Ours (Replace Add Feedback With Gen Command Replace New Max)
        'skyblue',        # Ours (Replace Add Feedback With Gen Command Replace New Max No Checker),
        'magenta',        # Ours (Replace Add Feedback With Gen Command Replace New New)
        'darkred',        # Ours (Replace Add Feedback With Gen Command Replace New New No Checker)
    ]
    for d, label, color in zip(data, labels, scatter_colors):
        x = [item[1] for item in d]  # 交换 x 和 y
        y = [item[0] for item in d]
        all_x.extend(x)
        all_y.extend(y)
        scatter_points.append((x, y, color, label))
        if len(d) > 1:
            plt.scatter(x, y, label=label, color=color, alpha=0.5)
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            plt.scatter([mean_x], [mean_y], color=color, marker='D', s=100, edgecolors='k', label=f"{label} Avg", alpha=0.8)
        else:
            plt.scatter(x, y, color=color, marker='D', s=100, edgecolors='k', label=label, alpha=0.8)
    # 获取当前 x, y 轴最大值
    x_m = max(all_x) if all_x else 0
    y_m = max(all_y) if all_y else 0
    x_min= min(all_x) if all_x else 0
    y_min = min(all_y) if all_y else 0
    # 绘制平行线
    plt.plot([0, x_m], [y_m, y_m], color='black', linestyle='--', linewidth=2, alpha=0.5)
    plt.plot([x_m, x_m], [0, y_m], color='black', linestyle='--', linewidth=2, alpha=0.5)
    # 在交点处写上坐标
    plt.text(x_m - 6, y_m, f'({x_m:.2f}, {y_m:.2f})', color='black', fontsize=12, ha='left', va='bottom', fontweight='bold')
    plt.xlabel('True Negative Rate (%)')
    plt.ylabel('True Positive Rate (%)')
    # 设置 x, y 轴范围
    plt.xlim(x_min - 2, x_m + 2)
    plt.ylim(y_min - 2, y_m + 2)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.grid(True)

    # 添加箭头：no checker 到正常，箭头在两点正中间，长度可调
    def arrow_between_points(x0, y0, x1, y1, alpha=0.5):
        # alpha: 箭头长度占两点距离的比例 (0, 1]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2  # 两点中点
        dx = x1 - x0
        dy = y1 - y0
        length = ((dx ** 2 + dy ** 2) ** 0.5) * alpha / 2
        if dx == 0 and dy == 0:
            return mx, my, mx, my  # 避免零向量
        # 单位方向向量
        ux = dx / (dx ** 2 + dy ** 2) ** 0.5
        uy = dy / (dx ** 2 + dy ** 2) ** 0.5
        # 箭头起点和终点
        sx = mx - ux * length
        sy = my - uy * length
        ex = mx + ux * length
        ey = my + uy * length
        return sx, sy, ex, ey


    # 为每组分配唯一颜色
    arrow_colors = [
        'blue',           # Original
        'purple',         # Original Only Generate
        'brown',          # Original Only Public Private
        'darkgreen',      # Original Only Public Private Generate
        'orange',         # Plus
        'crimson',        # Plus (Correct)
        'teal',           # Plus (3x)
        'olive',          # Plus (3x Correct)
        'deepskyblue',    # Plus (5x)
        'saddlebrown',    # Plus (5x Correct)
        'darkslategray',  # Ours
        'slateblue',      # Ours (Max)
        'darkviolet',     # Ours (10 Sample)
        'indigo',         # Ours (10 Sample Max)
        'goldenrod',      # Ours (Replace Add)
        'firebrick',      # Ours (Replace Add Feedback)
        'seagreen',       # Ours (Replace Add Max)
        'navy',           # Ours (Replace Add Feedback New)
        'magenta',        # Ours (Replace Add Feedback New New)
        'darkorange',    # Ours (Replace Add Feedback New New GPT4O)
        'lightcoral',     # Ours (Replace Add Feedback With Gen)
        'darkred',        # Ours (Replace Add Feedback With Gen First)
        'purple',         # Ours (Replace Add Feedback With Gen Command Replace)
        'brown',          # Ours (Replace Add Feedback With Gen Command Replace New All)
        'goldenrod',      # Ours (Replace Add Feedback With Gen Command Replace New)
        'purple',         # Ours (Replace Add Feedback With Gen Command Replace New First)
        'teal',           # Ours (Replace Add Feedback With Gen Command Replace New Max)
        'darkviolet',     # Ours (Replace Add Feedback With Gen Command Replace New New)
    ]
    arrow_labels = [
        'Original',
        'Original Only Generate',
        'Original Only Public Private',
        'Original Only Public Private Generate',
        'Plus',
        'Plus (Correct)',
        'Plus (3x)',
        'Plus (3x Correct)',
        'Plus (5x)',
        'Plus (5x Correct)',
        'Ours',
        'Ours (Max)',
        'Ours (10 Sample)',
        'Ours (10 Sample Max)',
        'Ours (Replace Add)',
        'Ours (Replace Add Feedback)',
        'Ours (Replace Add Max)',
        'Ours (Replace Add Feedback New)',
        'Ours (Replace Add Feedback New New)',
        'Ours (Replace Add Feedback New New GPT4O)',
        'Ours (Replace Add Feedback With Gen)',
        'Ours (Replace Add Feedback With Gen First)',
        'Ours (Replace Add Feedback With Gen Command Replace)',
        'Ours (Replace Add Feedback With Gen Command Replace New All)',
        'Ours (Replace Add Feedback With Gen Command Replace New)',
        'Ours (Replace Add Feedback With Gen Command Replace New First)',
        'Ours (Replace Add Feedback With Gen Command Replace New Max)',
        'Ours (Replace Add Feedback With Gen Command Replace New New)',
    ]
    # 数据对
    data_pairs = [
        (data_original_no_checker, data_original),
        (data_original_only_generate_no_checker, data_original_only_generate),
        (data_original_only_public_private_no_checker, data_original_only_public_private),
        (data_original_only_public_private_generate_no_checker, data_original_only_public_private_generate),
        (data_plus_no_checker, data_plus),
        (data_plus_correct_no_checker, data_plus_correct),
        (data_plus_3x_no_checker, data_plus_3x),
        (data_plus_3x_correct_no_checker, data_plus_3x_correct),
        (data_plus_5x_no_checker, data_plus_5x),
        (data_plus_5x_correct_no_checker, data_plus_5x_correct),
        (data_ours_no_checker, data_ours),
        (data_ours_max_no_checker, data_ours_max),
        (data_ours_10sample_no_checker, data_ours_10sample),
        (data_ours_10sample_max_no_checker, data_ours_10sample_max),
        (data_ours_replace_add_no_checker, data_ours_replace_add),
        (data_ours_replace_add_feedback_no_checker, data_ours_replace_add_feedback),
        (data_ours_replace_add_max_no_checker, data_ours_replace_add_max),
        (data_ours_replace_add_feedback_new_no_checker, data_ours_replace_add_feedback_new),
        (data_ours_replace_add_feedback_new_new_no_checker, data_ours_replace_add_feedback_new_new),
        (data_ours_replace_add_feedback_new_new_gpt4o_no_checker, data_ours_replace_add_feedback_new_new_gpt4o),
        (data_ours_replace_add_feedback_with_gen_no_checker, data_ours_replace_add_feedback_with_gen),
        (data_ours_replace_add_feedback_with_gen_first_no_checker, data_ours_replace_add_feedback_with_gen_first),
        (data_ours_replace_add_feedback_with_gen_command_replace_no_checker, data_ours_replace_add_feedback_with_gen_command_replace),
        (data_ours_replace_add_feedback_with_gen_command_replace_new_all_no_checker, data_ours_replace_add_feedback_with_gen_command_replace_new_all),
        (data_ours_replace_add_feedback_with_gen_command_replace_new_no_checker, data_ours_replace_add_feedback_with_gen_command_replace_new),
        (data_ours_replace_add_feedback_with_gen_command_replace_new_first_no_checker, data_ours_replace_add_feedback_with_gen_command_replace_new_first),
        (data_ours_replace_add_feedback_with_gen_command_replace_new_max_no_checker, data_ours_replace_add_feedback_with_gen_command_replace_new_max),
        (data_ours_replace_add_feedback_with_gen_command_replace_new_new_no_checker, data_ours_replace_add_feedback_with_gen_command_replace_new_new),
    ]
    arrow_text_offsets = [
        (-4, -1), (0.5, -0.2), (-5.5, -1), (-6.5, 2), (-2, -1), (1, -1),
        (-1.5, -2), (-8, 0), (-1, 1), (1, -1), (1, -1), (-0.5, 1),
        (-0.5, 1), (-0.5, 0), (0.2, 0), (0.2, 0), (0.2, 0), (-16, 0), (-16, 1), (-16, 2), 
        (0.5, -1), (0.5, -1), (0.5, -1), (0.5, -1), (0.5, -1), (0.5, -1), (0.5, -1), (0.5, -1), 
    ]
    for i, ((d0, d1), color, label, (dx, dy)) in enumerate(zip(data_pairs, arrow_colors, arrow_labels, arrow_text_offsets)):
        x0, y0 = d0[0][1], d0[0][0]
        x1, y1 = d1[0][1], d1[0][0]
        sx, sy, ex, ey = arrow_between_points(x0, y0, x1, y1, alpha=0.7)
        plt.annotate(
            '', xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.5),
            annotation_clip=False
        )
        plt.text((sx + ex) / 2 + dx, (sy + ey) / 2 + dy, label, color=color, alpha=0.5)

    plt.show()
plot_scatter(
    [
        data_original,
        data_original_no_checker,
        data_original_only_generate,
        data_original_only_generate_no_checker,
        data_original_only_public_private,
        data_original_only_public_private_no_checker,
        data_original_only_public_private_generate,
        data_original_only_public_private_generate_no_checker,
        data_plus,
        data_plus_no_checker,
        data_plus_correct,
        data_plus_correct_no_checker,
        data_plus_3x,
        data_plus_3x_no_checker,
        data_plus_3x_correct,
        data_plus_3x_correct_no_checker,
        data_plus_5x,
        data_plus_5x_no_checker,
        data_plus_5x_correct,
        data_plus_5x_correct_no_checker,
        data_ours,
        data_ours_no_checker,
        data_ours_max,
        data_ours_max_no_checker,
        data_ours_10sample,
        data_ours_10sample_no_checker,
        data_ours_10sample_max,
        data_ours_10sample_max_no_checker,
        data_ours_replace_add,
        data_ours_replace_add_no_checker,
        data_ours_replace_add_feedback,
        data_ours_replace_add_feedback_no_checker,
        data_ours_replace_add_max,
        data_ours_replace_add_max_no_checker,
        data_ours_replace_add_feedback_new,
        data_ours_replace_add_feedback_new_no_checker,
        data_ours_replace_add_feedback_new_new,
        data_ours_replace_add_feedback_new_new_no_checker,
        data_ours_replace_add_feedback_new_new_gpt4o,
        data_ours_replace_add_feedback_new_new_gpt4o_no_checker,
        data_ours_replace_add_feedback_with_gen,
        data_ours_replace_add_feedback_with_gen_no_checker,
        data_ours_replace_add_feedback_with_gen_first,
        data_ours_replace_add_feedback_with_gen_first_no_checker,
        data_ours_replace_add_feedback_with_gen_command_replace,
        data_ours_replace_add_feedback_with_gen_command_replace_no_checker,
        data_ours_replace_add_feedback_with_gen_command_replace_new_all,
        data_ours_replace_add_feedback_with_gen_command_replace_new_all_no_checker,
        data_ours_replace_add_feedback_with_gen_command_replace_new,
        data_ours_replace_add_feedback_with_gen_command_replace_new_no_checker,
        data_ours_replace_add_feedback_with_gen_command_replace_new_first,
        data_ours_replace_add_feedback_with_gen_command_replace_new_first_no_checker,
        data_ours_replace_add_feedback_with_gen_command_replace_new_max,
        data_ours_replace_add_feedback_with_gen_command_replace_new_max_no_checker,
        data_ours_replace_add_feedback_with_gen_command_replace_new_new,
        data_ours_replace_add_feedback_with_gen_command_replace_new_new_no_checker,
    ],
    [
        'Original',
        'Original (No Checker)',
        'Original (Only Generate)',
        'Original (Only Generate No Checker)',
        'Original (Only Public and Private)',
        'Original (Only Public and Private No Checker)',
        'Original (Only Public and Private Generate)',
        'Original (Only Public and Private Generate No Checker)',
        'Plus',
        'Plus (No Checker)',
        'Plus (Correct)',
        'Plus (Correct No Checker)',
        'Plus (3x)',
        'Plus (3x No Checker)',
        'Plus (3x Correct)',
        'Plus (3x Correct No Checker)',
        'Plus (5x)',
        'Plus (5x No Checker)',
        'Plus (5x Correct)',
        'Plus (5x Correct No Checker)',
        'Ours',
        'Ours (No Checker)',
        'Ours (Max)',
        'Ours (Max No Checker)',
        'Ours (10 Sample)',
        'Ours (10 Sample No Checker)',
        'Ours (10 Sample Max)',
        'Ours (10 Sample Max No Checker)',
        'Ours (Replace Add)',
        'Ours (Replace Add No Checker)',
        'Ours (Replace Add Feedback)',
        'Ours (Replace Add Feedback No Checker)',
        'Ours (Replace Add Max)',
        'Ours (Replace Add Max No Checker)',
        'Ours (Replace Add Feedback New)',
        'Ours (Replace Add Feedback New No Checker)',
        'Ours (Replace Add Feedback New New)',
        'Ours (Replace Add Feedback New New No Checker)',
        'Ours (Replace Add Feedback New New GPT4O)',
        'Ours (Replace Add Feedback New New GPT4O No Checker)',
        'Ours (Replace Add Feedback With Gen)',
        'Ours (Replace Add Feedback With Gen No Checker)',
        'Ours (Replace Add Feedback With Gen First)',
        'Ours (Replace Add Feedback With Gen First No Checker)',
        'Ours (Replace Add Feedback With Gen Command Replace)',
        'Ours (Replace Add Feedback With Gen Command Replace No Checker)',
        'Ours (Replace Add Feedback With Gen Command Replace New All)',
        'Ours (Replace Add Feedback With Gen Command Replace New All No Checker)',
        'Ours (Replace Add Feedback With Gen Command Replace New)',
        'Ours (Replace Add Feedback With Gen Command Replace New No Checker)',
        'Ours (Replace Add Feedback With Gen Command Replace New First)',
        'Ours (Replace Add Feedback With Gen Command Replace New First No Checker)',
        'Ours (Replace Add Feedback With Gen Command Replace New Max)',
        'Ours (Replace Add Feedback With Gen Command Replace New Max No Checker)',
        'Ours (Replace Add Feedback With Gen Command Replace New New)',
        'Ours (Replace Add Feedback With Gen Command Replace New New No Checker)',
    ],
    [
        'blue',        # Original
        'cyan',        # Original (No Checker)
        'purple',      # Original (Only Generate)
        'violet',      # Original (Only Generate No Checker)
        'brown',       # Original (Only Public and Private)
        'peru',        # Original (Only Public and Private No Checker)
        'darkgreen',   # Original (Only Public and Private Generate)
        'darkslategray',  # Original (Only Public and Private Generate No Checker)
        'orange',      # Plus
        'gold',        # Plus (No Checker)
        'lightcoral',  # Plus (Correct)
        'red',         # Plus (Correct No Checker)
        'lightblue',   # Plus (3x)
        'skyblue',     # Plus (3x No Checker)
        'lightgreen',  # Plus (3x Correct)
        'limegreen',   # Plus (3x Correct No Checker)
        'darkblue',    # Plus (5x)
        'deepskyblue', # Plus (5x No Checker)
        'darkorange',  # Plus (5x Correct)
        'darkred',     # Plus (5x Correct No Checker)
        'darkslategray',  # Ours
        'darkgray',    # Ours (No Checker)
        'darkcyan',    # Ours (Max)
        'darkolivegreen',  # Ours (Max No Checker)
        'darkviolet',   # Ours (10 Sample)
        'darkmagenta',  # Ours (10 Sample No Checker)
        'darkorange',  # Ours (10 Sample Max)
        'darkred',     # Ours (10 Sample Max No Checker)
        'darkgoldenrod',  # Ours (Replace Add)
        'darkviolet',   # Ours (Replace Add No Checker)
        'darkorange',  # Ours (Replace Add Feedback)
        'darkred',     # Ours (Replace Add Feedback No Checker)
        'darkorange',  # Ours (Replace Add Max)
        'darkred',     # Ours (Replace Add Max No Checker)
        'darkorange',  # Ours (Replace Add Feedback New)
        'darkred',     # Ours (Replace Add Feedback New No Checker)
        'darkorange',  # Ours (Replace Add Feedback New New)
        'darkred',     # Ours (Replace Add Feedback New New No Checker)
        'darkorange',  # Ours (Replace Add Feedback New New GPT4O)
        'darkred',     # Ours (Replace Add Feedback New New GPT4O No Checker)
        'lightcoral',  # Ours (Replace Add Feedback With Gen)
        'darkred',     # Ours (Replace Add Feedback With Gen No Checker)
        'lightcoral',  # Ours (Replace Add Feedback With Gen First)
        'darkred',     # Ours (Replace Add Feedback With Gen First No Checker)
        'purple',      # Ours (Replace Add Feedback With Gen Command Replace)
        'lightgreen',  # Ours (Replace Add Feedback With Gen Command Replace No Checker)
    ],
    'TPR vs TNR Comparison'
)
# %%
