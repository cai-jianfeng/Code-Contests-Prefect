"""
该模块的主要功能是使用每个 sample 自带的 checker 对结果进行进一步验证
1. 给定结果文件夹路径，加载所有 JSON 文件
2. 对每个文件，提取正确和错误的解决方案结果
3. 对于正确的解决方案，如果其内的 result 中 accepted 为 False，则使用 checker 进行进一步验证：
    4. 对于其内的 tests，遍历每个 test case
        5. 如果其的 status 为 "Success"，则提取它的 stdout 和 test case 的 output，并使用 checker 进行验证
            6. 如果验证通过，则将该 test case 的结果 (passed) 标记为 true
            7. 如果验证失败，则直接跳过剩余的 test cases
        8. 如果其的 status 为 "Failed"，则直接跳过该解决方案
"""
# %% setup
import os
import json
from tqdm import tqdm
from solutions_eval import dataset_read
import requests
import base64

def load_results(result_folder):
    result_files = os.listdir(result_folder)
    results = {}

    for result_file in tqdm(result_files):
        print(result_file)
        if result_file.endswith(".json") and "checker" not in result_file:
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    results[data.get("id", "")] = data

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {result_file}: {e}")
                except Exception as e:
                    print(f"An error occurred while processing {result_file}: {e}")
        else:
            print(f"Skipping non-JSON file: {result_file}")

    print(f"Loaded {len(results)} results from {result_folder}")
    return results

def load_datasets(result_folder):
    result_files = os.listdir(result_folder)
    results = {}

    for result_file in tqdm(result_files):
        print(result_file)
        if result_file.endswith(".json"):
            file_path = os.path.join(result_folder, result_file)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    results[data.get("id", "")] = data

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {result_file}: {e}")
                except Exception as e:
                    print(f"An error occurred while processing {result_file}: {e}")
        else:
            print(f"Skipping non-JSON file: {result_file}")

    print(f"Loaded {len(results)} results from {result_folder}")
    return results

def sandbox_call(api_path, payload):
    """
    调用远程的 sandbox API
    """
    res = requests.post(
        api_path,
        json=payload,
    )

    return res.json()


# %% main
if __name__ == "__main__":
    # result_folder = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_plus_test_results_190sandbox_190_1concur_1runconcur_190_collocate_queue_dir_161"
    # result_folder = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_test_results_512sandbox_512_1concur_1runconcur_512_collocate_queue_dir_165"
    # result_folder = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_ours_all_solutions_test_results_512sandbox_512_1concur_1runconcur_512_collocate_queue_dir_152_0_152"
    result_folder = "/aiarena/group/llmgroup/caijf/ut_gen/code_contests_ours_repeat_test_results_512sandbox_512_1concur_1runconcur_512_collocate_queue_dir_152_0_152"
    data_folder = "/aiarena/gpfs/Code-Contests-Ours/test_repeat"

    results = load_results(result_folder)
    dataset = load_datasets(data_folder)

    # codecontents_data_path = "/aiarena/gpfs/Code-Contests-Plus/test_plus_dataset"
    # codecontents_data_path = "/aiarena/gpfs/code_contests"

    # dataset = dataset_read(codecontents_data_path, transform="codecontents_plus", split="test")
    # dataset = dataset_read(codecontents_data_path, transform="codecontents", split="test")

    # 将 results 和 dataset 通过各自的 id 进行关联
    # 将 results 根据 id 进行排序
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    # 将 dataset 根据 id 进行排序
    sorted_dataset = sorted(dataset.items(), key=lambda x: x[0])

    sandbox_url = "http://10.244.81.219:8080/run_code"

    # for result, data in zip(sorted_results, sorted_dataset):
    #     result_id, result_data = result
    #     output_file = os.path.join(result_folder, f"{result_id}_checker_incorrect.json")
    #     # 如果 output_file 已存在，则跳过
    #     if os.path.exists(output_file):
    #         print(f"Output file {output_file} already exists, skipping.")
    #         continue
    #     checker_code = data.get("checker", None)
    #     # solution_results = result_data.get("solution_result", [])
    #     solution_results = result_data.get("incorrect_solution_result", [])
    #     for sidx, solution in enumerate(solution_results):
    #         single_result = solution['result']
    #         if not single_result.get("accepted", True):
    #             # 如果 result 中的 accepted 为 False，则使用 checker 进行验证
    #             tests = single_result.get("tests", [])
    #             for idx, test in enumerate(tests):
    #                 if not test['passed']:
    #                     if test['exec_info'].get('status') == "Success":

    #                         # 提取 stdout 和 test case 的 output
    #                         stdout = test['exec_info']['run_result'].get('stdout')
    #                         expected_output = test['test_info']['output'].get('stdout')

    #                         stdin = test['test_info']['input'].get('stdin', '')
    #                         # 使用 checker 进行验证
    #                         if checker_code:
    #                             try:
    #                                 # 直接将字符串编码为 base64
    #                                 stdin_b64 = base64.b64encode(stdin.encode('utf-8')).decode('utf-8')
    #                                 stdout_b64 = base64.b64encode(stdout.encode('utf-8')).decode('utf-8')
    #                                 expected_output_b64 = base64.b64encode(expected_output.encode('utf-8')).decode('utf-8')
    #                                 with open("testlib.h", 'rb') as file:
    #                                     testlib_data_b64 = base64.b64encode(file.read()).decode('utf-8')
    #                             except Exception as e:
    #                                 print(f"Error encoding base64 for test case {idx}: {e}")
    #                                 continue

    #                             files = {
    #                                 "input.txt": stdin_b64,
    #                                 "output.txt": stdout_b64,
    #                                 "answer.txt": expected_output_b64,
    #                                 "testlib.h": testlib_data_b64
    #                             }

    #                             # 构造 API 调用的 payload
    #                             payload = {
    #                                 "code": checker_code,
    #                                 "language": "cpp",
    #                                 "extra_args": "input.txt output.txt answer.txt",
    #                                 "files": files,
    #                             }

    #                             # 调用 sandbox API
    #                             try:
    #                                 response = sandbox_call(sandbox_url, payload)

    #                                 test['checker_info'] = response
                                    
    #                             except Exception as e:
    #                                 print(f"Error calling sandbox API for test case {idx}: {e}")

    #                     elif test['exec_info'].get('status') == "Failed":
    #                         print(f"Skipping failed test case for solution {sidx} in result {result_id}")

    #                 else:
    #                     print(f"Test case {idx} already passed, skipping checker validation.")

    #         else:
    #             print(f"Solution {sidx} in result {result_id} is already accepted, skipping checker validation.")
        
    #     # 保存更新后的结果 (同样针对每个 id 保存一个文件)
        
    #     with open(output_file, 'w') as f:
    #         json.dump(result_data, f, indent=4)
    #     print(f"Updated results saved to {output_file}")
    #     print(f"Processed result for {result_id}")
    def solution_checker(solution_results, checker_code, result_id):
        for sidx, solution in enumerate(solution_results):
            single_result = solution['result']
            if not single_result.get("accepted", True):
                # 如果 result 中的 accepted 为 False，则使用 checker 进行验证
                tests = single_result.get("tests", [])
                for idx, test in enumerate(tests):
                    if not test['passed']:
                        if test['exec_info'].get('status') == "Success":
                            # 提取 stdout 和 test case 的 output
                            stdout = test['exec_info']['run_result'].get('stdout')
                            expected_output = test['test_info']['output'].get('stdout')

                            stdin = test['test_info']['input'].get('stdin', '')
                            # 使用 checker 进行验证
                            if checker_code:
                                try:
                                    # 直接将字符串编码为 base64
                                    stdin_b64 = base64.b64encode(stdin.encode('utf-8')).decode('utf-8')
                                    stdout_b64 = base64.b64encode(stdout.encode('utf-8')).decode('utf-8')
                                    expected_output_b64 = base64.b64encode(expected_output.encode('utf-8')).decode('utf-8')
                                    with open("testlib.h", 'rb') as file:
                                        testlib_data_b64 = base64.b64encode(file.read()).decode('utf-8')
                                except Exception as e:
                                    print(f"Error encoding base64 for test case {idx}: {e}")
                                    continue

                                files = {
                                    "input.txt": stdin_b64,
                                    "output.txt": stdout_b64,
                                    "answer.txt": expected_output_b64,
                                    "testlib.h": testlib_data_b64
                                }

                                # 构造 API 调用的 payload
                                payload = {
                                    "code": checker_code,
                                    "language": "cpp",
                                    "extra_args": "input.txt output.txt answer.txt",
                                    "files": files,
                                }

                                # 调用 sandbox API
                                try:
                                    response = sandbox_call(sandbox_url, payload)

                                    test['checker_info'] = response
                                    
                                except Exception as e:
                                    print(f"Error calling sandbox API for test case {idx}: {e}")

                        elif test['exec_info'].get('status') == "Failed":
                            print(f"Skipping failed test case for solution {sidx} in result {result_id}")

                    else:
                        print(f"Test case {idx} already passed, skipping checker validation.")

            else:
                print(f"Solution {sidx} in result {result_id} is already accepted, skipping checker validation.")
        
        
    for result, data in zip(sorted_results, sorted_dataset):
        result_id, result_data = result
        data_id, data_info = data
        assert result_id == data_id, f"Result ID {result_id} does not match Dataset ID {data_id}"
        # output_file = os.path.join(result_folder, f"{result_id.replace('/', '_')}_checker.json")
        output_file = os.path.join(result_folder, f"{result_id.replace('/', '_')}_checker.json")
        # 如果 output_file 已存在，则跳过
        # if os.path.exists(output_file):
        #     print(f"Output file {output_file} already exists, skipping.")
        #     continue
        checker_code = data_info.get("checker", None)
        if not checker_code:
            print(f"No checker code for result {result_id}, skipping.")
            continue
        solution_results = result_data.get("solution_result", [])
        incorrect_solution_results = result_data.get("incorrect_solution_result", [])
        solution_checker(solution_results, checker_code, result_id)
        solution_checker(incorrect_solution_results, checker_code, result_id)
        # 保存更新后的结果 (同样针对每个 id 保存一个文件)
        
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        print(f"Updated results saved to {output_file}")
        print(f"Processed result for {result_id}")

# %%
