# %%
import json
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

result_folder = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_new_replace_add_feedback_new_new/"

files_list = os.listdir(result_folder)
files_list_filter = [file for file in files_list if file.endswith('.json') and "checker" not in file and 'result_list' not in file]

test_cases = {}

test_cases_set = set()

for file_name in tqdm(files_list_filter):
    file_path = os.path.join(result_folder, file_name)
    with open(file_path, 'r') as file:
        data = json.load(file)

    results = data['solution_result']

    for i, result in enumerate(results):
        language = result['language']
        solution = result['solution']
        tests = result['result'].get("tests", [])
        for test in tests:
            if test['exec_info'].get('status') == "Failed":
                test_cases_set.add(test['test_info']['input']['stdin'] + "; " + test['test_info']['output']['stdout'])
                if test['test_info']['input']['stdin'] not in test_cases:
                    test_cases[test['test_info']['input']['stdin'] + "; " + test['test_info']['output']['stdout']] = {
                        'id': data['id'],
                        "stdin": test['test_info']['input']['stdin'],
                        "language": language,
                        "solution": solution,
                        "stdout": test['test_info']['output']['stdout'],
                        'stderr': {
                            "compile_result": test['exec_info'].get('compile_result', ''),
                            "run_result": test['exec_info'].get('run_result', '')
                        }
                    }

from datasets import Dataset

dataset = Dataset.load_from_disk("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered")

dataset_dict = {}

for data in tqdm(dataset):
    dataset_dict[data['id']] = data

# %%
from openai import OpenAI
from pydantic import BaseModel

# 设置 OpenAI API 密钥
API_BASE = "https://lonlie.plus7.plus/v1"
API_KEY = "sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

model = "o4-mini"
max_tokens = 8192

class Result_Model(BaseModel):
    result: str
    decision: str
    reason: str

    def to_dict(self):
        return {
            "result": self.result,
            "decision": self.decision,
            "reason": self.reason
        }

result_list = []

PROMPT_TEMPLATE = """You are a programming contest expert. You are given a programming contest problem description, an input test case, and a solution code. You need to:

1. Actually execute the given solution code with the provided input test case
2. Analyze the execution result
3. Determine whether the input test case has issues, the solution code has issues, or both are correct

## Problem Description:
{description}

## Input Test Case:
{stdin}

## Solution Code:
{solution}

Please follow these steps for analysis:

1. First, understand the problem requirements and constraints
2. Check if the input test case conforms to the problem's input format and constraints
3. Actually run the solution code with the given input and observe the real execution result or error
4. Judge the correctness of the result based on the problem requirements

Please return your analysis result in JSON format:

{{
    "result": "The actual execution result/error message when running the solution code with the given stdin",
    "decision": "decision_number: decision_description",
    "reason": "Detailed reasoning for why you made this judgment"
}}

Decision format explanation:
- "0: Both stdin and solution are correct" - Both input and solution are correct
- "1: stdin does not meet problem requirements" - Input test case violates problem constraints (e.g., out of bounds)
- "2: solution cannot perfectly handle all edge conditions" - Solution code has defects and cannot correctly handle certain cases

Please analyze carefully and provide an accurate judgment."""

def process_single_test_case(test_case_data):
    """处理单个测试用例的函数"""
    test_case, data_dict, prompt_template = test_case_data
    
    global client
    
    try:
        # assert test_case['id'] in data_dict, f"ID {test_case['id']} not found in data_dict"

        description = data_dict['description']
        stdin = test_case['stdin']
        solution = test_case['solution']

        prompt = prompt_template.format(description=description, stdin=stdin, solution=solution)

        messages = [
            {"role": "system", "content": "You are a helpful assistant. You must strictly follow the user's instructions."},
            {"role": "user", "content": prompt}
        ]

        response = client.beta.chat.completions.parse(
                        model="o4-mini",
                        messages=messages,
                        max_tokens=8192,
                        response_format=Result_Model,
                        reasoning_effort="medium",
                        verbosity="medium"
                    )
        return {
            "description": description,
            "test_case": test_case,
            "model_decision": response.choices[0].message.parsed.to_dict()
        }
    except Exception as e:
        # 返回错误信息
        return {
            "description": description,
            "test_case": test_case,
            "model_decision": {
                "result": f"Error processing test case: {str(e)}",
                "decision": "no decision",
                "reason": f"Error occurred during processing: {str(e)}"
            }
        }

# 准备数据
result_file_path = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_new_replace_add_feedback_new_new/result_list.json"

# 检查是否存在结果文件
if os.path.exists(result_file_path):
    print("Found existing result file, loading previous results...")
    with open(result_file_path, "r") as f:
        existing_results = json.load(f)
    
    # 找出需要重新处理的测试用例（decision为"no decision"的）
    test_cases_to_reprocess = []
    final_results = []
    
    for result in existing_results:
        if result['model_decision']['decision'] == "no decision":
            # 需要重新处理的测试用例
            test_case = result['test_case']
            if test_case['id'] in dataset_dict:
                test_cases_to_reprocess.append(test_case)
            final_results.append(None)  # 占位符，后续会替换
        else:
            # 保留已有的正确结果
            final_results.append(result)
    
    print(f"Found {len(test_cases_to_reprocess)} test cases to reprocess (with 'no decision')")
    test_cases_list = test_cases_to_reprocess
    
else:
    print("No existing result file found, processing all test cases...")
    test_cases_list = list(test_cases.values())
    final_results = []

# 准备待处理的数据
if test_cases_list:
    test_cases_with_dataset = [(test_case, dataset_dict[test_case['id']], PROMPT_TEMPLATE) for test_case in test_cases_list]
else:
    test_cases_with_dataset = []

if __name__ == "__main__":
    if not test_cases_with_dataset:
        print("No test cases need to be processed!")
        if final_results:
            print("All test cases already have valid results.")
    else:
        # 使用多进程处理
        print(f"Processing {len(test_cases_list)} test cases using multiprocessing...")
        num_processes = min(cpu_count(), 8)  # 使用CPU核心数，但最多8个进程

        new_results = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # 提交所有任务
            future_to_test_case = {executor.submit(process_single_test_case, data): data 
                                  for data in test_cases_with_dataset}
            
            # 收集结果
            for future in tqdm(as_completed(future_to_test_case), total=len(test_cases_list), desc="Processing"):
                try:
                    result = future.result()
                    new_results.append(result)
                except Exception as e:
                    print(f"Error processing test case: {e}")
                    # 添加错误结果
                    new_results.append({
                        "description": future_to_test_case[future][1]['description'],
                        "test_case": future_to_test_case[future][0],
                        "model_decision": {
                            "result": f"Error processing test case: {str(e)}",
                            "decision": "no decision",
                            "reason": f"Error occurred during processing: {str(e)}"
                        }
                    })

        # 合并结果
        if os.path.exists(result_file_path):
            # 如果是增量处理，需要将新结果合并到原有结果中
            new_result_idx = 0
            for i, result in enumerate(final_results):
                if result is None:  # 这是需要重新处理的位置
                    final_results[i] = new_results[new_result_idx]
                    new_result_idx += 1
            result_list = final_results
        else:
            # 如果是全新处理，直接使用新结果
            result_list = new_results

        # 保存 result_list
        print(f"Saving {len(result_list)} results...")
        with open(result_file_path, "w") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)
        
        print("Processing completed!")

# %%
import json
"/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_new_replace_add_feedback_new_new/result_list.json"
with open("/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_new_replace_add_feedback_new_new/result_list.json", "r") as f:
    result_list = json.load(f)

decisions = [0, 0, 0, 0]
newlines = [0, 0, 0, 0]
for result in result_list:
    if result['model_decision']['decision'] in "0: Both stdin and solution are correct":
        decisions[0] += 1
        if "\\n" in result['test_case']['stdin'] or not result['test_case']['stdin'].endswith("\n"):
            newlines[0] += 1
    elif result['model_decision']['decision'] in "1: stdin does not meet problem requirements":
        decisions[1] += 1
        if "\\n" in result['test_case']['stdin'] or not result['test_case']['stdin'].endswith("\n"):
            newlines[1] += 1
    elif result['model_decision']['decision'] in "2: solution cannot perfectly handle all edge conditions":
        decisions[2] += 1
        if "\\n" in result['test_case']['stdin'] or not result['test_case']['stdin'].endswith("\n"):
            newlines[2] += 1
    elif result['model_decision']['decision'] in "no decision":
        decisions[3] += 1
        if "\\n" in result['test_case']['stdin'] or not result['test_case']['stdin'].endswith("\n"):
            newlines[3] += 1
    else:
        print(f"Unknown decision: {result['model_decision']['decision']}")
# %%
import os, json
from tqdm import tqdm
result_folder = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_new_replace_add_feedback_new/"

files_list = os.listdir(result_folder)
files_list_filter = [file for file in files_list if file.endswith('.json') and "checker" in file and "result_list" not in file]

test_cases_success = {}

test_cases_success_set = set()

for file_name in tqdm(files_list_filter):
    file_path = os.path.join(result_folder, file_name)
    with open(file_path, 'r') as file:
        data = json.load(file)

    results = data['solution_result']

    for i, result in enumerate(results):
        language = result['language']
        solution = result['solution']
        result = result['result']
        accepted = result['accepted']
        if not accepted:
            tests = result.get("tests", [])
            for test in tests:
                if test['exec_info'].get('status') == "Success" and not test['passed'] and test['checker_info'].get('status') != "Success":
                    test_cases_success_set.add(test['test_info']['input']['stdin'] + "; " + test['test_info']['output']['stdout'])
                    if test['test_info']['input']['stdin'] + "; " + test['test_info']['output']['stdout'] not in test_cases_success:
                        test_cases_success[test['test_info']['input']['stdin'] + "; " + test['test_info']['output']['stdout']] = {
                            'id': data['id'],
                            "stdin": test['test_info']['input']['stdin'],
                            "language": language,
                            "solution": solution,
                            "stdout": test['test_info']['output']['stdout'],
                            "output": test['exec_info']['run_result']['stdout'],
                            'stderr': {
                                "compile_result": test['exec_info'].get('compile_result', ''),
                                "run_result": test['exec_info'].get('run_result', ''),
                                'checker_info': {
                                    "compile_result": test['checker_info'].get('compile_result', ''),
                                    "run_result": test['checker_info'].get('run_result', ''),
                                }
                            }
                        }

# %%

from datasets import Dataset
from tqdm import tqdm
dataset = Dataset.load_from_disk("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered")

dataset_dict = {}

for data in tqdm(dataset):
    dataset_dict[data['id']] = data
# %%
data = dataset_dict['Codeforces/1586/C']

generator = data['generator']

with open('test.cpp', 'w', encoding='utf-8') as f:
    f.write(generator)

import subprocess

subprocess.run(['g++', '-std=c++17', 'test.cpp', '-o', 'test'], check=True)

# 
commands = ['./gen --n 1 --m 1 --q 1 --type random', './gen --n 1000000 --m 1 --q 200000 --type single_column', './gen --n 1 --m 1000000 --q 200000 --type single_row', './gen --n 1000 --m 1000 --q 200000 --type all_empty', './gen --n 1000 --m 1000 --q 200000 --type all_filled', './gen --n 1000 --m 1000 --q 200000 --type checkerboard', './gen --n 1000 --m 1000 --q 200000 --type staircase', './gen --n 1000 --m 1000 --q 200000 --type ambiguous', './gen --n 1234 --m 567 --q 200000 --type random', './gen --n 500 --m 2000 --q 200000 --type random']

results = []

for command in tqdm(commands):
    command = command.replace('./gen', './test')
    print(f'Running command: {command}')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f'Error executing command: {stderr}')
        continue
    results.append(stdout)
# %%
import base64

testliold_path = "testlib.h"
with open(testliold_path, 'rb') as file:
    testlib_data_b64 = base64.b64encode(file.read()).decode('utf-8')

canonical_solutions = data['canonical_solution']

result = results[0]

api_paths = [
    "10.244.188.149",
    "10.244.40.134",
    "10.244.204.96",
    "10.244.128.68",
    "10.244.81.216",
    "10.244.179.6",
]

api_paths = [f"http://{ip}:{8080+i}" for ip in api_paths for i in range(128)]

import requests

session = requests.Session()

language = 'cpp'
code = canonical_solutions[language]

payload = {
    'code': code,
    'language': language,
    'stdin': result,
}


for api_path in tqdm(api_paths):
    with session.post(api_path + '/run_code', json=payload) as response:
        response_data = response.json()
        if not response_data['run_result']['stdout']:
            print(f'Failed response from {api_path}')
# %%
commands = ['./gen --n 1 --m 1 --q 1 --type random', './gen --n 1 --m 1000000 --q 200000 --type single_row', './gen --n 1000000 --m 1 --q 200000 --type single_column', './gen --n 1000 --m 1000 --q 200000 --type random', './gen --n 1000 --m 1000 --q 200000 --type all_empty', './gen --n 1000 --m 1000 --q 200000 --type all_filled', './gen --n 1000 --m 1000 --q 200000 --type checkerboard', './gen --n 1000 --m 1000 --q 200000 --type staircase', './gen --n 1000 --m 1000 --q 200000 --type ambiguous', './gen --n 2 --m 500000 --q 200000 --type random']

stdins = []

for idx, command in enumerate(tqdm(commands)):
    language = 'cpp'
    payload = {
        'code': generator,
        'language': language,
        'extra_args': command.replace('./gen ', ''),
        'files': {"testlib.h": testlib_data_b64}
    }
    api_path = api_paths[idx]
    with session.post(api_path + '/run_code', json=payload) as response:
        response_data = response.json()
        stdins.append(response_data['run_result']['stdout'])
# %%
ips = [
    "10.244.188.149",
    "10.244.40.134",
    "10.244.204.96",
    "10.244.128.68",
    "10.244.81.216",
    "10.244.179.6",
]

language = 'cpp'
payload = {
    'code': code,
    'language': language,
    'stdin': stdin,
}

with session.post(api_path, json=payload) as response:
    response_data = response.json()

# %%
from tqdm import tqdm
folder = "/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_o4mini_new_replace_add_feedback_gen_command_replace_new"
import shutil
import os

files = os.listdir(folder + "/log")

log_files = [f.replace('.log', '') for f in files if f.endswith('.log')]

result_files = [f.replace('.json', '') for f in os.listdir(folder) if f.endswith('.json')]

empty_result_files = [f.replace('.json', '') for f in os.listdir(folder + '/empty') if f.endswith('.json')]

# result_files.extend(empty_result_files)

for log_file in tqdm(sorted(log_files)):
    if log_file not in result_files:
        print(log_file)

# for empty_file in tqdm(sorted(empty_result_files)):
#     log_file = folder + "/log/" + empty_file + ".log"
#     assert os.path.exists(log_file), f"{log_file} does not exist"
#     shutil.move(log_file, log_file + ".bak")
# %%
import shutil
from tqdm import tqdm
empty_files = [
    "Codeforces_1579_A",
    "Codeforces_1579_E2",
    "Codeforces_1582_C",
    "Codeforces_1582_D",
    "Codeforces_1582_F2",
    "Codeforces_1591_F",
    "Codeforces_1601_D",
    "Codeforces_1601_F",
    "Codeforces_1603_E",
    "Codeforces_1604_B",
    "Codeforces_1604_D",
    "Codeforces_1606_E",
    "Codeforces_1607_F",
    "Codeforces_1608_A",
    "Codeforces_1613_A",
    "Codeforces_1613_F",
    "Codeforces_1618_B",
    "Codeforces_1618_C",
    "Codeforces_1620_A",
    "Codeforces_1620_D",
    "Codeforces_1622_A",
    "Codeforces_1623_A",
    "Codeforces_1623_E",
]

def remove_prefix_file(new_path: str, old_path: str, output_path: str):
    # 读取 b 的字节数
    with open(old_path, "rb") as f_old:
        old_bytes = f_old.read()
    old_len = len(old_bytes)

    with open(new_path, "rb") as f_new, open(output_path, "wb") as fout:
        # 跳过前 old_len 个字节
        f_new.seek(old_len)
        # 一边读一边写，避免一次性占内存
        while chunk := f_new.read(1024 * 1024):
            fout.write(chunk)

    print(f"已去掉前 {old_len} 字节，输出保存到 {output_path}")

for empty_file in tqdm(empty_files):
    remove_prefix_file(folder + f'/log/{empty_file}.log', folder + f'/log/{empty_file}.log.bak', folder + f'/log/{empty_file}.log.new')

# for empty_file in tqdm(empty_files):
#     shutil.copy(folder + '/log/' + empty_file + '.log', folder + '/log/' + empty_file + '.log.bak')
# %%
import re
from tqdm import tqdm
import os
pattern = re.compile(
    r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] Feedback prepared for next iteration\.\n"
    r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] Generating commands\.\.\.\n"
    r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] Error generating corner case: Request timed out\."
)

folder = "/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_o4mini_new_replace_add_feedback_gen_command_replace/log"

files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.log')]

success_files = []

for file in tqdm(files):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()

    matches = pattern.findall(content)
    if matches:
        print(f'Found {len(matches)} matches in {file}')
        success_files.append(file)

# %%
commands_pattern = re.compile(
    r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] Current commands: (\[.*?\])"
)
commands_pattern = re.compile(
    r"input_constraints_summary=(['\"])(.*?)\1\s+command_list="
)
success_files = []
unsuccess_files = []
for file in tqdm(success_files_complete):
    file += ".bak"
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()

    commands = commands_pattern.findall(content)
    if not commands or len(commands) > 1:
        print(f'Found {len(commands)} commands in {file}')
        print(commands)
        unsuccess_files.append(file.replace(folder, ''))
    else:
        success_files.append(file.replace(folder, ''))
# %%
import json
file = '/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_o4mini_new_replace_add_feedback_gen_command_replace/Codeforces_1575_A.json'

with open(file, 'r', encoding='utf-8') as f:
    content_2 = json.load(f)
# %%
import shutil
folder = '/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_new_replace_add_feedback_gen_command_replace/'
for id in tqdm(ids):
    file = folder + f'{id.replace("/", "_")}.json'
    shutil.move(file, file + ".bak")
    file = folder + f'{id.replace("/", "_")}_checker.json'
    shutil.move(file, file + ".bak")
# %%
sample_id = 'Codeforces/1582/G'
result_file = f"/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_o4mini_new_replace_add_feedback_gen_command_replace/{sample_id.replace('/', '_')}.json.bak"

with open(result_file, 'r') as f:
    original_results = json.load(f)
    all_results = original_results['result']
    messages = original_results['result'][-1]['messages']

from corner_case_gen_parallel_with_gen import Init_Command_Model, Command_Model

from openai import OpenAI
API_BASE = "https://lonlie.plus7.plus/v1"
API_KEY = "sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

response = client.beta.chat.completions.parse(
    model='o4-mini',
    messages=messages,
    max_tokens=2000,
    response_format=Command_Model,
    reasoning_effort="low",
    verbosity="medium",
    timeout=100,
)

# %%

import json


# from corner_case_gen_parallel_with_gen import Init_Command_Model, Command_Model

# from openai import OpenAI
# API_BASE = "https://lonlie.plus7.plus/v1"
# API_KEY = "sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"

# client = OpenAI(base_url=API_BASE, api_key=API_KEY)

data = []

for id in tqdm(new_ids):
    result_file = f"/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_o4mini_new_replace_add_feedback_gen_command_replace/{id.replace('/', '_')}.json.bak"

    # with open(result_file, 'r') as f:
    #     original_results = json.load(f)
    #     all_results = original_results['result']
    #     messages = original_results['result'][-1]['messages']
    #     length = 0
    #     for message in messages:
    #         length += len(message['content'])
    #     data.append(length)
    assert os.path.exists(result_file), f'{result_file} does not exist'
    assert not os.path.exists(result_file.replace('.bak', '')), f'{result_file.replace(".bak", "")} does exist'
    shutil.move(result_file, result_file.replace('.bak', ''))

# %%
import requests

session = requests.Session()

code = """
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
"""

language = 'cpp'

payload = {
    'code': code,
    'language': language
}

with session.post("http://10.244.31.71:8080/run_code", json=payload) as response:
    print(response.json())
# %%
