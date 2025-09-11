"""
该代码的主要流程是:
1. 设计一个 Prompt 来引导 LLM 改进给定的 testlib.h
"""
from tqdm import tqdm
import base64
import requests
from corner_case_gen_parallel import DatasetProcessor, SandboxClient, OpenAIClient
import subprocess
import shlex
from openai import OpenAI
import re

# 设置 OpenAI API 密钥
API_BASE = "https://lonlie.plus7.plus/v1"
API_KEY = "sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

COMMAND_TEMPLATE = """Command: {command}
Compile Result:
{compile_result}
Run Result:
{run_result}"""

SAMPLE_TEMPLATE = """You are given the following C++ header file (testlib.h):

{testlib_data}

Below is a generator program that uses testlib.h:

{generator}

The following commands were used to compile and run the generator, but some failed. For each failed command, the compile and run results are shown:

{failed_command_results}

Please analyze the errors and suggest improvements to `testlib.h` to make it more robust and compatible with the generator and its usage scenarios.

When you propose changes, return them using one or more replacement blocks in this exact format (no additional prose before or after the blocks):

<<<<<<< SEARCH
<lines to search for (the original code fragment)>
=======
<replacement lines (the updated code fragment)>
>>>>>>> REPLACE

Notes:
- Use the smallest possible search/replace fragments that correctly express the change.
- Keep context lines to help locate the replacement.
- If multiple independent edits are needed, produce multiple blocks separated by a single blank line.
- Pay close attention to code indentation, spaces, and line breaks; do not omit or alter them in the search/replace fragments.
- Do not return full-file rewrites unless necessary; prefer focused replacements.
"""

import re

def apply_code_patches(original_code: str, model_response: str) -> str:
    """
    根据模型 response 中的多个 <<<<<<< SEARCH / ======= / >>>>>>> REPLACE 块
    自动对原始代码进行替换，返回改进后的代码。

    :param original_code: 原始代码字符串
    :param model_response: 模型返回的包含多个代码块的字符串
    :return: 替换后的代码
    """
    # 匹配每个替换块
    pattern = re.compile(
        r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        re.DOTALL
    )
    
    modified_code = original_code

    # 找到所有匹配的替换块
    for search_block, replace_block in pattern.findall(model_response):
        search_block = search_block.strip("\n")
        replace_block = replace_block.strip("\n")
        
        # 在代码中进行替换
        if search_block in modified_code:
            modified_code = modified_code.replace(search_block, replace_block)
        else:
            print(f"Warning: Search block not found in code:\n{search_block}\n")

    return modified_code

dataset_processor = DatasetProcessor()
dataset = dataset_processor.read_dataset("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered", "test")

# sandbox_client = SandboxClient()
session = requests.Session()

base_api_path = [
    # "10.244.179.30",
    # "10.244.188.165",
    # "10.244.204.77",
    # "10.244.128.93",
    # "10.244.81.198",
    "10.244.40.153"
]

api_paths = [
    f"http://{host}:{port+8080}/run_code" for host in base_api_path for port in range(128)
]

api_paths = [f"http://10.244.166.215:8080/run_code"]

testlib_path = "testlib.h"
with open(testlib_path, 'rb') as file:
    testlib_data_b64 = base64.b64encode(file.read()).decode('utf-8')

with open(testlib_path, 'r', encoding='utf-8') as file:
    testlib_data = file.read()

files = {
    "testlib.h": testlib_data_b64
}

api_path_begin = 0

for idx, sample in enumerate(tqdm(dataset)):
    if sample['id'] not in ['Codeforces/1619/D', 'Codeforces/1575/A', 'Codeforces/1608/B', 'Codeforces/1582/D']:
        continue
    generator = sample['generator']
    generator_cmd = sample['generator_cmd']
    command_list = [line.strip() for line in generator_cmd.splitlines() if line.strip() and not line.strip().startswith('#')]
    
    max_retry = 0
    testlib_data_back = testlib_data
    while max_retry < 1:
        failed_command_results = []
        max_retry += 1
        for command in command_list:
            api_path = api_paths[api_path_begin % len(api_paths)]
            api_path_begin += 1

            payload = {
                "code": generator,
                "language": "cpp",
                "extra_args": command.replace("./gen ", ""),
                "files": files,
                "run_timeout": 40
            }

            with session.post(api_path, json=payload) as res:
                response = res.json()
            
            if response.get('status') != 'Success':
                compile_result = response.get('compile_result', {})
                run_result = response.get('run_result', {})
                failed_command_result = COMMAND_TEMPLATE.format(
                        command=command,
                        compile_result=compile_result,
                        run_result=run_result
                )
                failed_command_results.append(failed_command_result)

        if not failed_command_results:
            break
        else:
            print(f"failed sample id: {sample['id']}, retry {max_retry}, failed commands: {len(failed_command_results)}")
        
        prompt = SAMPLE_TEMPLATE.format(
            testlib_data=testlib_data,
            generator=generator,
            failed_command_results=failed_command_results
        )

        messages = [
                {"role": "system", "content": "You are a helpful assistant. You are an editor-style assistant specialized in automatically generating code-fix suggestions. You must strictly follow the user's instructions."},
                {"role": "user", "content": prompt}
            ]
        
        model_response = client.responses.create(
            model="o4-mini",
            input=messages,
            max_output_tokens=8192,
            reasoning={"effort": "low"},
        )

        testlib_data_update = apply_code_patches(testlib_data, model_response.output_text)

        file = {
            "testlib.h": base64.b64encode(testlib_data_update.encode('utf-8')).decode('utf-8')
        }

        testlib_data = testlib_data_update
    
    if failed_command_results:
        testlib_data = testlib_data_back
        file = {
            'testlib.h': base64.b64encode(testlib_data.encode('utf-8')).decode('utf-8')
        }
        print(f"Sample {idx} still has failures after retries. Reverting to previous testlib.h.")

with open('testlib_update.h', 'w', encoding='utf-8') as f:
    f.write(testlib_data)


