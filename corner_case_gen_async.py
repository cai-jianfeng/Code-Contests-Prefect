"""
该代码的主要流程是:
1. 设计一个 Prompt 来引导 LLM 生成特定的 corner case;
2. 调用 OpenAI 的 API 生成 corner case;
3. 调用远程的 sandbox API 来验证生成的 corner case 是否正确:
    3.1: 将生成的 corner case 和给定的 solution/incorrect_solution 一起提交到 sandbox;
    3.2: sandbox 会运行给定的 solution/incorrect_solution 并返回结果;
    3.3: 根据 sandbox 返回的结果来判断 corner case 是否正确;
4. 如果
"""

# %% set up
import os
import json
import requests
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm  # 可选，或直接用 tqdm
from openai import OpenAI
from datasets import load_dataset
import asyncio
import aiohttp

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = "sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"
os.environ["OPENAI_API_BASE"] = "https://lonlie.plus7.plus/v1"

TEMPLATE = """
Here is a {language} solution to the problem: 
```{language}
{solution}
```
"""
LANGUAGE = ["UNKNOWN_LANGUAGE", "PYTHON", "CPP", "PYTHON3", "JAVA"]

# %% define functions
def generate_corner_case(prompt, model="gpt-4o", max_tokens=1000):
    """
    使用 OpenAI API 生成 corner case。
    
    :param prompt: 用于生成 corner case 的提示语。
    :param model: 使用的模型名称，默认为 gpt-4o。
    :param max_tokens: 生成文本的最大长度。
    :return: 生成的 corner case 文本。
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content.strip()


# 读取 codecontests 数据集
def dataset_read(data_path, split):
    data = load_dataset(data_path, split=split)
    data = list(data)
    # 将其转化为 CommonOJ 格式
    format_data = []
    for sample in tqdm(data):
        format_sample = transform_codecontents(sample)
        format_data.append(format_sample)
    return format_data


def transform_codecontents(sample):
    """
    ['name', 'description', 'public_tests', 'private_tests', 'generated_tests', 'source', 'difficulty', 'solutions', 'incorrect_solutions', 'cf_contest_id', 'cf_index', 'cf_points', 'cf_rating', 'cf_tags', 'is_description_translated', 'untranslated_description', 'time_limit', 'memory_limit_bytes', 'input_file', 'output_file'] -> ['id', 'content', 'test', 'labels', 'canonical_solution']
    """
    
    format_sample = {}
    for key, value in sample.items():
        format_sample[key] = value
    format_sample['id'] = "/".join(['Codeforces'] + sample['name'].split('.')[0].split('_'))
    format_sample['content'] = '# ' + sample['name'].split('.')[-1].strip() + '\n\n' + sample['description']
    format_sample['labels'] = {
        "tag": sample['cf_tags'],
        "title": sample['name'].split('.')[-1].strip()
    }
    canonical_solution = {}
    for i in range(len(LANGUAGE)):
        if i in sample['solutions']['language']:
            if "PYTHON" in LANGUAGE[i]:  # 将 PYTHON 和 PYTHON3 统一处理
                canonical_solution['python'] = sample['solutions']['solution'][sample['solutions']['language'].index(i)]
            else:
                canonical_solution[LANGUAGE[i].lower()] = sample['solutions']['solution'][sample['solutions']['language'].index(i)]
        # else:
        #     canonical_solution[LANGUAGE[i].lower()] = None
    
    
    format_sample['canonical_solution'] = canonical_solution

    test_cases = []
    test_cases_trasform(sample['public_tests'], test_cases)
    test_cases_trasform(sample['private_tests'], test_cases)
    test_cases_trasform(sample['generated_tests'], test_cases)

    format_sample['test'] = test_cases
    return format_sample
    

def test_cases_trasform(test_cases, format_test_cases):
    for input_data, output_data in zip(test_cases['input'], test_cases['output']):
        format_test_cases.append({
            'input': {
                'stdin': input_data
            },
            'output': {
                'stdout': output_data
            }
            })
        

async def sandbox_call(config, completion, api_path, id, dataset_type, session, semaphore):
    """
    异步调用远程的 sandbox API 来验证生成的 corner case 是否正确。
    """
    payload = {
        'dataset': dataset_type,
        'id': '',
        'completion': completion,
        'config': config
    }
    async with semaphore:
        try:
            async with session.post(api_path, json=payload) as res:
                print("submitted to sandbox:", id)
                return await res.json()
        except asyncio.TimeoutError:
            print(f"TimeoutError: {id} 请求超时")
            return {"error": "timeout", "id": id}
        except Exception as e:
            print(f"Error: {id} 请求异常: {e}")
            return {"error": str(e), "id": id}

async def solution_call(config, solutions, api_path, id, dataset_type, session, semaphore):
    """
    异步调用远程的 sandbox API 来验证生成的 corner case 是否正确
    """
    results = []
    tasks = []
    for language_index, solution in zip(solutions['language'], solutions['solution']):
        language = LANGUAGE[language_index]
        if language == "UNKNOWN_LANGUAGE":
            continue
        if "PYTHON" in language:
            language = "PYTHON"
        config_copy = config.copy()
        config_copy['language'] = language.lower()
        completion = TEMPLATE.format(language=language.lower(), solution=solution)
        tasks.append(
            asyncio.create_task(
                sandbox_call(config_copy, completion, api_path, id, dataset_type, session, semaphore)
            )
        )
        results.append({
            'language': language,
            'solution': solution,
            'result': None  # placeholder, will be filled after await
        })
    if tasks:
        responses = await asyncio.gather(*tasks)
        for i, resp in enumerate(responses):
            results[i]['result'] = resp
    return results

async def codecontests_call(dataset, api_path, dataset_type="code_contests_test", results_path=None):
    """
    异步调用远程的 sandbox API 来验证生成的 corner case 是否正确。
    """
    results = []
    timeout = aiohttp.ClientTimeout(total=60)  # 设置总超时时间为 60 秒
    semaphore = asyncio.Semaphore(64)  # 控制并发量为 64
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for sample in dataset:
            id = sample['id']
            config = {
                'language': None,
                'locale': "en",
                'compile_timeout': 20,
                'run_timeout': 20,
                'dataset_type': "CommonOJDataset"
            }
            key_list = ['id', 'content', 'test', 'labels', 'canonical_solution']
            provided_data = {}
            for key in key_list:
                provided_data[key] = sample[key]
            config['provided_data'] = provided_data
            solutions = sample['solutions']
            incorrect_solutions = sample['incorrect_solutions']

            async def process_sample(config, solutions, incorrect_solutions, api_path, id, dataset_type, session, semaphore):
                res = await solution_call(config, solutions, api_path, id, dataset_type, session, semaphore)
                incorrect_res = await solution_call(config, incorrect_solutions, api_path, id, dataset_type, session, semaphore)
                return {
                    'id': id,
                    'solution_result': res,
                    'incorrect_solution_result': incorrect_res,
                }

            tasks.append(
                asyncio.create_task(
                    process_sample(config, solutions, incorrect_solutions, api_path, id, dataset_type, session, semaphore)
                )
            )
        if tasks:
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="验证进度"):
                result = await coro
                results.append(result)
                # print(result)
                # 实时保存结果
                if results_path:
                    async with file_lock:
                        with open(results_path, "w") as f:
                            json.dump(results, f, indent=4)
    return results


if __name__ == "__main__":
    codecontests_data_path = "/aiarena/gpfs/code_contests"
    sandbox_api_path = "http://192.168.8.5:45649/submit"
    dataset_type = "code_contests_test"
    dataset = dataset_read(codecontests_data_path, "test")
    # 用 asyncio.run 正确调用异步函数
    file_lock = asyncio.Lock()
    results = asyncio.run(codecontests_call(dataset, sandbox_api_path, dataset_type, "codecontests_results.json"))
