"""
简化的串行版本的solution评估代码
将原来的并发多线程处理简化为串行处理，每个sample使用独立函数处理
"""

# %% set up
import os
import json
import requests
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import random
import time
import gc

completed_tasks = 0  # 全局计数器，用于跟踪已完成的任务数量
MAX_TIME = 1000
TEMPLATE = """
Here is a {language} solution to the problem: 
```{language}
{solution}
```
"""
LANGUAGE = ["UNKNOWN_LANGUAGE", "PYTHON", "CPP", "PYTHON3", "JAVA"]

# %% functions
# 读取 codecontests 数据集
def dataset_read(data_path, transform="codecontents", *args, **kwargs):
    print("load dataset from", data_path)
    data = load_from_disk(data_path)
    data = list(data)
    print(f"Loaded {len(data)} samples from {data_path} with transform {transform}")
    # 将其转化为 CommonOJ 格式
    format_data = []
    for sample in tqdm(data):
        if transform == "codecontents":
            test_cases = []
            test_cases_transform(sample['generated_tests'], test_cases)
            sample['test'] = test_cases
        elif transform == "codecontents_plus":
            test_cases = []
            if kwargs['split'] == '1x':
                test_cases_transform_codecontests_plus(sample['test_cases'], test_cases)
            elif kwargs['split'] == '3x':
                test_cases_transform_codecontests_plus(sample['3x_test_cases'], test_cases)
            elif kwargs['split'] == '5x':
                test_cases_transform_codecontests_plus(sample['5x_test_cases'], test_cases)
            sample['test'] = test_cases
        else:
            raise ValueError(f"Unknown transform type: {transform}")
        # 如果 format_sample 中的 test 是空列表，则跳过该 sample
        if not sample['test']:
            print(f"Sample {sample['id']} has no test cases, skipping.")
            continue
        format_data.append(sample)
    print(f"Transformed {len(format_data)} samples to CommonOJ format")
    return format_data


def test_cases_transform(test_cases, format_test_cases):
    for input_data, output_data in zip(test_cases['input'], test_cases['output']):
        format_test_cases.append({
            'input': {
                'stdin': input_data
            },
            'output': {
                'stdout': output_data
            }
            })

def test_cases_transform_codecontests_plus(test_cases, format_test_cases):
    """
    test_cases 的格式如下：
    [{'input': ..., 'output': ...}, {'input': ..., 'output': ...}]
    """
    for test_case in test_cases:
        if isinstance(test_case, dict) and 'input' in test_case and 'output' in test_case:
            format_test_cases.append({
                'input': {
                    'stdin': test_case['input']
                },
                'output': {
                    'stdout': test_case['output']
                }
            })
        else:
            raise ValueError(f"Invalid test case format: {test_case}")


SESSION = requests.Session()

def is_time_limit_exceeded(response):
    """
    检查响应是否包含 TimeLimitExceeded 状态
    返回值：
    - False: 没有 TimeLimitExceeded
    - "sandbox_blocked": Sandbox 内部阻塞，没有实际运行结果
    - "real_timeout": 代码真实运行超时，有部分输出结果
    """
    tests = response.get("tests", [])
    for test in tests:
        if test.get('exec_info', {}).get('status') == "Failed":
            compile_result = test.get('exec_info', {}).get('compile_result')
            run_result = test.get('exec_info', {}).get('run_result')
            if (isinstance(compile_result, dict) and compile_result.get('status') == "TimeLimitExceeded") or \
               (isinstance(run_result, dict) and run_result.get('status') == "TimeLimitExceeded"):
                # 检查是否有运行结果输出
                if isinstance(run_result, dict) and run_result.get('stdout'):
                    actual_stdout = run_result['stdout']
                    expected_stdout = test['test_info']['output']['stdout']
                    if actual_stdout not in expected_stdout:
                        return False  # 输出不匹配，可能程序逻辑有问题
                    return "real_timeout"  # 有输出，说明代码真实运行超时
                else:
                    return "sandbox_blocked"  # 无输出，说明是 sandbox 内部阻塞
    return False

def calculate_timeout_multiplier(actual_stdout, expected_stdout):
    """
    计算基于输出完整性的超时时间倍数
    
    Args:
        actual_stdout: 实际输出
        expected_stdout: 期望输出
        
    Returns:
        tuple: (is_partial, multiplier)
        - is_partial: 是否是部分输出
        - multiplier: 建议的超时时间倍数
    """
    if not actual_stdout or not expected_stdout:
        return False, 1
    
    actual_stdout = str(actual_stdout).strip()
    expected_stdout = str(expected_stdout).strip()
    
    # 检查实际输出是否是期望输出的前缀
    if actual_stdout and expected_stdout.startswith(actual_stdout):
        # 计算比例
        multiplier = len(expected_stdout) / len(actual_stdout)
        
        # 限制最大倍数避免过长等待
        multiplier = min(multiplier, 10)
        return True, multiplier
    else:
        # 输出不匹配，可能程序逻辑有问题
        return False, 1

def sandbox_call_simple(config, completion, api_path, id, dataset_type, max_retries=3, retry_delay=1):
    """
    简化版的sandbox API调用，只进行基本的重试
    
    Args:
        config: sandbox 配置
        completion: 代码补全内容
        api_path: API 路径
        id: 样本ID
        dataset_type: 数据集类型
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        
    Returns:
        dict: API响应结果
    """
    global MAX_TIME
    payload = {
        'dataset': dataset_type,
        'id': '',
        'completion': completion,
        'config': config
    }
    
    # 记录原始超时配置
    original_run_timeout = config.get('run_timeout', 20)
    original_compile_timeout = config.get('compile_timeout', 20)
    
    for attempt in range(max_retries + 1):
        try:
            with SESSION.post(
                api_path,
                json=payload,
            ) as res:
                response = res.json()
                
                # 检查是否有 TimeLimitExceeded，如果有则尝试增加超时时间重试一次
                tle_type = is_time_limit_exceeded(response)
                if tle_type == "real_timeout" and attempt == 0:
                    # 只在第一次尝试时进行超时时间调整
                    for test in response.get("tests", []):
                        exec_info = test.get('exec_info', {})
                        run_result = exec_info.get('run_result', {})
                        
                        if run_result.get('status') == 'TimeLimitExceeded':
                            actual_stdout = run_result.get('stdout', '')
                            
                            # 从config中获取期望输出
                            provided_data = config.get('provided_data', {})
                            test_cases = provided_data.get('test', [])
                            
                            # 找到对应的测试用例期望输出
                            if test_cases:
                                test_index = response.get("tests", []).index(test)
                                if test_index < len(test_cases):
                                    expected_stdout = test_cases[test_index].get('output', {}).get('stdout', '')
                                    
                                    is_partial, multiplier = calculate_timeout_multiplier(actual_stdout, expected_stdout)
                                    
                                    if is_partial:
                                        # 输出是部分的，增加超时时间重试
                                        new_run_timeout = int(original_run_timeout * multiplier)
                                        new_compile_timeout = int(original_compile_timeout * multiplier)
                                        
                                        # 检查是否超过最大超时限制
                                        if new_run_timeout <= MAX_TIME:
                                            config['run_timeout'] = new_run_timeout
                                            config['compile_timeout'] = new_compile_timeout
                                            payload['config'] = config
                                            
                                            print(f"Real timeout with partial output for {id}, increasing timeout by {multiplier}x to {new_run_timeout}s")
                                            continue
                                    break
                
                return response
                
        except Exception as e:
            if attempt < max_retries:
                print(f"Request failed for {id} on {api_path}, retrying... (attempt {attempt + 1}/{max_retries}): {e}")
                if retry_delay > 0:
                    time.sleep(retry_delay)
                continue
            else:
                print(f"Request failed for {id} on {api_path} after {max_retries} retries: {e}")
                return {'error': f'Exception: {str(e)}', 'api_path': api_path}
    
    return response

def process_single_sample(sample, api_path, dataset_type="code_contests_test"):
    """
    处理单个sample的所有solutions和incorrect_solutions
    
    Args:
        sample: 要处理的样本数据
        api_path: API端点URL
        dataset_type: 数据集类型
        
    Returns:
        dict: 包含所有处理结果的字典
    """
    sample_id = sample['id']
    solution_results = []
    incorrect_solution_results = []
    
    # 准备sample的基本配置
    key_list = ['id', 'content', 'test', 'labels', 'canonical_solution']
    provided_data = {key: sample[key] for key in key_list}
    
    print(f"Processing sample {sample_id}...")
    
    # 处理正确的solutions
    total_solutions = len([lang for lang in sample['solutions']['language'] if LANGUAGE[lang] != "UNKNOWN_LANGUAGE"])
    processed_solutions = 0
    
    for idx, (lang_idx, solution) in enumerate(zip(sample['solutions']['language'], sample['solutions']['solution'])):
        language = LANGUAGE[lang_idx]
        if language != "UNKNOWN_LANGUAGE":
            print(f"  Processing solution {processed_solutions + 1}/{total_solutions} ({language})...")
            
            # 统一PYTHON处理
            if "PYTHON" in language:
                language = "PYTHON"
            
            # 创建配置
            config = {
                'language': language.lower(),
                'locale': "en",
                'compile_timeout': 20,
                'run_timeout': 20,
                'dataset_type': "CommonOJDataset",
                'provided_data': provided_data
            }
            
            # 创建completion
            completion = TEMPLATE.format(language=language.lower(), solution=solution)
            
            # 调用sandbox API
            resp = sandbox_call_simple(config, completion, api_path, sample_id, dataset_type)
            
            solution_result = {
                'language': LANGUAGE[lang_idx],
                'solution': solution,
                'result': resp
            }
            solution_results.append(solution_result)
            processed_solutions += 1
    
    # 处理错误的solutions
    total_incorrect_solutions = len([lang for lang in sample['incorrect_solutions']['language'] if LANGUAGE[lang] != "UNKNOWN_LANGUAGE"])
    processed_incorrect_solutions = 0
    
    for idx, (lang_idx, solution) in enumerate(zip(sample['incorrect_solutions']['language'], sample['incorrect_solutions']['solution'])):
        language = LANGUAGE[lang_idx]
        if language != "UNKNOWN_LANGUAGE":
            print(f"  Processing incorrect solution {processed_incorrect_solutions + 1}/{total_incorrect_solutions} ({language})...")
            
            # 统一PYTHON处理
            if "PYTHON" in language:
                language = "PYTHON"
            
            # 创建配置
            config = {
                'language': language.lower(),
                'locale': "en",
                'compile_timeout': 20,
                'run_timeout': 20,
                'dataset_type': "CommonOJDataset",
                'provided_data': provided_data
            }
            
            # 创建completion
            completion = TEMPLATE.format(language=language.lower(), solution=solution)
            
            # 调用sandbox API
            resp = sandbox_call_simple(config, completion, api_path, sample_id, dataset_type)
            
            solution_result = {
                'language': LANGUAGE[lang_idx],
                'solution': solution,
                'result': resp
            }
            incorrect_solution_results.append(solution_result)
            processed_incorrect_solutions += 1
    
    # 构建最终结果
    result = {
        'id': sample_id,
        'solution_result': solution_results,
        'incorrect_solution_result': incorrect_solution_results,
        'api_used': api_path,
        'progress': {
            'completed_solutions': processed_solutions,
            'total_solutions': total_solutions,
            'completed_incorrect_solutions': processed_incorrect_solutions,
            'total_incorrect_solutions': total_incorrect_solutions,
            'is_complete': True
        }
    }
    
    print(f"Completed sample {sample_id}: {processed_solutions} solutions, {processed_incorrect_solutions} incorrect solutions")
    return result

def codecontests_call_serial(dataset, api_path, dataset_type="code_contests_test", results_path=None):
    """
    串行版本的codecontests处理函数
    
    Args:
        dataset: 要处理的数据集
        api_path: 单个API端点URL
        dataset_type: 数据集类型
        results_path: 结果保存路径（文件夹）
        
    Returns:
        list: 所有处理结果的列表
    """
    # 创建结果文件夹
    if results_path:
        os.makedirs(results_path, exist_ok=True)
    
    results = []
    
    # 检查已有结果
    existing_results = set()
    if results_path:
        for fname in os.listdir(results_path):
            if fname.endswith('.json'):
                sample_id = os.path.splitext(fname)[0].replace('_', '/')
                existing_results.add(sample_id)
    
    # 筛选需要处理的样本
    samples_to_process = [sample for sample in dataset if sample['id'] not in existing_results]
    
    if not samples_to_process:
        print("All samples have been processed. No new tasks to run.")
        return results
    
    print(f"Found {len(samples_to_process)} samples to process out of {len(dataset)} total.")
    print(f"Using API endpoint: {api_path}")
    
    # 串行处理每个sample
    for i, sample in enumerate(tqdm(samples_to_process, desc="Processing samples")):
        try:
            # 处理单个sample
            result = process_single_sample(sample, api_path, dataset_type)
            results.append(result)
            
            # 保存结果到文件
            if results_path:
                safe_sample_id = sample['id'].replace('/', '_')
                result_path = os.path.join(results_path, f"{safe_sample_id}.json")
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=4)
                print(f"Saved result for sample {sample['id']} ({i + 1}/{len(samples_to_process)})")
            
            # 清理内存
            if i % 10 == 0:  # 每处理10个样本清理一次
                gc.collect()
                
        except Exception as e:
            print(f"Error processing sample {sample['id']}: {e}")
            # 创建错误结果
            error_result = {
                'id': sample['id'],
                'solution_result': [],
                'incorrect_solution_result': [],
                'api_used': api_path,
                'error': str(e),
                'progress': {
                    'completed_solutions': 0,
                    'total_solutions': 0,
                    'completed_incorrect_solutions': 0,
                    'total_incorrect_solutions': 0,
                    'is_complete': False
                }
            }
            results.append(error_result)
            
            # 保存错误结果
            if results_path:
                safe_sample_id = sample['id'].replace('/', '_')
                result_path = os.path.join(results_path, f"{safe_sample_id}.json")
                with open(result_path, "w") as f:
                    json.dump(error_result, f, indent=4)
    
    print(f"Serial processing completed. Processed {len(results)} samples.")
    return results

# %% main
if __name__ == "__main__":
    codecontests_data_path = "/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_gpt5_new_replace_add_feedback_gen_command_replace_new_new/"
    # 使用单个sandbox API端点
    sandbox_api_path = "http://10.244.166.255:8080/submit"
    
    SESSION.mount('http://', requests.adapters.HTTPAdapter(pool_maxsize=1))
    data_type = "test"
    dataset_type = f"code_contests_{data_type}"

    def get_corner_cases_with_first(sample):
        """
        获取样本的 corner cases
        """
        all_results = sample['result']
        corner_cases = all_results[0]['corner_cases'] if all_results else []
        return corner_cases

    file_paths = os.listdir(codecontests_data_path)
    dataset = []
    for file_path in tqdm(file_paths):
        if file_path.endswith('.json'):
            file_path = os.path.join(codecontests_data_path, file_path)
            with open(file_path, 'r') as f:
                sample = json.load(f)
                sample['test'] = sample['corner_cases']
                sample['solutions'] = sample['filter_solutions']
                sample['incorrect_solutions'] = sample['filter_incorrect_solutions']
                assert 'id' in sample, f"Sample {file_path} does not have 'id' field"
                assert 'content' in sample, f"Sample {file_path} does not have 'content' field"
                assert 'test' in sample, f"Sample {file_path} does not have 'test' field"
                assert 'labels' in sample, f"Sample {file_path} does not have 'labels' field"
                assert 'canonical_solution' in sample, f"Sample {file_path} does not have 'canonical_solution' field"
                if sample['corner_cases']:
                    dataset.append(sample)
                else:
                    print(f"Sample {file_path} has no corner cases, skipping")
        else:
            print(f"File {file_path} is not a JSON file, skipping")
    
    start = 0
    end = len(dataset)
    end = min(end, len(dataset))
    print(f"Total samples: {len(dataset)}")
    n = end - start
    # 采样 start 到 end 条数据
    dataset = dataset[start:end]

    results_path = f"/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_serial_test/"
    time_consume_path = f"/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_serial_test.txt"
    
    # 获取初始时间
    start_time = time.time()
    results = codecontests_call_serial(dataset, sandbox_api_path, dataset_type, results_path)
    # 获取结束时间
    end_time = time.time()
    # 计算总耗时
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    
    # 将总耗时写入结果文件
    with open(time_consume_path, "w") as f:
        f.write(f"Total processing time: {total_time:.2f} seconds; sample number: {len(dataset)}\n")