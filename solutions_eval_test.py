"""
该代码的主要流程是:
1. 设计一个 Prompt 来引导 LLM 生成特定的 corner case;
2. 调用 OpenAI 的 API 生成 corner case;
3. 调用远程的 sandbox API 来验证生成的 corner case 是否正确:
    3.1: 将生成的 corner case 和给定的 solution/incorrect_solution 一起提交到 sandbox;
    3.2: sandbox 会运行给定的 solution/incorrect_solution 并返回结果;
    3.3: 根据 sandbox 返回的结果来判断 corner case 是否正确;
4. 如果 corner case 不正确, 则重新设计 Prompt 并生成新的 corner case, 直到生成正确的 corner case 或达到最大尝试次数为止。
"""

# %% set up
import os
import json
import requests
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import time
import queue
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
    # if transform == "codecontents_plus":
    #     # data = load_from_disk(data_path)
    #     data = load_dataset(data_path, '1x')
    #     data = data[kwargs['split']] if 'split' in kwargs else data['train']
    # elif transform == "codecontents":
    #     data = load_dataset(data_path, *args, **kwargs)
    # else:
    #     raise ValueError(f"Unknown transform type: {transform}")
    data = load_from_disk(data_path)
    # data = data[kwargs['split']] if 'split' in kwargs and kwargs['split'] else data
    data = list(data)
    print(f"Loaded {len(data)} samples from {data_path} with transform {transform}")
    # 将其转化为 CommonOJ 格式
    format_data = []
    for sample in tqdm(data):
        # if transform == "codecontents":
        #     format_sample = transform_codecontents(sample)
        # elif transform == "codecontents_plus":
        #     format_sample = transform_codecontents_plus(sample)
        # else:
        if transform == "codecontents":
            # sample['test'] = sample['original_test']
            test_cases = []
            test_cases_transform(sample['generated_tests'], test_cases)
            # test_cases_transform(sample['public_tests'], test_cases)
            # test_cases_transform(sample['private_tests'], test_cases)
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
    # return data


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
    test_cases_transform(sample['public_tests'], test_cases)
    test_cases_transform(sample['private_tests'], test_cases)
    test_cases_transform(sample['generated_tests'], test_cases)

    format_sample['test'] = test_cases
    return format_sample


def transform_codecontents_plus(sample):
    """
    ['source', 'id', 'title', 'description', 'time_limit', 'memory_limit', 'validator', 'generator', 'generator_cmd', 'checker', 'correct_submissions', 'incorrect_submissions', 'test_cases', 'true_positive_rate', 'true_negative_rate'] -> ['id', 'content', 'test', 'labels', 'canonical_solution']
    """
    format_sample = {}
    for key, value in sample.items():
        format_sample[key] = value
    format_sample['id'] = sample['id']
    format_sample['content'] = '# ' + sample['title'].strip() + '\n\n' + sample['description']
    format_sample['labels'] = {
        "tag": [],
        "title": sample['title'].strip()
    }
    
    canonical_solution = {}
    # sample['correct_submissions'] 的格式如下:
    # [{'code': "...", 'language': language}]
    # 其中 language 取值为 ['cpp', 'py2', 'java', 'py3']
    for submission in sample['correct_submissions']:
        language = submission['language']
        if language == 'py2' or language == 'py3':
            language = 'python'
        elif language == 'cpp':
            language = 'cpp'
        elif language == 'java':
            language = 'java'
        else:
            continue
        if language not in canonical_solution:
            canonical_solution[language] = submission['code']
        else:
            # 如果已经存在该语言的代码，则不覆盖
            continue
    
    format_sample['canonical_solution'] = canonical_solution
    format_sample['solutions'] = submission_transform_codecontests_plus(sample['correct_submissions'])
    format_sample['incorrect_solutions'] = submission_transform_codecontests_plus(sample['incorrect_submissions'])
    test_cases = []
    test_cases_transform_codecontests_plus(sample['test_cases'], test_cases)

    format_sample['test'] = test_cases
    return format_sample


def submission_transform_codecontests_plus(submissions):
    """
    submission 的格式如下:
    {'code': '...', 'language': 'py3'}
    需要转化为 (language_index, code) 的形式
    """
    transformed = {"language": [], "solution": []}
    for submission in submissions:
        language = submission['language']
        if language == 'py2' or language == 'py3':
            language = 'PYTHON'
        elif language == 'cpp':
            language = 'CPP'
        elif language == 'java':
            language = 'JAVA'
        else:
            language = 'UNKNOWN_LANGUAGE'

        if language in LANGUAGE:
            language_index = LANGUAGE.index(language)
        else:
            language_index = 0  # UNKNOWN_LANGUAGE

        transformed['language'].append(language_index)
        transformed['solution'].append(submission['code'])
    return transformed

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

def has_time_limit_exceeded_in_results(result_file_path):
    """
    检查结果文件中是否包含 TimeLimitExceeded 状态
    返回需要重新测试的 solution 和 incorrect_solution 的索引列表，以及建议的超时配置
    """
    global MAX_TIME
    try:
        with open(result_file_path, 'r') as f:
            result_data = json.load(f)
        
        retry_solutions = []  # 需要重新测试的 solution 索引
        retry_incorrect_solutions = []  # 需要重新测试的 incorrect_solution 索引
        timeout_configs = {}  # 存储每个需要重试的solution的建议超时配置
        
        # 检查 solution_result
        for idx, solution_result in enumerate(result_data.get('solution_result', [])):
            result = solution_result.get('result', {})
            tle_type = is_time_limit_exceeded(result)
            if tle_type:
                should_retry = True
                suggested_timeout = None
                
                if tle_type == "real_timeout":
                    # 分析real_timeout的具体情况
                    for test in result.get("tests", []):
                        exec_info = test.get('exec_info', {})
                        run_result = exec_info.get('run_result', {})
                        
                        if run_result.get('status') == 'TimeLimitExceeded':
                            actual_stdout = run_result.get('stdout', '')
                            
                            # 更精确的判断：只有非空输出才考虑重试
                            if actual_stdout and actual_stdout.strip():
                                # 有输出，说明程序在运行，可能需要更多时间
                                should_retry = True
                                output_length = len(actual_stdout.strip())
                                expected_length = len(test['test_info']['output']['stdout'])
                                run_time = run_result['execution_time']
                                suggested_timeout = run_time * (expected_length / output_length) if output_length > 0 else run_time
                                
                                if suggested_timeout > MAX_TIME:
                                    should_retry = False
                            else:
                                # 无输出的 real_timeout，实际上可能是 sandbox 问题，跳过
                                should_retry = False
                                break
                elif tle_type == "sandbox_blocked":
                    # sandbox阻塞，应该重试
                    should_retry = True
                
                if should_retry:
                    retry_solutions.append(idx)
                    if suggested_timeout:
                        timeout_configs[f'solution_{idx}'] = {
                            'run_timeout': suggested_timeout,
                            'compile_timeout': suggested_timeout
                        }
        
        # 检查 incorrect_solution_result  
        for idx, incorrect_result in enumerate(result_data.get('incorrect_solution_result', [])):
            result = incorrect_result.get('result', {})
            tle_type = is_time_limit_exceeded(result)
            if tle_type:
                should_retry = True
                suggested_timeout = None
                
                if tle_type == "real_timeout":
                    # 类似的分析逻辑
                    for test in result.get("tests", []):
                        exec_info = test.get('exec_info', {})
                        run_result = exec_info.get('run_result', {})
                        
                        if run_result.get('status') == 'TimeLimitExceeded':
                            actual_stdout = run_result.get('stdout', '')
                            
                            # 更精确的判断：只有非空输出才考虑重试
                            if actual_stdout and actual_stdout.strip():
                                # 有输出，说明程序在运行，可能需要更多时间
                                should_retry = True
                                output_length = len(actual_stdout.strip())
                                expected_length = len(test['test_info']['output']['stdout'])
                                run_time = run_result['execution_time']
                                suggested_timeout = run_time * (expected_length / output_length) if output_length > 0 else run_time
                                
                                if suggested_timeout > MAX_TIME:
                                    should_retry = False
                            else:
                                # 无输出的 real_timeout，实际上可能是 sandbox 问题，跳过
                                should_retry = False
                                break
                elif tle_type == "sandbox_blocked":
                    # sandbox阻塞，应该重试
                    should_retry = True
                
                if should_retry:
                    retry_incorrect_solutions.append(idx)
                    if suggested_timeout:
                        timeout_configs[f'incorrect_solution_{idx}'] = {
                            'run_timeout': suggested_timeout,
                            'compile_timeout': suggested_timeout
                        }
        
        return retry_solutions, retry_incorrect_solutions, timeout_configs
        
    except Exception as e:
        print(f"Error reading result file {result_file_path}: {e}")
        return [], [], {}

def shuffle_queue_safely(task_queue, queue_lock):
    """
    安全地打乱队列中的任务顺序
    
    Args:
        task_queue: 要打乱的任务队列
        queue_lock: 队列操作锁
    """
    with queue_lock:
        # 将队列中的所有任务取出
        tasks = []
        while not task_queue.empty():
            try:
                tasks.append(task_queue.get_nowait())
            except queue.Empty:
                break
        
        # 只有当任务数量大于1时才打乱
        if len(tasks) > 1:
            # 打乱任务顺序
            random.shuffle(tasks)
            
            # 将任务重新放回队列
            for task in tasks:
                task_queue.put(task)
            
            print(f"Shuffled {len(tasks)} tasks in queue to improve load balancing")
        else:
            # 如果只有一个或没有任务，直接放回
            for task in tasks:
                task_queue.put(task)

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

def sandbox_call(config, completion, api_path, id, dataset_type, max_retries=3, retry_delay=1):
    """
    调用远程的 sandbox API 来验证生成的 corner case 是否正确。
    当遇到 TimeLimitExceeded 时会进行智能重试。
    
    Args:
        config: sandbox 配置
        completion: 代码补全内容
        api_path: API 路径
        id: 样本ID
        dataset_type: 数据集类型
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        
    Returns:
        dict: 正常响应或包含特殊状态的字典
        如果返回 {'status': 'REQUEUE_NEEDED', 'reason': 'TimeLimitExceeded'} 表示需要重新入队
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
    timeout_increase_attempts = 0
    max_timeout_attempts = 3
    
    for attempt in range(max_retries + 1):
        try:
            with SESSION.post(
                api_path,
                json=payload,
            ) as res:
                response = res.json()
                
                # 检查是否有 TimeLimitExceeded
                tle_type = is_time_limit_exceeded(response)
                if tle_type:
                    if tle_type == "real_timeout" and timeout_increase_attempts < max_timeout_attempts:
                        # 智能分析输出并调整超时时间
                        should_retry = False
                        
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
                                    # 假设按顺序匹配测试用例
                                    test_index = response.get("tests", []).index(test)
                                    if test_index < len(test_cases):
                                        expected_stdout = test_cases[test_index].get('output', {}).get('stdout', '')
                                        
                                        is_partial, multiplier = calculate_timeout_multiplier(actual_stdout, expected_stdout)
                                        
                                        if is_partial:
                                            # 输出是部分的，增加超时时间重试
                                            new_run_timeout = int(original_run_timeout * multiplier)
                                            new_compile_timeout = int(original_compile_timeout * multiplier)
                                            
                                            config['run_timeout'] = new_run_timeout
                                            config['compile_timeout'] = new_compile_timeout
                                            payload['config'] = config
                                            
                                            print(f"Real timeout with partial output for {id} on {api_path}, increasing timeout by {multiplier}x to {new_run_timeout}s")
                                            timeout_increase_attempts += 1
                                            should_retry = True
                                            # 检查是否超过最大超时限制
                                            
                                            if new_run_timeout > MAX_TIME:
                                                should_retry = False
                                            break
                                        else:
                                            # 输出不匹配，可能程序逻辑有问题，不重试
                                            print(f"Real timeout with mismatched output for {id} on {api_path}, not retrying")
                                            return response
                        
                        if should_retry:
                            continue
                        else:
                            # 没有找到合适的重试条件，返回原响应
                            return response
                            
                    elif tle_type == "sandbox_blocked":
                        # Sandbox 内部阻塞，正常重试
                        if attempt < max_retries:
                            print(f"Sandbox blocked detected for {id} on {api_path}, retrying... (attempt {attempt + 1}/{max_retries})")
                            if retry_delay > 0:
                                time.sleep(retry_delay)
                            continue
                        else:
                            print(f"Sandbox blocking persists for {id} on {api_path} after {max_retries} retries, will requeue")
                            return {'status': 'REQUEUE_NEEDED', 'reason': 'SandboxBlocked', 'api_path': api_path}
                    else:
                        # 其他超时情况或已达到最大超时尝试次数
                        print(f"TimeLimitExceeded persists for {id} on {api_path} after attempts, will requeue")
                        return {'status': 'REQUEUE_NEEDED', 'reason': 'TimeLimitExceeded', 'api_path': api_path}
                
                return response
                
        except Exception as e:
            if attempt < max_retries:
                print(f"Request failed for {id} on {api_path}, retrying... (attempt {attempt + 1}/{max_retries}): {e}")
                if retry_delay > 0:
                    time.sleep(retry_delay)
                continue
            else:
                print(f"Request failed for {id} on {api_path} after {max_retries} retries, will requeue: {e}")
                return {'status': 'REQUEUE_NEEDED', 'reason': f'Exception: {str(e)}', 'api_path': api_path}
    
    return response

def sandbox_call_original(config, completion, api_path, id, dataset_type):
    """
    调用远程的 sandbox API 来验证生成的 corner case 是否正确。
    """
    payload = {
        'dataset': dataset_type,
        'id': '',
        'completion': completion,
        'config': config
    }
    with SESSION.post(
        api_path,
        json=payload,
    ) as res:
        return res.json()

def codecontests_call(dataset, api_paths, dataset_type="code_contests_valid", results_path=None, max_workers=1, debug=False, save_one_step=False):
    """
    并发调用远程的 sandbox API 来验证生成的 corner case 是否正确。
    使用任务队列实现动态负载均衡的多 API 并行调用。
    现在队列的每个元素为每个 solution/incorrect_solution，实现 solution 级别的并行。
    
    Args:
        max_workers: solution 级别的并行度，总并行线程数
        debug: 是否启用调试模式，会输出任务分配信息
        save_one_step: 是否每完成一个任务就保存一次 (True) 或等到整个 sample 完成后再保存 (False)
    """

    # results_path 现在是结果文件夹
    results_dir = results_path
    os.makedirs(results_dir, exist_ok=True)
    # 不需要全局的 results 列表，改为每个 sample 的结果单独存储
    # results = []
    # 已有的 sample id 和需要重新测试的信息
    existing_results = {}  # 存储已有结果和需要重新测试的信息
    for fname in os.listdir(results_dir):
        if fname.endswith('.json'):
            sample_id = os.path.splitext(fname)[0].replace('_', '/')
            result_file_path = os.path.join(results_dir, fname)
            if not os.path.exists(result_file_path):
                continue
            
            try:
                with open(result_file_path, 'r') as f:
                    existing_data = json.load(f)
                
                # 检查是否有进度信息，判断任务是否完全完成
                progress = existing_data.get('progress', {})
                is_complete = progress.get('is_complete', False)
                
                # 如果任务未完成，或者有 TimeLimitExceeded 需要重试
                retry_solutions, retry_incorrect_solutions, timeout_configs = has_time_limit_exceeded_in_results(result_file_path)
                
                # 如果任务未完成，需要标记为需要继续处理
                if not is_complete:
                    print(f"Found incomplete sample {sample_id}: {progress.get('completed_solutions', 0)}/{progress.get('total_solutions', 0)} solutions, {progress.get('completed_incorrect_solutions', 0)}/{progress.get('total_incorrect_solutions', 0)} incorrect_solutions")
                
                existing_results[sample_id] = {
                    'result_file_path': result_file_path,
                    'retry_solutions': retry_solutions,
                    'retry_incorrect_solutions': retry_incorrect_solutions,
                    'is_complete': is_complete,
                    'existing_data': existing_data,
                    'timeout_configs': timeout_configs
                }
            except Exception as e:
                print(f"Error reading existing result file {result_file_path}: {e}")
                # 如果读取失败，标记为需要重新处理
                existing_results[sample_id] = {
                    'result_file_path': result_file_path,
                    'retry_solutions': [],
                    'retry_incorrect_solutions': [],
                    'is_complete': False,
                    'existing_data': None,
                    'timeout_configs': {}
                }

    results_lock = threading.Lock()  # 用于保护结果列表的线程锁
    solution_results = {}  # 用于收集每个sample的solution结果
    solution_results_lock = threading.Lock()  # 保护solution_results的线程锁
    
    # 筛选出需要处理的样本（包括新样本、未完成的样本和需要重新测试的样本）
    samples_to_process = []
    for sample in dataset:
        sample_id = sample['id']
        if sample_id not in existing_results:
            # 新样本，完全重新处理
            samples_to_process.append(sample)
        else:
            # 已有结果的样本，检查是否需要继续处理
            existing_info = existing_results[sample_id]
            if (not existing_info['is_complete'] or 
                existing_info['retry_solutions'] or 
                existing_info['retry_incorrect_solutions']):
                # 任务未完成或需要重新测试部分 solution/incorrect_solution
                samples_to_process.append(sample)
    
    if not samples_to_process:
        print("All samples have been completely processed and no TimeLimitExceeded cases need retry. No new tasks to run.")
        return None

    # 创建任务队列 - 现在队列元素为每个solution/incorrect_solution
    task_queue = queue.Queue()
    task_queue_lock = threading.Lock()  # 添加队列操作锁，用于打乱队列时的线程安全
    total_solutions = 0
    sample_pool = {}
    for sample in samples_to_process:
        sample_id = sample['id']
        sample_pool[sample_id] = sample
        
        # 检查是否是已有结果的样本
        is_existing_sample = sample_id in existing_results
        existing_info = existing_results.get(sample_id, {})
        retry_solutions = existing_info.get('retry_solutions', [])
        retry_incorrect_solutions = existing_info.get('retry_incorrect_solutions', [])
        existing_data = existing_info.get('existing_data')
        timeout_configs = existing_info.get('timeout_configs', {})
        
        if is_existing_sample and existing_data:
            # 已有结果的样本，先加载现有结果
            existing_solution_results = existing_data.get('solution_result', [])
            existing_incorrect_results = existing_data.get('incorrect_solution_result', [])
            
            # 获取进度信息
            progress = existing_data.get('progress', {})
            completed_solutions = progress.get('completed_solutions', 0)
            completed_incorrect_solutions = progress.get('completed_incorrect_solutions', 0)
        else:
            existing_solution_results = []
            existing_incorrect_results = []
            completed_solutions = 0
            completed_incorrect_solutions = 0
        
        # 为每个sample初始化结果存储
        solution_results[sample_id] = {
            'sample': sample,
            'solution_result': existing_solution_results.copy(),
            'incorrect_solution_result': existing_incorrect_results.copy(),
            'completed_solutions': completed_solutions,
            'completed_incorrect_solutions': completed_incorrect_solutions,
            'total_solutions': len([lang for lang in sample['solutions']['language'] if LANGUAGE[lang] != "UNKNOWN_LANGUAGE"]),
            'total_incorrect_solutions': len([lang for lang in sample['incorrect_solutions']['language'] if LANGUAGE[lang] != "UNKNOWN_LANGUAGE"]),
            'retry_solutions': retry_solutions,
            'retry_incorrect_solutions': retry_incorrect_solutions
        }
        
        # 添加正确的solutions到队列
        for idx, (lang_idx, solution) in enumerate(zip(sample['solutions']['language'], sample['solutions']['solution'])):
            language = LANGUAGE[lang_idx]
            if language != "UNKNOWN_LANGUAGE":
                # 确定是否需要处理这个solution
                need_to_process = False
                if not is_existing_sample:
                    # 新样本，需要处理所有solution
                    need_to_process = True
                elif idx in retry_solutions:
                    # 需要重新测试的solution
                    need_to_process = True
                elif idx >= len(existing_solution_results):
                    # 之前未完成的solution（索引超出已有结果）
                    need_to_process = True
                
                if need_to_process:
                    # 检查是否有建议的超时配置
                    suggested_timeout = timeout_configs.get(f'solution_{idx}')
                    
                    task_queue.put({
                        'type': 'solution',
                        'sample_id': sample_id,
                        'language_index': lang_idx,
                        'language': language,
                        'solution': solution,
                        'solution_index': idx,  # 添加索引用于更新结果
                        'retry_count': 0,  # 添加重试计数
                        'suggested_timeout': suggested_timeout  # 添加建议的超时配置
                        # 'sample': sample
                    })
                    total_solutions += 1
                else:
                    # 不需要重新处理，已有的任务已完成
                    pass  # completed_solutions 已经在加载时设置
        
        # 添加错误的solutions到队列  
        for idx, (lang_idx, solution) in enumerate(zip(sample['incorrect_solutions']['language'], sample['incorrect_solutions']['solution'])):
            language = LANGUAGE[lang_idx]
            if language != "UNKNOWN_LANGUAGE":
                # 确定是否需要处理这个incorrect_solution
                need_to_process = False
                if not is_existing_sample:
                    # 新样本，需要处理所有incorrect_solution
                    need_to_process = True
                elif idx in retry_incorrect_solutions:
                    # 需要重新测试的incorrect_solution
                    need_to_process = True
                elif idx >= len(existing_incorrect_results):
                    # 之前未完成的incorrect_solution（索引超出已有结果）
                    need_to_process = True
                
                if need_to_process:
                    # 检查是否有建议的超时配置
                    suggested_timeout = timeout_configs.get(f'incorrect_solution_{idx}')
                    
                    task_queue.put({
                        'type': 'incorrect_solution',
                        'sample_id': sample_id,
                        'language_index': lang_idx,
                        'language': language,
                        'solution': solution,
                        'solution_index': idx,  # 添加索引用于更新结果
                        'retry_count': 0,  # 添加重试计数
                        'suggested_timeout': suggested_timeout  # 添加建议的超时配置
                        # 'sample': sample
                    })
                    total_solutions += 1
                else:
                    # 不需要重新处理，已有的任务已完成
                    pass  # completed_incorrect_solutions 已经在加载时设置
    
    print(f"Found {len(samples_to_process)} samples to process out of {len(dataset)} total.")
    
    # 统计重新测试的信息
    new_samples = 0
    retry_samples = 0
    total_retry_solutions = 0
    total_retry_incorrect_solutions = 0
    
    for sample in samples_to_process:
        sample_id = sample['id']
        if sample_id not in existing_results:
            new_samples += 1
        else:
            retry_samples += 1
            existing_info = existing_results[sample_id]
            total_retry_solutions += len(existing_info['retry_solutions'])
            total_retry_incorrect_solutions += len(existing_info['retry_incorrect_solutions'])
    
    print(f"New samples: {new_samples}")
    print(f"Samples with TimeLimitExceeded to retry: {retry_samples}")
    if retry_samples > 0:
        print(f"  - Solutions to retry: {total_retry_solutions}")
        print(f"  - Incorrect solutions to retry: {total_retry_incorrect_solutions}")
    print(f"Total solutions/incorrect_solutions to process: {total_solutions}")
    # print(f"Using {len(api_paths)} API endpoints: {api_paths}")
    print(f"Total parallel workers: {len(api_paths) * max_workers}")

    # 用于调试的任务分配跟踪
    if debug:
        task_assignment_lock = threading.Lock()
        assigned_tasks = set()

    def api_worker(api_path, worker_id):
        """每个API的工作函数，从任务队列中获取任务并处理"""
        processed_count = 0
        # 在函数开头定义全局计数器
        global completed_tasks
        while True:
            try:
                # 使用锁来安全获取任务，避免与shuffle操作冲突
                with task_queue_lock:
                    task = task_queue.get(timeout=1)
                if debug:
                    with task_assignment_lock:
                        assigned_tasks.add(f"{task['sample_id']}_{task['type']}_{task['language']}")
            except queue.Empty:
                break
            try:
                # sample = task['sample']
                sample = sample_pool[task['sample_id']]
                language = task['language']
                if "PYTHON" in language:
                    language = "PYTHON"
                # 为每个任务创建独立的配置副本，避免修改影响其他任务
                config = {
                    'language': language.lower(),
                    'locale': "en",
                    'compile_timeout': 20,
                    'run_timeout': 20,
                    'dataset_type': "CommonOJDataset"
                }
                
                # 应用建议的超时配置（如果有的话）
                suggested_timeout = task.get('suggested_timeout')
                if suggested_timeout:
                    config['compile_timeout'] = suggested_timeout.get('compile_timeout', 20)
                    config['run_timeout'] = suggested_timeout.get('run_timeout', 20)
                    print(f"Using suggested timeout for {task['sample_id']}_{task['type']}: {config['run_timeout']}s")
                
                key_list = ['id', 'content', 'test', 'labels', 'canonical_solution']
                provided_data = {key: sample[key] for key in key_list}
                config['provided_data'] = provided_data
                completion = TEMPLATE.format(language=language.lower(), solution=task['solution'])
                resp = sandbox_call(config, completion, api_path, task['sample_id'], dataset_type)
                
                # 检查是否需要重新入队
                if isinstance(resp, dict) and resp.get('status') == 'REQUEUE_NEEDED':
                    max_requeue_attempts = 3  # 最大重新入队次数
                    retry_count = task.get('retry_count', 0)
                    
                    if retry_count < max_requeue_attempts:
                        print(f"Requeuing task for {task['sample_id']}_{task['type']} due to {resp.get('reason', 'unknown reason')} on {resp.get('api_path', api_path)} (attempt {retry_count + 1}/{max_requeue_attempts})")
                        # 增加重试计数并重新放回队列
                        task['retry_count'] = retry_count + 1
                        with task_queue_lock:
                            task_queue.put(task)
                        
                        # 在重新入队后打乱队列，提高负载均衡
                        # 同时检查队列大小，避免在队列很小时频繁打乱
                        if task_queue.qsize() > 5:
                            shuffle_queue_safely(task_queue, task_queue_lock)
                        
                        continue
                    else:
                        print(f"Max requeue attempts reached for {task['sample_id']}_{task['type']}, using last response")
                        # 达到最大重试次数，调用 sandbox_call_original 获取最后的响应
                        resp = sandbox_call_original(config, completion, api_path, task['sample_id'], dataset_type)
                
                solution_result = {
                    'language': task['language'],
                    'solution': task['solution'],
                    'result': resp
                }
                # 修改任务完成时的计数逻辑，支持重新测试的情况，每完成一个任务就保存
                with solution_results_lock:
                    sample_id = task['sample_id']
                    sr = solution_results[sample_id]
                    solution_index = task.get('solution_index')
                    
                    if task['type'] == 'solution':
                        # 检查是否是重新测试
                        if solution_index is not None and solution_index in sr.get('retry_solutions', []):
                            # 替换原有结果
                            sr['solution_result'][solution_index] = solution_result
                        else:
                            # 新增结果
                            sr['solution_result'].append(solution_result)
                        sr['completed_solutions'] += 1
                    else:  # incorrect_solution
                        # 检查是否是重新测试
                        if solution_index is not None and solution_index in sr.get('retry_incorrect_solutions', []):
                            # 替换原有结果
                            sr['incorrect_solution_result'][solution_index] = solution_result
                        else:
                            # 新增结果
                            sr['incorrect_solution_result'].append(solution_result)
                        sr['completed_incorrect_solutions'] += 1
                    
                    # 根据 save_one_step 参数决定保存策略
                    is_sample_complete = (sr['completed_solutions'] == sr['total_solutions'] and 
                                        sr['completed_incorrect_solutions'] == sr['total_incorrect_solutions'])
                    
                    result = {
                        'id': sample_id,
                        'solution_result': sr['solution_result'],
                        'incorrect_solution_result': sr['incorrect_solution_result'],
                        'api_used': api_path,
                        'worker_id': worker_id,
                        'progress': {
                            'completed_solutions': sr['completed_solutions'],
                            'total_solutions': sr['total_solutions'],
                            'completed_incorrect_solutions': sr['completed_incorrect_solutions'],
                            'total_incorrect_solutions': sr['total_incorrect_solutions'],
                            'is_complete': is_sample_complete
                        }
                    }
                    
                    # 决定是否保存
                    should_save = False
                    if save_one_step:
                        # 每完成一个任务就保存一次
                        should_save = True
                    else:
                        # 只有当整个 sample 完成时才保存
                        should_save = is_sample_complete
                    
                    if should_save:
                        with results_lock:
                            # 保存当前进度到文件
                            safe_sample_id = sample_id.replace('/', '_')
                            result_path = os.path.join(results_dir, f"{safe_sample_id}.json")
                            with open(result_path, "w") as f:
                                json.dump(result, f, indent=4)
                            
                            if save_one_step:
                                print(f"Saved progress for {sample_id}: {sr['completed_solutions']}/{sr['total_solutions']} solutions, {sr['completed_incorrect_solutions']}/{sr['total_incorrect_solutions']} incorrect_solutions")
                            else:
                                print(f"Sample {sample_id} completed, saved final result: {sr['completed_solutions']}/{sr['total_solutions']} solutions, {sr['completed_incorrect_solutions']}/{sr['total_incorrect_solutions']} incorrect_solutions")
                    
                    # 只有当所有任务都完成时才清理内存
                    if is_sample_complete:
                        if not save_one_step:
                            print(f"Sample {sample_id} completed all tasks, cleaning up memory")
                        else:
                            print(f"Sample {sample_id} completed all tasks, cleaning up memory")
                        # 删除对应的 solution_results[sample_id]，避免内存泄漏
                        del solution_results[sample_id]
                        gc.collect()
                
                # 只有任务成功完成时才更新计数器（不包括重新入队的情况）
                completed_tasks += 1  # 实时更新计数器
                processed_count += 1
            except Exception as e:
                print(f"Solution processing failed for {task.get('sample_id', 'unknown')}_{task.get('type', 'unknown')} on {api_path}: {e}")
            finally:
                task_queue.task_done()
        print(f"Worker {worker_id} ({api_path}) processed {processed_count} solutions")

    # 创建工作线程，每个API分配 max_workers 个线程
    threads = []
    for api_path in api_paths:
        for i in range(max_workers):
            thread = threading.Thread(
                target=api_worker,
                args=(api_path, f"{api_path.split('/')[-2]}_{i}")
            )
            thread.start()
            threads.append(thread)
    
    # 创建进度条
    with tqdm(total=total_solutions, desc="Processing solutions") as pbar:
        while not task_queue.empty() or any(thread.is_alive() for thread in threads):
            time.sleep(0.5)  # 每0.5秒更新一次进度
            with solution_results_lock:
                pbar.n = completed_tasks  # 使用实时更新的计数器
                pbar.refresh()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print("All tasks completed.")
    # 最后再次根据id排序
    # results.sort(key=lambda x: x['id'])
    # 不再保存大文件，直接返回
    # return results

# %% main
if __name__ == "__main__":
    codecontests_data_path = "/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_gpt5_new_replace_add_feedback_gen_command_replace_new_new/"
    sandbox_num = 128
    sandbox_api_paths = []
    for i in range(sandbox_num):
        # sandbox_api_paths.append(f"http://10.244.128.90:{8080+i}/submit")  # 190 CPU
        # sandbox_api_paths.append(f"http://10.244.53.152:{8080+i}/submit")  # 190 CPU

        sandbox_api_paths.append(f"http://10.244.188.149:{8080+i}/submit")
        sandbox_api_paths.append(f"http://10.244.40.134:{8080+i}/submit")
        sandbox_api_paths.append(f"http://10.244.204.96:{8080+i}/submit")
        sandbox_api_paths.append(f"http://10.244.128.68:{8080+i}/submit")
        sandbox_api_paths.append(f"http://10.244.81.216:{8080+i}/submit")
        sandbox_api_paths.append(f"http://10.244.166.233:{8080+i}/submit")  # 128 CPU

        # sandbox_api_paths.append(f"http://10.244.40.153:{8080+i}/submit")

    random.shuffle(sandbox_api_paths)

    SESSION.mount('http://', requests.adapters.HTTPAdapter(pool_maxsize=len(sandbox_api_paths)))
    data_type = "test"
    dataset_type = f"code_contests_{data_type}"

    def get_corner_cases_with_max(sample):
        """
        获取样本的 corner cases
        """
        all_results = sample['result']
        solutions_num = float('inf')
        incorrect_solutions_num = float('inf')
        corner_cases = None
        for result in all_results:
            if len(result['result']['solution_result']) < solutions_num:
                solutions_num = len(result['result']['solution_result'])
                incorrect_solutions_num = len(result['result']['incorrect_solution_result'])
                corner_cases = result['corner_cases']
            elif len(result['result']['solution_result']) == solutions_num and len(result['result']['incorrect_solution_result']) < incorrect_solutions_num:
                incorrect_solutions_num = len(result['result']['incorrect_solution_result'])
                corner_cases = result['corner_cases']
        return corner_cases
    
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
                corner_cases = get_corner_cases_with_max(sample)
                # corner_cases = get_corner_cases_with_first(sample)
                if corner_cases:
                    sample['corner_cases'] = corner_cases
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
    # 随机采样 n 条数据
    # dataset = random.sample(dataset, n) if len(dataset) > n else dataset
    # 采样 start 到 end 条数据
    dataset = dataset[start:end]

    
    results_path = f"/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_gpt5_new_replace_add_feedback_gen_command_replace_new_new/"
    time_consume_path = f"/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_gpt5_new_replace_add_feedback_gen_command_replace_new_new.txt"
    # 获取初始时间
    start_time = time.time()
    codecontests_call(dataset, sandbox_api_paths, dataset_type, results_path, max_workers=1)
    # 获取结束时间
    end_time = time.time()
    # 计算总耗时
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    # 将总耗时写入结果文件
    with open(time_consume_path, "w") as f:
        f.write(f"Total processing time: {total_time:.2f} seconds; sample number: {len(dataset)}\n")


# 使用示例：
# 
# 1. 每完成一个任务就保存一次（默认行为，提供更好的故障恢复能力）：
# codecontests_call(dataset, api_paths, save_one_step=True)
#
# 2. 等到整个样本完成后再保存（减少 I/O 操作，提升性能）：
# codecontests_call(dataset, api_paths, save_one_step=False)
#
# save_one_step=True: 
#   - 优点：每个任务完成后立即保存，故障恢复能力强，进度实时可见
#   - 缺点：频繁的文件 I/O 操作，可能影响性能
#
# save_one_step=False:
#   - 优点：减少文件 I/O 操作，性能更好
#   - 缺点：如果程序中途崩溃，可能丢失部分进度
