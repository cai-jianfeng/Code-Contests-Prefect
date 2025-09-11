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
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import time
import queue
TEMPLATE = """
Here is a {language} solution to the problem: 
```{language}
{solution}
```
"""
LANGUAGE = ["UNKNOWN_LANGUAGE", "PYTHON", "CPP", "PYTHON3", "JAVA"]

# %% functions
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


def sandbox_call(config, completion, api_path, id, dataset_type):
    """
    调用远程的 sandbox API 来验证生成的 corner case 是否正确。
    """
    payload = {
        'dataset': dataset_type,
        'id': '',
        'completion': completion,
        'config': config
    }
    res = requests.post(
        api_path,
        json=payload,
        # timeout=payload['config']['run_timeout'] + 10  # 设置超时时间为 30 秒
    )

    return res.json()

def solution_call(config, solutions, api_path, id, dataset_type, i, total, max_workers=8):
    """
    并发调用远程的 sandbox API 来验证生成的 corner case 是否正确
    """
    def process_single_solution(language_index, solution):
        language = LANGUAGE[language_index]
        if language == "UNKNOWN_LANGUAGE":
            return None
        if "PYTHON" in language:
            language = "PYTHON"
        config_copy = config.copy()
        config_copy['language'] = language.lower()
        completion = TEMPLATE.format(language=language.lower(), solution=solution)
        resp = sandbox_call(config_copy, completion, api_path, id, dataset_type)
        return {
            'language': language,
            'solution': solution,
            'result': resp
        }
    
    results = []
    solution_pairs = list(zip(solutions['language'], solutions['solution']))
    
    # 使用线程池进行并发处理
    with ThreadPoolExecutor(max_workers=min(max_workers, len(solution_pairs) if len(solution_pairs) > 0 else 1)) as executor:
        # 提交所有任务
        future_to_solution = {
            executor.submit(process_single_solution, lang_idx, sol): (lang_idx, sol)
            for lang_idx, sol in solution_pairs
        }
        
        # 收集结果
        with tqdm(total=len(future_to_solution), desc=f"Processing solutions {i+1}/{total}", leave=False) as pbar:
            for future in as_completed(future_to_solution):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    lang_idx, sol = future_to_solution[future]
                    print(f"Solution processing failed for language {LANGUAGE[lang_idx]}: {e}")
                finally:
                    pbar.update(1)
    
    return results


def codecontests_call(dataset, api_paths, dataset_type="code_contests_valid", results_path=None, max_workers=4, debug=False):
    """
    并发调用远程的 sandbox API 来验证生成的 corner case 是否正确。
    使用任务队列实现动态负载均衡的多 API 并行调用。
    
    Args:
        debug: 是否启用调试模式，会输出任务分配信息
    """
    results = []
    existing_ids = set()
    if results_path and os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
                existing_ids = {res['id'] for res in results}
                print(f"Loaded {len(results)} existing results from {results_path}")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Could not read or parse existing results file at {results_path}. Starting fresh.")
            results = []
            existing_ids = set()

    results_lock = threading.Lock()  # 用于保护结果列表的线程锁
    
    # 筛选出需要处理的新样本
    samples_to_process = [
        (i, sample) for i, sample in enumerate(dataset) if sample['id'] not in existing_ids
    ]
    
    if not samples_to_process:
        print("All samples have already been processed. No new tasks to run.")
        return results

    # 创建任务队列
    task_queue = queue.Queue()
    for task in samples_to_process:
        task_queue.put(task)
    
    print(f"Found {len(samples_to_process)} new samples to process out of {len(dataset)} total.")
    print(f"Using {len(api_paths)} API endpoints: {api_paths}")
    
    # 计算总的并行度：每个 API * max_workers
    total_max_workers = max_workers * len(api_paths)
    print(f"Total parallel workers: {total_max_workers} (max_workers={max_workers} * {len(api_paths)} APIs)")

    # 用于调试的任务分配跟踪
    if debug:
        task_assignment_lock = threading.Lock()
        assigned_tasks = set()

    def api_worker(api_path, worker_id):
        """每个API的工作函数，从任务队列中获取任务并处理"""
        processed_count = 0
        
        while True:
            try:
                # 从队列中获取任务，超时时间为1秒
                # queue.Queue.get() 是线程安全的，多个线程同时调用不会有问题
                i, sample = task_queue.get(timeout=1)
                
                # 可选：添加调试信息验证任务分配
                # print(f"Worker {worker_id} got task {sample['id']}")
                if debug:
                    with task_assignment_lock:
                        assigned_tasks.add(sample['id'])
                
            except queue.Empty:
                # 队列为空，退出工作线程
                break
                
            try:
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

                res = solution_call(config, solutions, api_path, id, dataset_type, i, len(dataset))
                incorrect_res = solution_call(config, incorrect_solutions, api_path, id, dataset_type, i, len(dataset))
                
                result = {
                    'id': id,
                    'solution_result': res,
                    'incorrect_solution_result': incorrect_res,
                    'api_used': api_path,  # 记录使用的 API
                    'worker_id': worker_id  # 记录处理的工作线程ID
                }
                
                # 线程安全地添加结果
                with results_lock:
                    results.append(result)
                    processed_count += 1
                    # 实时保存结果
                    if results_path:
                        # 对结果进行排序，以保持一致性
                        results.sort(key=lambda x: x['id'])
                        with open(results_path, "w") as f:
                            json.dump(results, f, indent=4)
                
            except Exception as e:
                print(f"Sample processing failed for {sample.get('id', f'index {i}')} on {api_path}: {e}")
            finally:
                # 标记任务完成
                task_queue.task_done()
        
        print(f"Worker {worker_id} ({api_path}) processed {processed_count} samples")

    # 创建工作线程
    threads = []
    worker_id = 0
    for api_path in api_paths:
        for i in range(max_workers):
            worker_id += 1
            thread = threading.Thread(
                target=api_worker, 
                args=(api_path, f"{api_path.split('/')[-2]}_{i}")
            )
            thread.start()
            threads.append(thread)
    
    # 创建进度条
    total_tasks = len(samples_to_process)
    with tqdm(total=total_tasks, desc="Processing samples") as pbar:
        completed_before = 0
        while not task_queue.empty() or any(thread.is_alive() for thread in threads):
            time.sleep(0.5)  # 每0.5秒更新一次进度
            with results_lock:
                completed_now = len(results) - len([r for r in results if r['id'] in existing_ids])
                if completed_now > completed_before:
                    pbar.update(completed_now - completed_before)
                    completed_before = completed_now
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 最后再次根据id排序
    results.sort(key=lambda x: x['id'])

    # 保存结果
    if results_path:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_path}")
    return results


if __name__ == "__main__":
    codecontests_data_path = "/aiarena/gpfs/code_contests"
    # sandbox_api_paths = [
    #     "http://10.244.166.228:8080/submit",
    #     "http://10.244.81.250:8080/submit",
    #     "http://10.244.204.90:8080/submit",
    #     "http://10.244.204.70:8080/submit",
    #     "http://10.244.179.35:8080/submit",
    #     "http://10.244.179.31:8080/submit",
    #     "http://10.244.166.199:8080/submit",
    #     "http://10.244.81.204:8080/submit"  # 新增的 API 路径
    # ]
    sandbox_api_paths = []
    for i in range(4):
        sandbox_api_paths.append(f"http://10.244.128.96:{8080+i}/submit")
    dataset_type = "code_contests_train"
    results_path = "/aiarena/gpfs/code_contests_train_results_32_32concur_4*4*8_collocat.json"
    time_consume_path = "/aiarena/gpfs/code_contests_train_results_32_32concur_4*4*8_collocat.txt"
    dataset = dataset_read(codecontests_data_path, "train")
    # 随机采样 n 条数据
    n = 256
    dataset = random.sample(dataset, n) if len(dataset) > n else dataset
    # 获取初始时间
    start_time = time.time()
    results = codecontests_call(dataset, sandbox_api_paths, dataset_type, results_path, max_workers=4)
    # 获取结束时间
    end_time = time.time()
    # 计算总耗时
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    # 将总耗时写入结果文件
    with open(time_consume_path, "w") as f:
        f.write(f"Total processing time: {total_time:.2f} seconds; sample number: {n}\n")