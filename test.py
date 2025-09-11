# %% setup
from datasets import load_from_disk
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import time
import gc

dataset = load_from_disk("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered")

# dataset_plus = load_from_disk("/aiarena/gpfs/Code-Contests-Plus/test_plus_dataset")
# %%
'''
features: ['checker', 'generator_cmd', 'correct_submissions', 'true_negative_rate', 'test_cases', 'generator', 'true_positive_rate', 'memory_limit', 'incorrect_submissions', 'title', 'validator', 'cf_points', 'untranslated_description', 'is_description_translated', 'name', 'output_file', 'generated_tests', 'cf_index', 'difficulty', 'private_tests', 'input_file', 'cf_contest_id', 'cf_rating', 'cf_tags', 'public_tests', 'labels', 'plus_canonical_solution', 'original_canonical_solution', 'content', 'description', 'time_limit', 'original_test', 'incorrect_solutions', 'id', 'solutions', 'source', 'canonical_solution'],

test cases 是 CodeContests-Plus 的 test cases
original_test 是 CodeContests 的 test cases
'''
ERROR_SOLUTIONS_NUM = 0
sandbox_num = 128
sandbox_api_paths = []
for i in range(sandbox_num):

    sandbox_api_paths.append(f"http://10.244.188.142:{8080+i}/submit")  # 128 CPU
    sandbox_api_paths.append(f"http://10.244.179.53:{8080+i}/submit")
    sandbox_api_paths.append(f"http://10.244.128.66:{8080+i}/submit")
    sandbox_api_paths.append(f"http://10.244.204.124:{8080+i}/submit")

TEMPLATE = """
Here is a {language} solution to the problem: 
```{language}
{solution}
```
"""
LANGUAGE = ["UNKNOWN_LANGUAGE", "PYTHON", "CPP", "PYTHON3", "JAVA"]

SESSION = requests.Session()
# 配置 SESSION 的连接池
SESSION.mount('http://', requests.adapters.HTTPAdapter(pool_maxsize=len(sandbox_api_paths)))

completed_tasks = 0  # 全局计数器，用于跟踪已完成的任务数量

def sandbox_call(api_path, payload):
    """
    调用远程的 sandbox API 来验证生成的 corner case 是否正确。
    """
    with SESSION.post(
        api_path,
        json=payload,
    ) as res:
        return res.json()

def solution_eval_parallel(dataset, api_paths, max_workers=1):
    """
    并发调用远程的 sandbox API 来评估所有样本的 solutions。
    使用任务队列实现动态负载均衡的多 API 并行调用。
    
    Args:
        dataset: 数据集列表
        api_paths: API 端点列表
        max_workers: 每个 API 的并行度
    """
    global completed_tasks
    completed_tasks = 0
    
    # 结果存储
    results_lock = threading.Lock()
    sample_results = {}  # 用于收集每个sample的结果
    sample_results_lock = threading.Lock()
    
    # 创建任务队列 - 现在队列元素为每个solution/incorrect_solution
    task_queue = queue.Queue()
    total_solutions = 0
    
    # 为每个sample初始化结果存储并添加任务到队列
    for sample_idx, sample in enumerate(dataset):
        sample_id = sample['id']
        public_tests = sample['public_tests']
        stdin = public_tests['input'][0]
        
        # 初始化结果存储
        sample_results[sample_id] = {
            'sample_idx': sample_idx,
            'sample': sample,
            'stdin': stdin,
            'solution_results': [],
            'incorrect_solution_results': [],
            'completed_solutions': 0,
            'completed_incorrect_solutions': 0,
            'total_solutions': len([lang for lang in sample['solutions']['language'] if LANGUAGE[lang] != "UNKNOWN_LANGUAGE"]),
            'total_incorrect_solutions': len([lang for lang in sample['incorrect_solutions']['language'] if LANGUAGE[lang] != "UNKNOWN_LANGUAGE"])
        }
        
        # 添加正确的solutions到队列
        for lang_idx, solution in zip(sample['solutions']['language'], sample['solutions']['solution']):
            language = LANGUAGE[lang_idx]
            if language != "UNKNOWN_LANGUAGE":
                task_queue.put({
                    'type': 'solution',
                    'sample_id': sample_id,
                    'language_index': lang_idx,
                    'language': language,
                    'solution': solution,
                    'stdin': stdin
                })
                total_solutions += 1
        
        # 添加错误的solutions到队列  
        for lang_idx, solution in zip(sample['incorrect_solutions']['language'], sample['incorrect_solutions']['solution']):
            language = LANGUAGE[lang_idx]
            if language != "UNKNOWN_LANGUAGE":
                task_queue.put({
                    'type': 'incorrect_solution',
                    'sample_id': sample_id,
                    'language_index': lang_idx,
                    'language': language,
                    'solution': solution,
                    'stdin': stdin
                })
                total_solutions += 1
    
    print(f"Total solutions/incorrect_solutions to process: {total_solutions}")
    print(f"Using {len(api_paths)} API endpoints")
    print(f"Total parallel workers: {len(api_paths) * max_workers}")

    def api_worker(api_path, worker_id):
        """每个API的工作函数，从任务队列中获取任务并处理"""
        processed_count = 0
        global completed_tasks
        
        while True:
            try:
                task = task_queue.get(timeout=1)
            except queue.Empty:
                break
                
            try:
                # 调用 sandbox API
                payload = {
                    "code": task['solution'],
                    "stdin": task['stdin'],
                    "language": task['language'],
                }
                response = sandbox_call(api_path, payload)
                
                # 处理结果
                solution_result = {
                    'language_index': task['language_index'],
                    'language': task['language'],
                    'solution': task['solution'],
                    'response': response,
                    'valid': response.get("status") != "Failed"
                }
                
                if not solution_result['valid']:
                    global ERROR_SOLUTIONS_NUM
                    ERROR_SOLUTIONS_NUM += 1
                    print(f"Error solution for {task['sample_id']}_{task['type']} on {api_path}: {solution_result}")
                # 更新结果
                with sample_results_lock:
                    sample_id = task['sample_id']
                    if task['type'] == 'solution':
                        sample_results[sample_id]['solution_results'].append(solution_result)
                        sample_results[sample_id]['completed_solutions'] += 1
                    else:
                        sample_results[sample_id]['incorrect_solution_results'].append(solution_result)
                        sample_results[sample_id]['completed_incorrect_solutions'] += 1
                        
                completed_tasks += 1
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
            pbar.n = completed_tasks
            pbar.refresh()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print("All tasks completed.")
    
    # 处理结果并更新原始数据集
    for sample_id, result_data in sample_results.items():
        sample_idx = result_data['sample_idx']
        sample = dataset[sample_idx]
        
        # 过滤有效的solutions
        filter_solutions = {"language": [], "solution": []}
        for sol_result in result_data['solution_results']:
            if sol_result['valid']:
                filter_solutions["language"].append(sol_result['language'])
                filter_solutions["solution"].append(sol_result['solution'])
        
        # 过滤有效的incorrect_solutions
        filter_incorrect_solutions = {"language": [], "solution": []}
        for sol_result in result_data['incorrect_solution_results']:
            if sol_result['valid']:
                filter_incorrect_solutions["language"].append(sol_result['language'])
                filter_incorrect_solutions["solution"].append(sol_result['solution'])
        
        # 更新样本
        sample['solutions'] = filter_solutions
        sample['incorrect_solutions'] = filter_incorrect_solutions
        
        print(f"Processed sample {sample['id']} with {len(filter_solutions['solution'])} valid solutions and {len(filter_incorrect_solutions['solution'])} valid incorrect solutions.")


# 原始单线程版本（保留作为备份）
def solution_eval(solutions, stdin, api_path):
    """
    调用 sandbox API 来评估 solutions。
    """
    filter_solutions = {"language": [], "solution": []}
    for language_idx, solution in zip(solutions['language'], solutions['solution']):
        language = LANGUAGE[language_idx]
        if language == "UNKNOWN_LANGUAGE":
            print(f"Skipping solution with UNKNOWN_LANGUAGE: {solution}")
            continue
        payload = {
            "code": solution,
            "stdin": stdin,
            "language": language,
        }
        response = sandbox_call(api_path, payload)
        if response.get("status") == "Failed":
            print(f"Solution failed for solution {solution}: {response}")
            continue
        filter_solutions["language"].append(language)
        filter_solutions["solution"].append(solution)
    return filter_solutions

# 使用多进程版本处理数据集
start_time = time.time()
solution_eval_parallel(dataset, sandbox_api_paths, max_workers=1)
end_time = time.time()

print(f"Total processing time: {end_time - start_time:.2f} seconds")

# 原始单线程版本（注释掉）
# for sample in dataset:
#     public_tests = sample['public_tests']
#     stdin = public_tests[0]['input']
#     solutions = sample['solutions']
#     filter_solutions = solution_eval(solutions, stdin, sandbox_api_paths[0])
#     incorrect_solutions = sample['incorrect_solutions']
#     filter_incorrect_solutions = solution_eval(incorrect_solutions, stdin, sandbox_api_paths[0])

#     sample['solutions'] = filter_solutions
#     sample['incorrect_solutions'] = filter_incorrect_solutions

#     print(f"Processed sample {sample['id']} with {len(filter_solutions['solution'])} valid solutions and {len(filter_incorrect_solutions['solution'])} valid incorrect solutions.")

# Save the processed dataset
# dataset.save_to_disk("/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered_solutions")
print("Dataset processed and saved successfully.")
print(f"Total error solutions: {ERROR_SOLUTIONS_NUM}")

# %%

from datasets import Dataset, DatasetDict, load_from_disk, load_dataset

dataset = load_dataset("/aiarena/group/llmgroup/Code-Contests-Plus", "3x")

dataset = dataset['train']

