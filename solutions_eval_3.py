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
from datasets import load_dataset, load_from_disk
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import time
import queue
import gc

completed_tasks = 0  # 全局计数器，用于跟踪已完成的任务数量
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
    data = data[kwargs['split']] if 'split' in kwargs and kwargs['split'] else data
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
            sample['test'] = sample['original_test']
        elif transform == "codecontents_plus":
            test_cases = []
            test_cases_transform_codecontests_plus(sample['test_cases'], test_cases)
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
    # res = requests.post(
    #     api_path,
    #     json=payload,
    #     # timeout=payload['config']['run_timeout'] + 10  # 设置超时时间为 30 秒
    # )

    # return res.json()
    with SESSION.post(
        api_path,
        json=payload,
        # timeout=payload['config']['run_timeout'] + 10  # 设置超时时间为 30 秒
    ) as res:
        return res.json()

# def solution_call(config, solutions, api_path, id, dataset_type, i, total, max_workers=8):
#     """
#     并发调用远程的 sandbox API 来验证生成的 corner case 是否正确
#     """
#     def process_single_solution(language_index, solution):
#         language = LANGUAGE[language_index]
#         if language == "UNKNOWN_LANGUAGE":
#             return None
#         if "PYTHON" in language:
#             language = "PYTHON"
#         config_copy = config.copy()
#         config_copy['language'] = language.lower()
#         completion = TEMPLATE.format(language=language.lower(), solution=solution)
#         resp = sandbox_call(config_copy, completion, api_path, id, dataset_type)
#         return {
#             'language': language,
#             'solution': solution,
#             'result': resp
#         }
    
#     results = []
#     solution_pairs = list(zip(solutions['language'], solutions['solution']))
    
#     # 使用线程池进行并发处理
#     with ThreadPoolExecutor(max_workers=min(max_workers, len(solution_pairs) if len(solution_pairs) > 0 else 1)) as executor:
#         # 提交所有任务
#         future_to_solution = {
#             executor.submit(process_single_solution, lang_idx, sol): (lang_idx, sol)
#             for lang_idx, sol in solution_pairs
#         }
        
#         # 收集结果
#         with tqdm(total=len(future_to_solution), desc=f"Processing solutions {i+1}/{total}", leave=False) as pbar:
#             for future in as_completed(future_to_solution):
#                 try:
#                     result = future.result()
#                     if result is not None:
#                         results.append(result)
#                 except Exception as e:
#                     lang_idx, sol = future_to_solution[future]
#                     print(f"Solution processing failed for language {LANGUAGE[lang_idx]}: {e}")
#                 finally:
#                     pbar.update(1)
    
#     return results


def codecontests_call(dataset, api_paths, dataset_type="code_contests_valid", results_path=None, max_workers=1, debug=False):
    """
    并发调用远程的 sandbox API 来验证生成的 corner case 是否正确。
    使用任务队列实现动态负载均衡的多 API 并行调用。
    现在队列的每个元素为每个 solution/incorrect_solution，实现 solution 级别的并行。
    
    Args:
        max_workers: solution 级别的并行度，总并行线程数
        debug: 是否启用调试模式，会输出任务分配信息
    """

    # results_path 现在是结果文件夹
    results_dir = results_path
    os.makedirs(results_dir, exist_ok=True)
    # 不需要全局的 results 列表，改为每个 sample 的结果单独存储
    # results = []
    # 已有的 sample id
    existing_ids = set()
    for fname in os.listdir(results_dir):
        if fname.endswith('.json'):
            existing_ids.add(os.path.splitext(fname)[0])

    results_lock = threading.Lock()  # 用于保护结果列表的线程锁
    solution_results = {}  # 用于收集每个sample的solution结果
    solution_results_lock = threading.Lock()  # 保护solution_results的线程锁
    
    # 筛选出需要处理的新样本
    samples_to_process = [
        sample for sample in dataset if sample['id'].replace('/', '_') not in existing_ids
    ]
    
    if not samples_to_process:
        print("All samples have already been processed. No new tasks to run.")
        return None

    # 创建任务队列 - 现在队列元素为每个solution/incorrect_solution
    task_queue = queue.Queue()
    total_solutions = 0
    sample_pool = {}
    for sample in samples_to_process:
        sample_id = sample['id']
        sample_pool[sample_id] = sample
        # 为每个sample初始化结果存储
        solution_results[sample_id] = {
            'sample': sample,
            'solution_result': [],
            'incorrect_solution_result': [],
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
                    # 'sample': sample
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
                    # 'sample': sample
                })
                total_solutions += 1
    
    print(f"Found {len(samples_to_process)} new samples to process out of {len(dataset)} total.")
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
                config = {
                    'language': language.lower(),
                    'locale': "en",
                    'compile_timeout': 20,
                    'run_timeout': 20,
                    'dataset_type': "CommonOJDataset"
                }
                key_list = ['id', 'content', 'test', 'labels', 'canonical_solution']
                provided_data = {key: sample[key] for key in key_list}
                config['provided_data'] = provided_data
                completion = TEMPLATE.format(language=language.lower(), solution=task['solution'])
                resp = sandbox_call(config, completion, api_path, task['sample_id'], dataset_type)
                solution_result = {
                    'language': task['language'],
                    'solution': task['solution'],
                    'result': resp
                }
                # 修改任务完成时的计数逻辑
                with solution_results_lock:
                    sample_id = task['sample_id']
                    if task['type'] == 'solution':
                        solution_results[sample_id]['solution_result'].append(solution_result)
                        solution_results[sample_id]['completed_solutions'] += 1
                    else:
                        solution_results[sample_id]['incorrect_solution_result'].append(solution_result)
                        solution_results[sample_id]['completed_incorrect_solutions'] += 1
                    sr = solution_results[sample_id]
                    if (sr['completed_solutions'] == sr['total_solutions'] and 
                        sr['completed_incorrect_solutions'] == sr['total_incorrect_solutions']):
                        result = {
                            'id': sample_id,
                            'solution_result': sr['solution_result'],
                            'incorrect_solution_result': sr['incorrect_solution_result'],
                            'api_used': api_path,
                            'worker_id': worker_id
                        }
                        with results_lock:
                            # results.append(result)
                            # 保存单个 sample 结果到 results_dir/{sample_id}.json，sample_id 中的 / 替换为 _
                            safe_sample_id = sample_id.replace('/', '_')
                            result_path = os.path.join(results_dir, f"{safe_sample_id}.json")
                            with open(result_path, "w") as f:
                                json.dump(result, f, indent=4)
                        # 删除对应的 solution_results[sample_id]，避免内存泄漏
                        del solution_results[sample_id]
                        gc.collect()
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
    # codecontests_data_path = "/aiarena/gpfs/code_contests"
    # codecontests_data_path = "/aiarena/gpfs/Code-Contests-Plus/test_plus_dataset"
    # codecontests_data_path = "/aiarena/gpfs/Code-Contests-Ours/test_all_solutions_repeat_2"
    # codecontests_data_path = "/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered"
    codecontests_data_path = "/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_o4mini_10maxsample_repeat_2/"
    # codecontests_data_path = "/aiarena/group/llmgroup/Code-Contests-Plus"
    # sandbox_api_paths = [
    #     # "http://10.244.166.211:8080/submit",
    #     # "http://10.244.166.219:8080/submit",
    #     "http://10.244.166.233:8080/submit",
    # ]
    sandbox_num = 128
    sandbox_api_paths = []
    for i in range(sandbox_num):
        # sandbox_api_paths.append(f"http://10.244.128.90:{8080+i}/submit")  # 190 CPU
        # sandbox_api_paths.append(f"http://10.244.53.152:{8080+i}/submit")  # 190 CPU

        sandbox_api_paths.append(f"http://10.244.188.142:{8080+i}/submit")  # 128 CPU
        sandbox_api_paths.append(f"http://10.244.179.53:{8080+i}/submit")
        sandbox_api_paths.append(f"http://10.244.128.66:{8080+i}/submit")
        sandbox_api_paths.append(f"http://10.244.204.124:{8080+i}/submit")

    SESSION.mount('http://', requests.adapters.HTTPAdapter(pool_maxsize=len(sandbox_api_paths)))
    data_type = "test"
    dataset_type = f"code_contests_{data_type}"
    
    # dataset = dataset_read(codecontests_data_path, "codecontents", split=None)
    file_paths = os.listdir(codecontests_data_path)
    dataset = []
    for file_path in file_paths:
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
    # dataset = dataset_read(codecontests_data_path, "codecontents_plus", split=data_type)
    # original_dataset = dataset_read("/aiarena/gpfs/code_contests", transform="codecontents", split=data_type)

    # # 先构建 id 到 original_sample 的映射
    # original_dict = {sample['id']: sample for sample in original_dataset}

    # for sample in dataset:
    #     sample['id'] = "/".join(["Codeforces"] + sample['id'].split("_"))
    # # 遍历 dataset，查找对应的 original_sample
    # new_dataset = []
    # for sample in dataset:
    #     sample_id = sample['id']
    #     original_sample = original_dict.get(sample_id)
    #     if original_sample is not None:
    #         # 找到了对应的 original_sample
    #         original_sample['test'] = sample['test']  # 保留 test cases
    #         new_dataset.append(original_sample)
    #     else:
    #         # 没有找到对应的 original_sample
    #         pass
    # dataset = new_dataset

    start = 0
    end = len(dataset)
    end = min(end, len(dataset))
    n = end - start
    # 随机采样 n 条数据
    # dataset = random.sample(dataset, n) if len(dataset) > n else dataset
    # 采样 start 到 end 条数据
    dataset = dataset[start:end]

    # results_path = f"/aiarena/group/llmgroup/caijf/ut_gen/code_contests_ours_all_solutions_repeat_2_{data_type}_results_{len(sandbox_api_paths)}sandbox_{len(sandbox_api_paths)}_1concur_1runconcur_{len(sandbox_api_paths)}_collocate_queue_dir_{len(dataset)}_{start}_{end}"
    # time_consume_path = f"/aiarena/group/llmgroup/caijf/ut_gen/code_contests_ours_all_solutions_repeat_2_{data_type}_results_{len(sandbox_api_paths)}sandbox_{len(sandbox_api_paths)}_1concur_1runconcur_{len(sandbox_api_paths)}_collocate_queue_dir_{len(dataset)}_{start}_{end}.txt"
    results_path = f"/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_10maxsample_repeat_2"
    time_consume_path = f"/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_10maxsample_repeat_2.txt"
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
