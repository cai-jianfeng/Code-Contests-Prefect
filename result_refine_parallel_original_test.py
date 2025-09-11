"""
该模块的主要功能是使用每个 sample 自带的 checker 对结果进行进一步验证 (多进程版本)
1. 给定结果文件夹路径，加载所有 JSON 文件
2. 对每个文件，提取正确和错误的解决方案结果
3. 对于正确的解决方案，如果其内的 result 中 accepted 为 False，则使用 checker 进行进一步验证：
    4. 对于其内的 tests，遍历每个 test case
        5. 如果其的 status 为 "Success"，则提取它的 stdout 和 test case 的 output，并使用 checker 进行验证
            6. 如果验证通过，则将该 test case 的结果 (passed) 标记为 true
            7. 如果验证失败，则直接跳过剩余的 test cases
        8. 如果其的 status 为 "Failed"，则直接跳过该解决方案

多进程改进：
- 支持多个 sandbox API 端点并行处理
- 使用线程池和任务队列实现负载均衡
- 支持test case级别的并行处理
"""

# %% setup
import os
import json
import time
import queue
import threading
from tqdm import tqdm
from solutions_eval import dataset_read
import requests
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any

# 全局计数器
completed_tasks = 0
results_lock = threading.Lock()

def load_results(result_folder):
    """加载结果文件夹中的所有JSON文件"""
    result_files = os.listdir(result_folder)
    results = {}

    for result_file in tqdm(result_files, desc="Loading result files"):
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
            if result_file.endswith(".json"):
                print(f"Skipping checker file: {result_file}")

    print(f"Loaded {len(results)} results from {result_folder}")
    return results


class SandboxSession:
    """Sandbox API 会话管理器"""
    
    def __init__(self, api_paths: List[str]):
        self.session = requests.Session()
        # 设置连接池大小
        self.session.mount('http://', requests.adapters.HTTPAdapter(
            pool_maxsize=len(api_paths) * 4
        ))
    
    def is_time_limit_exceeded(self, response):
        """
        检查响应是否包含 TimeLimitExceeded 状态
        """
        tests = response.get("tests", [])
        for test in tests:
            if test.get('exec_info', {}).get('status') == "Failed":
                compile_result = test.get('exec_info', {}).get('compile_result')
                run_result = test.get('exec_info', {}).get('run_result')
                if (isinstance(compile_result, dict) and compile_result.get('status') == "TimeLimitExceeded") or \
                (isinstance(run_result, dict) and run_result.get('status') == "TimeLimitExceeded"):
                    return True
        return False


def has_time_limit_exceeded_in_checker_results(result_file_path):
    """
    检查结果文件中是否包含 checker_info 中的 TimeLimitExceeded 状态
    返回需要重新测试的 solution 和 incorrect_solution 的索引列表和具体的 test case 索引
    """
    try:
        with open(result_file_path, 'r') as f:
            result_data = json.load(f)
        
        retry_solutions = []  # [(solution_idx, [test_indices])]
        retry_incorrect_solutions = []  # [(solution_idx, [test_indices])]
        
        # 检查 solution_result
        for sol_idx, solution_result in enumerate(result_data.get('solution_result', [])):
            tests = solution_result.get('result', {}).get('tests', [])
            retry_test_indices = []
            
            for test_idx, test in enumerate(tests):
                checker_info = test.get('checker_info', {})
                if isinstance(checker_info, dict):
                    # 检查 checker_info 中是否有 TimeLimitExceeded
                    checker_tests = checker_info.get("tests", [])
                    for checker_test in checker_tests:
                        if checker_test.get('exec_info', {}).get('status') == "Failed":
                            compile_result = checker_test.get('exec_info', {}).get('compile_result')
                            run_result = checker_test.get('exec_info', {}).get('run_result')
                            if (isinstance(compile_result, dict) and compile_result.get('status') == "TimeLimitExceeded") or \
                               (isinstance(run_result, dict) and run_result.get('status') == "TimeLimitExceeded"):
                                retry_test_indices.append(test_idx)
                                break
            
            if retry_test_indices:
                retry_solutions.append((sol_idx, retry_test_indices))
        
        # 检查 incorrect_solution_result  
        for sol_idx, incorrect_result in enumerate(result_data.get('incorrect_solution_result', [])):
            tests = incorrect_result.get('result', {}).get('tests', [])
            retry_test_indices = []
            
            for test_idx, test in enumerate(tests):
                checker_info = test.get('checker_info', {})
                if isinstance(checker_info, dict):
                    # 检查 checker_info 中是否有 TimeLimitExceeded
                    checker_tests = checker_info.get("tests", [])
                    for checker_test in checker_tests:
                        if checker_test.get('exec_info', {}).get('status') == "Failed":
                            compile_result = checker_test.get('exec_info', {}).get('compile_result')
                            run_result = checker_test.get('exec_info', {}).get('run_result')
                            if (isinstance(compile_result, dict) and compile_result.get('status') == "TimeLimitExceeded") or \
                               (isinstance(run_result, dict) and run_result.get('status') == "TimeLimitExceeded"):
                                retry_test_indices.append(test_idx)
                                break
            
            if retry_test_indices:
                retry_incorrect_solutions.append((sol_idx, retry_test_indices))
        
        return retry_solutions, retry_incorrect_solutions
        
    except Exception as e:
        print(f"Error reading result file {result_file_path}: {e}")
        return [], []

def create_checker_payload(checker_code: str, stdin: str, stdout: str, expected_output: str) -> Optional[Dict]:
    """创建checker验证的payload"""
    try:
        # 直接将字符串编码为 base64
        stdin_b64 = base64.b64encode(stdin.encode('utf-8')).decode('utf-8')
        stdout_b64 = base64.b64encode(stdout.encode('utf-8')).decode('utf-8')
        expected_output_b64 = base64.b64encode(expected_output.encode('utf-8')).decode('utf-8')
        
        # 读取 testlib.h 文件
        testlib_path = "testlib.h"
        if not os.path.exists(testlib_path):
            print(f"Warning: testlib.h not found at {testlib_path}")
            return None
            
        with open(testlib_path, 'rb') as file:
            testlib_data_b64 = base64.b64encode(file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding base64: {e}")
        return None

    files = {
        "input.txt": stdin_b64,
        "output.txt": stdout_b64,
        "answer.txt": expected_output_b64,
        "testlib.h": testlib_data_b64
    }

    payload = {
        "code": checker_code,
        "language": "cpp",
        "extra_args": "input.txt output.txt answer.txt",
        "files": files,
    }
    
    return payload


def process_single_test_case(task: Dict, sandbox_session: SandboxSession) -> Dict:
    """处理单个测试用例的checker验证"""
    api_path = task['api_path']
    test_info = task['test_info']
    checker_code = task['checker_code']
    result_id = task['result_id']
    solution_idx = task['solution_idx']
    test_idx = task['test_idx']
    solution_type = task['solution_type']
    
    try:
        test = test_info['test']
        
        # 检查是否需要处理
        if test['passed'] or test['exec_info'].get('status') != "Success":
            return {
                'result_id': result_id,
                'solution_idx': solution_idx,
                'solution_type': solution_type,
                'test_idx': test_idx,
                'status': 'skipped',
                'reason': 'already_passed_or_failed'
            }
        
        # 提取数据
        stdout = test['exec_info']['run_result'].get('stdout', '')
        expected_output = test['test_info']['output'].get('stdout', '')
        stdin = test['test_info']['input'].get('stdin', '')
        
        # 创建 payload
        payload = create_checker_payload(checker_code, stdin, stdout, expected_output)
        if payload is None:
            return {
                'result_id': result_id,
                'solution_idx': solution_idx,
                'solution_type': solution_type,
                'test_idx': test_idx,
                'status': 'error',
                'reason': 'payload_creation_failed'
            }
        
        # 调用 API
        response = sandbox_session.call_api(api_path, payload)
        
        return {
            'result_id': result_id,
            'solution_idx': solution_idx,
            'solution_type': solution_type,
            'test_idx': test_idx,
            'status': 'success',
            'checker_response': response
        }
        
    except Exception as e:
        print(f"Error processing test case {test_idx} for solution {solution_idx} in result {result_id}: {e}")
        return {
            'result_id': result_id,
            'solution_idx': solution_idx,
            'test_idx': test_idx,
            'status': 'error',
            'reason': str(e)
        }


def parallel_checker_validation(results_data: Dict, dataset_data: Dict, retry_info: Dict, api_paths: List[str], 
                               max_workers: int = 4, debug: bool = False) -> Dict:
    """并行执行checker验证，支持部分重新测试"""
    
    # 创建任务队列
    task_queue = queue.Queue()
    total_tasks = 0
    
    # 为每个需要验证的test case创建任务
    for result_id, result_data in results_data.items():
        if result_id not in dataset_data:
            print(f"Warning: Result ID {result_id} not found in dataset, skipping.")
            continue
            
        checker_code = dataset_data[result_id].get("checker", None)
        if not checker_code:
            print(f"No checker code for result {result_id}, skipping.")
            continue
        
        # 获取重试信息
        retry_data = retry_info.get(result_id, {})
        retry_solutions = retry_data.get('retry_solutions', [])
        retry_incorrect_solutions = retry_data.get('retry_incorrect_solutions', [])
        is_existing_result = retry_data.get('existing_file') is not None
        
        # 创建重试映射，方便查找
        retry_solution_map = {}
        for sol_idx, test_indices in retry_solutions:
            retry_solution_map[sol_idx] = set(test_indices)
        
        retry_incorrect_solution_map = {}
        for sol_idx, test_indices in retry_incorrect_solutions:
            retry_incorrect_solution_map[sol_idx] = set(test_indices)
        
        # 处理正确解决方案
        solution_results = result_data.get("solution_result", [])
        for sidx, solution in enumerate(solution_results):
            single_result = solution['result']
            if not single_result.get("accepted", True):
                tests = single_result.get("tests", [])
                for idx, test in enumerate(tests):
                    # 检查是否需要处理这个test case
                    should_process = False
                    
                    if not is_existing_result:
                        # 新结果，按原逻辑处理
                        if not test['passed'] and test['exec_info'].get('status') == "Success":
                            should_process = True
                    else:
                        # 已有结果，只处理需要重新测试的
                        if sidx in retry_solution_map and idx in retry_solution_map[sidx]:
                            should_process = True
                    
                    if should_process:
                        task_queue.put({
                            'api_path': None,  # 将在worker中分配
                            'test_info': {'test': test},
                            'checker_code': checker_code,
                            'result_id': result_id,
                            'solution_idx': sidx,
                            'test_idx': idx,
                            'solution_type': 'correct',
                            'is_retry': is_existing_result and sidx in retry_solution_map and idx in retry_solution_map[sidx]
                        })
                        total_tasks += 1
        
        # 处理错误解决方案
        incorrect_solution_results = result_data.get("incorrect_solution_result", [])
        for sidx, solution in enumerate(incorrect_solution_results):
            single_result = solution['result']
            if not single_result.get("accepted", True):
                tests = single_result.get("tests", [])
                for idx, test in enumerate(tests):
                    # 检查是否需要处理这个test case
                    should_process = False
                    
                    if not is_existing_result:
                        # 新结果，按原逻辑处理
                        if not test['passed'] and test['exec_info'].get('status') == "Success":
                            should_process = True
                    else:
                        # 已有结果，只处理需要重新测试的
                        if sidx in retry_incorrect_solution_map and idx in retry_incorrect_solution_map[sidx]:
                            should_process = True
                    
                    if should_process:
                        task_queue.put({
                            'api_path': None,  # 将在worker中分配
                            'test_info': {'test': test},
                            'checker_code': checker_code,
                            'result_id': result_id,
                            'solution_idx': sidx,
                            'test_idx': idx,
                            'solution_type': 'incorrect',
                            'is_retry': is_existing_result and sidx in retry_incorrect_solution_map and idx in retry_incorrect_solution_map[sidx]
                        })
                        total_tasks += 1
    
    if total_tasks == 0:
        print("No test cases need checker validation.")
        return {}
    
    print(f"Total test cases to validate: {total_tasks}")
    print(f"Using {len(api_paths)} API endpoints with {max_workers} workers per endpoint")
    
    # 结果存储
    processing_results = {}
    processing_results_lock = threading.Lock()
    
    # 全局计数器
    global completed_tasks
    completed_tasks = 0
    
    def api_worker(api_path: str, worker_id: str):
        """API worker函数"""
        sandbox_session = SandboxSession([api_path])
        processed_count = 0
        global completed_tasks
        
        while True:
            try:
                task = task_queue.get(timeout=1)
                task['api_path'] = api_path
                
                if debug:
                    print(f"Worker {worker_id} processing task for {task['result_id']}")
                
            except queue.Empty:
                break
            
            try:
                result = process_single_test_case(task, sandbox_session)
                # 传递 is_retry 标志
                result['is_retry'] = task.get('is_retry', False)
                
                with processing_results_lock:
                    if result['result_id'] not in processing_results:
                        processing_results[result['result_id']] = []
                    processing_results[result['result_id']].append(result)
                    completed_tasks += 1
                
                processed_count += 1
                
            except Exception as e:
                print(f"Worker {worker_id} error processing task: {e}")
            finally:
                task_queue.task_done()
        
        print(f"Worker {worker_id} ({api_path}) processed {processed_count} test cases")
    
    # 创建工作线程
    threads = []
    for api_path in api_paths:
        for i in range(max_workers):
            thread = threading.Thread(
                target=api_worker,
                args=(api_path, f"{api_path.split('/')[-1]}_{i}")
            )
            thread.start()
            threads.append(thread)
    
    # 创建进度条
    with tqdm(total=total_tasks, desc="Processing test cases") as pbar:
        while not task_queue.empty() or any(thread.is_alive() for thread in threads):
            time.sleep(0.5)
            with processing_results_lock:
                current_completed = completed_tasks
            pbar.n = current_completed
            pbar.refresh()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print("All checker validation tasks completed.")
    return processing_results


def apply_checker_results(results_data: Dict, processing_results: Dict) -> None:
    """将checker验证结果应用到原始数据中，支持重新测试的结果替换"""
    for result_id, checker_results in processing_results.items():
        if result_id not in results_data:
            continue
            
        result_data = results_data[result_id]
        
        # 按solution type分组
        for checker_result in checker_results:
            solution_type = checker_result['solution_type']
            solution_idx = checker_result['solution_idx']
            test_idx = checker_result['test_idx']
            is_retry = checker_result.get('is_retry', False)
            
            if solution_type == 'correct':
                solutions = result_data.get("solution_result", [])
            else:
                solutions = result_data.get("incorrect_solution_result", [])
            
            if solution_idx < len(solutions):
                tests = solutions[solution_idx]['result'].get("tests", [])
                if test_idx < len(tests):
                    test = tests[test_idx]
                    
                    if checker_result['status'] == 'success':
                        if is_retry:
                            # 重新测试，替换现有的 checker_info
                            test['checker_info'] = checker_result['checker_response']
                            print(f"Updated checker_info for {result_id} solution {solution_idx} test {test_idx}")
                        else:
                            # 新测试，直接设置
                            test['checker_info'] = checker_result['checker_response']
                    elif checker_result['status'] == 'error':
                        if is_retry:
                            # 重新测试，替换现有的错误信息
                            test['checker_error'] = {
                                'reason': checker_result['reason'],
                                'status': 'failed'
                            }
                            # 移除可能存在的旧 checker_info
                            test.pop('checker_info', None)
                            print(f"Updated checker_error for {result_id} solution {solution_idx} test {test_idx}")
                        else:
                            # 新测试，直接设置错误信息
                            test['checker_error'] = {
                                'reason': checker_result['reason'],
                                'status': 'failed'
                            }


def save_results(results_data: Dict, result_folder: str) -> None:
    """保存处理后的结果"""
    for result_id, result_data in results_data.items():
        output_file = os.path.join(result_folder, f"{result_id.replace('/', '_')}_checker.json")
        
        try:
            with open(output_file, 'w') as f:
                json.dump(result_data, f, indent=4)
            print(f"Updated results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results for {result_id}: {e}")


def filter_existing_results(results_data: Dict, dataset_data: Dict, result_folder: str) -> Tuple[Dict, Dict, Dict]:
    """过滤已经处理过的结果，并检查是否需要重新测试有 TimeLimitExceeded 的情况"""
    filtered_results = {}
    filtered_dataset = {}
    retry_info = {}  # 存储需要重新测试的信息
    
    for result_id in results_data.keys():
        output_file = os.path.join(result_folder, f"{result_id.replace('/', '_')}_checker.json")
        
        if result_id not in dataset_data:
            print(f"Result ID {result_id} not found in dataset, skipping.")
            continue
            
        checker_code = dataset_data[result_id].get("checker", None)
        if not checker_code:
            print(f"No checker code for result {result_id}, skipping.")
            continue
        
        if os.path.exists(output_file):
            # 检查是否有需要重新测试的 TimeLimitExceeded 情况
            retry_solutions, retry_incorrect_solutions = has_time_limit_exceeded_in_checker_results(output_file)
            
            if retry_solutions or retry_incorrect_solutions:
                # 需要重新测试，加载现有结果
                try:
                    with open(output_file, 'r') as f:
                        existing_result_data = json.load(f)
                    filtered_results[result_id] = existing_result_data
                    filtered_dataset[result_id] = dataset_data[result_id]
                    retry_info[result_id] = {
                        'retry_solutions': retry_solutions,
                        'retry_incorrect_solutions': retry_incorrect_solutions,
                        'existing_file': output_file
                    }
                    print(f"Found TimeLimitExceeded in {output_file}, will retry affected test cases.")
                except Exception as e:
                    print(f"Error loading existing result file {output_file}: {e}")
                    continue
            else:
                print(f"Output file {output_file} already exists and no TimeLimitExceeded found, skipping.")
                continue
        else:
            # 新结果，需要完全处理
            filtered_results[result_id] = results_data[result_id]
            filtered_dataset[result_id] = dataset_data[result_id]
            retry_info[result_id] = {
                'retry_solutions': [],
                'retry_incorrect_solutions': [],
                'existing_file': None
            }
    
    return filtered_results, filtered_dataset, retry_info


def main():
    """主函数"""
    # 配置参数
    result_folder = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_only_public_private/"
    data_folder = "/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered/"
    
    # 配置sandbox API端点
    sandbox_num = 128  # 可以根据需要调整
    sandbox_api_paths = []
    base_urls = [
        "http://10.244.40.153",
    ]
    
    for base_url in base_urls:
        for i in range(sandbox_num):
            sandbox_api_paths.append(f"{base_url}:{8080+i}/run_code")
    
    max_workers_per_api = 1  # 每个API端点的并发数
    debug = False
    
    print("=== Parallel Checker Validation Configuration ===")
    print(f"Result folder: {result_folder}")
    print(f"Data folder: {data_folder}")
    print(f"Number of API endpoints: {len(sandbox_api_paths)}")
    print(f"Workers per API: {max_workers_per_api}")
    print(f"Total parallel workers: {len(sandbox_api_paths) * max_workers_per_api}")
    print(f"Debug mode: {debug}")
    print("=" * 50)
    
    # 加载数据
    print("Loading results and dataset...")
    results = load_results(result_folder)
    # dataset = load_results(data_folder)
    dataset = dataset_read(data_folder, "codecontents", split=None)
    construct_dataset = {item['id']: item for item in dataset}
    dataset = construct_dataset
    if not results:
        print("No results found. Exiting.")
        return
    
    if not dataset:
        print("No dataset found. Exiting.")
        return
    
    # 过滤已处理的结果
    print("Filtering already processed results and checking for TimeLimitExceeded...")
    filtered_results, filtered_dataset, retry_info = filter_existing_results(results, dataset, result_folder)
    
    if not filtered_results:
        print("All samples have been processed and no TimeLimitExceeded cases need retry. No new tasks to run.")
        return
    
    # 统计重新测试的信息
    new_samples = 0
    retry_samples = 0
    total_retry_test_cases = 0
    
    for result_id, info in retry_info.items():
        if info['existing_file'] is None:
            new_samples += 1
        else:
            retry_samples += 1
            # 计算需要重试的test case数量
            for _, test_indices in info['retry_solutions']:
                total_retry_test_cases += len(test_indices)
            for _, test_indices in info['retry_incorrect_solutions']:
                total_retry_test_cases += len(test_indices)
    
    print(f"Found {len(filtered_results)} samples to process out of {len(results)} total.")
    print(f"New samples: {new_samples}")
    print(f"Samples with TimeLimitExceeded to retry: {retry_samples}")
    if retry_samples > 0:
        print(f"  - Total test cases to retry: {total_retry_test_cases}")
    
    # 开始处理
    start_time = time.time()
    
    print("Starting parallel checker validation...")
    processing_results = parallel_checker_validation(
        filtered_results, 
        filtered_dataset, 
        retry_info,
        sandbox_api_paths, 
        max_workers_per_api, 
        debug
    )
    
    # 应用结果
    print("Applying checker results...")
    apply_checker_results(filtered_results, processing_results)
    
    # 保存结果
    print("Saving results...")
    save_results(filtered_results, result_folder)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 输出统计信息
    print("=== Processing Complete ===")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Samples processed: {len(filtered_results)}")
    if len(filtered_results) > 0:
        print(f"Average time per sample: {total_time/len(filtered_results):.2f} seconds")
    
    # 保存时间统计
    time_file = os.path.join(result_folder, "checker_processing_stats.txt")
    with open(time_file, "w") as f:
        f.write(f"Total processing time: {total_time:.2f} seconds\n")
        f.write(f"Samples processed: {len(filtered_results)}\n")
        f.write(f"API endpoints used: {len(sandbox_api_paths)}\n")
        f.write(f"Workers per API: {max_workers_per_api}\n")
        f.write(f"Total parallel workers: {len(sandbox_api_paths) * max_workers_per_api}\n")
        if len(filtered_results) > 0:
            f.write(f"Average time per sample: {total_time/len(filtered_results):.2f} seconds\n")


if __name__ == "__main__":
    main()
