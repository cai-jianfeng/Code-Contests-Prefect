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
    
    def is_time_limit_exceeded(self, response: Dict) -> bool:
        """检查响应是否包含 TimeLimitExceeded 状态"""
        # 检查顶层的 status
        if response.get('status') == 'TLE':
            return True
        
        # 检查 run_result 中的状态
        run_result = response.get('run_result', {})
        if run_result.get('status') == 'TLE' or run_result.get('status') == 'TimeLimitExceeded':
            return True
            
        # 检查 compile_result 中的状态
        compile_result = response.get('compile_result', {})
        if compile_result.get('status') == 'TLE' or compile_result.get('status') == 'TimeLimitExceeded':
            return True
            
        return False
    
    def call_api(self, api_path: str, payload: Dict, max_retries: int = 3, retry_delay: int = 1) -> Dict:
        """调用远程的 sandbox API，支持重试功能"""
        for attempt in range(max_retries + 1):
            try:
                response = self.session.post(api_path, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                # 检查是否有 TimeLimitExceeded
                if self.is_time_limit_exceeded(result):
                    if attempt < max_retries:
                        print(f"TimeLimitExceeded detected on {api_path}, retrying... (attempt {attempt + 1}/{max_retries})")
                        if retry_delay > 0:
                            time.sleep(retry_delay)
                        continue
                    else:
                        print(f"TimeLimitExceeded persists on {api_path} after {max_retries} retries, will requeue")
                        return {'status': 'REQUEUE_NEEDED', 'reason': 'TimeLimitExceeded', 'api_path': api_path}
                
                return result
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    print(f"Request failed on {api_path}, retrying... (attempt {attempt + 1}/{max_retries}): {e}")
                    if retry_delay > 0:
                        time.sleep(retry_delay)
                    continue
                else:
                    print(f"Request failed on {api_path} after {max_retries} retries, will requeue: {e}")
                    return {'status': 'REQUEUE_NEEDED', 'reason': f'Exception: {str(e)}', 'api_path': api_path}
            except Exception as e:
                if attempt < max_retries:
                    print(f"Unexpected error on {api_path}, retrying... (attempt {attempt + 1}/{max_retries}): {e}")
                    if retry_delay > 0:
                        time.sleep(retry_delay)
                    continue
                else:
                    print(f"Unexpected error on {api_path} after {max_retries} retries, will requeue: {e}")
                    return {'status': 'REQUEUE_NEEDED', 'reason': f'Exception: {str(e)}', 'api_path': api_path}
        
        return result


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
        
        # 检查是否需要重新入队
        if isinstance(response, dict) and response.get('status') == 'REQUEUE_NEEDED':
            return {
                'result_id': result_id,
                'solution_idx': solution_idx,
                'solution_type': solution_type,
                'test_idx': test_idx,
                'status': 'requeue_needed',
                'reason': response.get('reason', 'unknown'),
                'failed_api': response.get('api_path', api_path)
            }
        
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


def parallel_checker_validation(results_data: Dict, dataset_data: Dict, api_paths: List[str], 
                               max_workers: int = 4, debug: bool = False) -> Dict:
    """并行执行checker验证"""
    
    def call_fallback_api(task: Dict, exclude_apis: set = None) -> Optional[Dict]:
        """调用备用API处理重新入队的任务"""
        exclude_apis = exclude_apis or set()
        available_apis = [api for api in api_paths if api not in exclude_apis]
        
        if not available_apis:
            print(f"No available fallback APIs for task {task['result_id']}")
            return None
            
        # 选择第一个可用的API作为备用
        fallback_api = available_apis[0]
        print(f"Using fallback API {fallback_api} for task {task['result_id']}")
        
        try:
            fallback_session = SandboxSession([fallback_api])
            task['api_path'] = fallback_api
            result = process_single_test_case(task, fallback_session)
            
            # 如果备用API也需要重新入队，返回None
            if result.get('status') == 'requeue_needed':
                print(f"Fallback API {fallback_api} also failed for task {task['result_id']}")
                return None
                
            return result
        except Exception as e:
            print(f"Fallback API {fallback_api} error for task {task['result_id']}: {e}")
            return None
    
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
        
        # 处理正确解决方案
        solution_results = result_data.get("solution_result", [])
        for sidx, solution in enumerate(solution_results):
            single_result = solution['result']
            if not single_result.get("accepted", True):
                tests = single_result.get("tests", [])
                for idx, test in enumerate(tests):
                    if not test['passed'] and test['exec_info'].get('status') == "Success":
                        task_queue.put({
                            'api_path': None,  # 将在worker中分配
                            'test_info': {'test': test},
                            'checker_code': checker_code,
                            'result_id': result_id,
                            'solution_idx': sidx,
                            'test_idx': idx,
                            'solution_type': 'correct',
                            'retry_count': 0  # 添加重试计数
                        })
                        total_tasks += 1
        
        # 处理错误解决方案
        incorrect_solution_results = result_data.get("incorrect_solution_result", [])
        for sidx, solution in enumerate(incorrect_solution_results):
            single_result = solution['result']
            if not single_result.get("accepted", True):
                tests = single_result.get("tests", [])
                for idx, test in enumerate(tests):
                    if not test['passed'] and test['exec_info'].get('status') == "Success":
                        task_queue.put({
                            'api_path': None,  # 将在worker中分配
                            'test_info': {'test': test},
                            'checker_code': checker_code,
                            'result_id': result_id,
                            'solution_idx': sidx,
                            'test_idx': idx,
                            'solution_type': 'incorrect',
                            'retry_count': 0  # 添加重试计数
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
                
                # 检查是否需要重新入队
                if result.get('status') == 'requeue_needed':
                    max_requeue_attempts = 5  # 最大重新入队次数
                    retry_count = task.get('retry_count', 0)
                    
                    if retry_count < max_requeue_attempts:
                        print(f"Requeuing task for {task['result_id']} due to {result.get('reason', 'unknown reason')} on {result.get('failed_api', api_path)} (attempt {retry_count + 1}/{max_requeue_attempts})")
                        # 增加重试计数并重新放回队列
                        task['retry_count'] = retry_count + 1
                        task_queue.put(task)
                        continue
                    else:
                        print(f"Max requeue attempts reached for {task['result_id']}, trying fallback API")
                        # 达到最大重试次数，尝试使用备用API
                        failed_api = result.get('failed_api', api_path)
                        fallback_result = call_fallback_api(task, exclude_apis={failed_api})
                        
                        if fallback_result and fallback_result.get('status') == 'success':
                            print(f"Fallback API succeeded for {task['result_id']}")
                            result = fallback_result
                        else:
                            print(f"Fallback API also failed for {task['result_id']}, using error result")
                            result = {
                                'result_id': task['result_id'],
                                'solution_idx': task['solution_idx'],
                                'solution_type': task['solution_type'],
                                'test_idx': task['test_idx'],
                                'status': 'failed',
                                'reason': 'max_retries_exceeded_and_fallback_failed'
                            }
                
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
    """将checker验证结果应用到原始数据中"""
    for result_id, checker_results in processing_results.items():
        if result_id not in results_data:
            continue
            
        result_data = results_data[result_id]
        
        # 按solution type分组
        for checker_result in checker_results:
            solution_type = checker_result['solution_type']
            solution_idx = checker_result['solution_idx']
            test_idx = checker_result['test_idx']
            
            if solution_type == 'correct':
                solutions = result_data.get("solution_result", [])
            else:
                solutions = result_data.get("incorrect_solution_result", [])
            
            if solution_idx < len(solutions):
                tests = solutions[solution_idx]['result'].get("tests", [])
                if test_idx < len(tests):
                    test = tests[test_idx]
                    
                    if checker_result['status'] == 'success':
                        test['checker_info'] = checker_result['checker_response']
                    elif checker_result['status'] == 'error':
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


def filter_existing_results(results_data: Dict, dataset_data: Dict, result_folder: str) -> Tuple[Dict, Dict]:
    """过滤已经处理过的结果"""
    filtered_results = {}
    filtered_dataset = {}
    
    for result_id in results_data.keys():
        output_file = os.path.join(result_folder, f"{result_id.replace('/', '_')}_checker.json")
        
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists, skipping.")
            continue
            
        if result_id not in dataset_data:
            print(f"Result ID {result_id} not found in dataset, skipping.")
            continue
            
        checker_code = dataset_data[result_id].get("checker", None)
        if not checker_code:
            print(f"No checker code for result {result_id}, skipping.")
            continue
            
        filtered_results[result_id] = results_data[result_id]
        filtered_dataset[result_id] = dataset_data[result_id]
    
    return filtered_results, filtered_dataset


def main():
    """主函数"""
    # 配置参数
    result_folder = "/aiarena/group/llmgroup/caijf/final_ut_gen/code_contests_ours_all_solutions_o4mini_10maxsample/"
    data_folder = "/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_o4mini_10maxsample/"
    
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
    dataset = load_results(data_folder)
    # dataset = dataset_read(data_folder, "codecontents", split=None)
    # construct_dataset = {item['id']: item for item in dataset}
    # dataset = construct_dataset
    if not results:
        print("No results found. Exiting.")
        return
    
    if not dataset:
        print("No dataset found. Exiting.")
        return
    
    # 过滤已处理的结果
    print("Filtering already processed results...")
    filtered_results, filtered_dataset = filter_existing_results(results, dataset, result_folder)
    
    if not filtered_results:
        print("All samples have already been processed. No new tasks to run.")
        return
    
    print(f"Found {len(filtered_results)} samples to process out of {len(results)} total.")
    
    # 开始处理
    start_time = time.time()
    
    print("Starting parallel checker validation...")
    processing_results = parallel_checker_validation(
        filtered_results, 
        filtered_dataset, 
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
