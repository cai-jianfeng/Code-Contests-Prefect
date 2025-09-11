"""
该代码的主要流程是:
1. 设计一个 Prompt 来引导 LLM 生成特定的 corner case;
2. 通过给定程序 generation 来调用 OpenAI 的 API 生成用于生成 corner case 的 command list;
3. 通过执行生成的 command 获得 corner cases；
4. 调用远程的 sandbox API 来验证生成的 corner case 是否正确:
    4.1: 将生成的 corner case 和给定的 solution/incorrect_solution 一起提交到 sandbox;
    4.2: sandbox 会运行给定的 solution/incorrect_solution 并返回结果;
    4.3: 根据 sandbox 返回的结果来判断 corner case 是否正确;
5. 基于并行处理架构，支持多个 API 端点同时处理不同的样本
"""

# %% set up
import os
import json
import requests
import sys
from tqdm import tqdm
from openai import OpenAI
# import openai
from datasets import load_dataset, Dataset
import ast
import base64
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import queue
import gc
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
from pydantic import BaseModel

import re

class Command_Model(BaseModel):
    replace_command_list: list[str]
    add_command_list: list[str]

class Init_Command_Model(BaseModel):
    input_constraints_summary: str
    command_list: list[str]

_global_truncated_length = 1000

# 全局变量
completed_tasks = 0  # 全局计数器，用于跟踪已完成的任务数量

# 全局 API 队列和锁
_global_api_queue = None
_global_api_lock = threading.Lock()

_global_specific_api_queue = None
_global_specific_api_lock = threading.Lock()

# 全局日志系统
_global_logger = None
_logger_lock = threading.Lock()

testlib_path = "testlib.h"
with open(testlib_path, 'rb') as file:
    testlib_data_b64 = base64.b64encode(file.read()).decode('utf-8')

_global_files = {
    "testlib.h": testlib_data_b64
}

class LoggerManager:
    """日志管理器，用于管理全局日志和样本日志"""
    
    def __init__(self, results_dir: str = None):
        self.results_dir = results_dir
        self.log_dir = None
        self.global_logger = None
        self.sample_loggers = {}
        self.logger_lock = threading.Lock()
        
        if self.results_dir:
            self.setup_log_directory()
            self.setup_global_logger()
    
    def setup_log_directory(self):
        """设置日志目录"""
        if self.results_dir:
            self.log_dir = os.path.join(self.results_dir, "log")
            os.makedirs(self.log_dir, exist_ok=True)
    
    def setup_global_logger(self):
        """设置全局日志器"""
        if not self.log_dir:
            return
            
        global_log_path = os.path.join(self.log_dir, "global.log")
        
        # 创建全局日志器
        self.global_logger = logging.getLogger('global')
        self.global_logger.setLevel(logging.INFO)
        
        # 清除现有的处理器
        for handler in self.global_logger.handlers[:]:
            self.global_logger.removeHandler(handler)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(global_log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        self.global_logger.addHandler(file_handler)
        
        # 防止传播到根日志器
        self.global_logger.propagate = False
    
    def get_sample_logger(self, sample_id: str):
        """获取或创建样本日志器"""
        if not self.log_dir:
            return self.global_logger
            
        with self.logger_lock:
            if sample_id not in self.sample_loggers:
                # 清理样本ID用作文件名
                safe_sample_id = sample_id.replace('/', '_').replace('\\', '_')
                sample_log_path = os.path.join(self.log_dir, f"{safe_sample_id}.log")
                
                # 创建样本日志器
                logger = logging.getLogger(f'sample_{safe_sample_id}')
                logger.setLevel(logging.INFO)
                
                # 清除现有的处理器
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                
                # 创建文件处理器
                file_handler = logging.FileHandler(sample_log_path, mode='a', encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                
                # 创建格式器
                formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
                file_handler.setFormatter(formatter)
                
                # 添加处理器
                logger.addHandler(file_handler)
                
                # 防止传播到根日志器
                logger.propagate = False
                
                self.sample_loggers[sample_id] = logger
            
            return self.sample_loggers[sample_id]
    
    def log_global(self, message: str):
        """记录全局日志"""
        if self.global_logger:
            self.global_logger.info(message)
        else:
            # 如果没有设置日志器，回退到 print
            print(f"[GLOBAL] {message}")
    
    def log_sample(self, sample_id: str, message: str):
        """记录样本日志"""
        logger = self.get_sample_logger(sample_id)
        if logger:
            logger.info(message)
        else:
            # 如果没有设置日志器，回退到 print
            print(f"[{sample_id}] {message}")
    
    def cleanup(self):
        """清理日志器资源"""
        with self.logger_lock:
            # 关闭全局日志器
            if self.global_logger:
                for handler in self.global_logger.handlers:
                    handler.close()
                    self.global_logger.removeHandler(handler)
            
            # 关闭样本日志器
            for logger in self.sample_loggers.values():
                for handler in logger.handlers:
                    handler.close()
                    logger.removeHandler(handler)
            
            self.sample_loggers.clear()

def get_global_logger_manager():
    """获取全局日志管理器"""
    global _global_logger
    return _global_logger

def initialize_logger_manager(results_dir: str = None):
    """初始化全局日志管理器"""
    global _global_logger
    with _logger_lock:
        if _global_logger is None:
            _global_logger = LoggerManager(results_dir)

def log_global(message: str):
    """记录全局日志的便捷函数"""
    logger_manager = get_global_logger_manager()
    if logger_manager:
        logger_manager.log_global(message)
    else:
        print(f"[GLOBAL] {message}")

def log_sample(sample_id: str, message: str):
    """记录样本日志的便捷函数"""
    logger_manager = get_global_logger_manager()
    if logger_manager:
        logger_manager.log_sample(sample_id, message)
    else:
        print(f"[{sample_id}] {message}")

def initialize_global_api_queue(api_paths: List[str]):
    """初始化全局 API 队列"""
    global _global_api_queue
    with _global_api_lock:
        if _global_api_queue is None:
            _global_api_queue = queue.Queue()
            for api_path in api_paths:
                _global_api_queue.put(api_path)
    
def initialize_global_specific_api_queue(specific_api_paths: List[str]):
    """初始化全局特定 API 队列"""
    global _global_specific_api_queue
    with _global_specific_api_lock:
        if _global_specific_api_queue is None:
            _global_specific_api_queue = queue.Queue()
            for api_path in specific_api_paths:
                _global_specific_api_queue.put(api_path)

def get_api_path(timeout: float = 0.1) -> Optional[str]:
    """从全局队列获取 API 路径"""
    global _global_api_queue
    if _global_api_queue is None:
        return None
    try:
        return _global_api_queue.get(timeout=timeout)
    except queue.Empty:
        return None

def return_api_path(api_path: str):
    """将 API 路径归还到全局队列"""
    global _global_api_queue
    if _global_api_queue is not None and api_path is not None:
        _global_api_queue.put(api_path)

def get_specific_api_path(timeout: float = 0.1) -> Optional[str]:
    """从全局特定队列获取 API 路径"""
    global _global_specific_api_queue
    if _global_specific_api_queue is None:
        return None
    try:
        return _global_specific_api_queue.get(timeout=timeout)
    except queue.Empty:
        return None

def return_specific_api_path(api_path: str):
    """将特定 API 路径归还到全局队列"""
    global _global_specific_api_queue
    if _global_specific_api_queue is not None and api_path is not None:
        _global_specific_api_queue.put(api_path)

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

# 设置 OpenAI API 密钥
API_BASE = "https://lonlie.plus7.plus/v1"
API_KEY = "sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

# 常量定义
LANGUAGE = ["UNKNOWN_LANGUAGE", "PYTHON", "CPP", "PYTHON3", "JAVA"]

PROMPT_TEMPLATE = """
You are an expert in generating command-line arguments for corner case generation programs for programming problems.

Given the following problem statement and a C++ generation program, your tasks are:
1. Carefully read and understand the problem statement.
2. Carefully read and understand the provided generation program, which is designed to generate corner case inputs for this problem.
3. Identify and summarize the constraints of the input data.
4. Analyze the problem and the generation program to anticipate common mistakes or edge cases that contestants might overlook.
5. Based on your analysis, design and output a diverse set of command-line commands ("command_list") that, when executed, will use the generation program to generate corner case inputs that cover as many special and adversarial cases as possible.

Problem Statement:
{problem_statement}

Generation Program (C++):
{generator}

**Strictly follow these output requirements:**
- Your response must be in JSON format matching this structure:
    {{
        "input_constraints_summary": "string describing input constraints from the problem statement",
        "command_list": ["./gen --arg1 value1 ...", "./gen --arg2 value2 ...", ...]
    }}
- The "input_constraints_summary" field should contain a clear and concise summary of all input constraints, including both explicit constraints mentioned in the problem statement (such as input size limits, value ranges, format requirements, etc.) and any implicit constraints that can be inferred from the problem description (such as properties, invariants, or hidden requirements implied by the problem context).
- The "command_list" field must contain a list of shell commands, each starting with './gen' and followed by the appropriate arguments for the generation program. Each command should be designed to generate one corner case input. All corner case inputs generated by these commands should be as diverse and adversarial as possible, covering a wide range of edge cases and adversarial scenarios.
- Do not generate the corner case inputs directly; only generate the command lines to run the generation program.
- The commands should be ready to execute in a Linux shell and should use proper argument formatting as required by the generation program.
"""

ADD_PROMPT_TEMPLATE = """
Now you need to refine the previously generated command list for the corner case generation program based on evaluation feedback.

You previously generated a set of commands for the given programming problem. The process is as follows:
1. Each command is executed to generate one or more corner case inputs.
2. For each generated corner case input, the canonical solution is executed to obtain the corresponding output, thus forming a complete corner case (input + output).
3. These corner cases are then used to evaluate both correct solutions and incorrect solutions.

Current command list: {current_command_list}

For each command, here are the corresponding generated corner case input(s) (for some commands that generate very long inputs, the input for that command is replaced by `[input]`):
{command_to_input_map}

If any command failed to execute or produced errors when generating input, here are the error messages (if any):
{command_run_errors} (ideally, this should be empty)

The evaluation results are as follows (ideally, all three should be empty):
- Outputs from correct solutions: These are cases where the generated corner cases incorrectly cause correct solutions to fail (i.e., the correct solution is judged as wrong on these cases). {correct_results}
- Outputs from incorrect solutions: These are cases where the generated corner cases fail to expose bugs in incorrect solutions (i.e., the incorrect solution is judged as correct on these cases). {incorrect_results}
- Outputs from the canonical solution (only includes results for cases that failed when run with the canonical solution): These are cases where the canonical solution itself fails or produces errors on the generated corner cases. {outputs}

Please note:
- For some commands that generate very long inputs, the `stdin` field in `correct_results`/`incorrect_results`/`outputs` may be replaced by the corresponding command string, and the field will have a trailing ` [command]` tag to indicate this substitution. When you see such a `stdin` value, you should use the provided mapping between commands and generated inputs to implicitly convert the command back to its actual `stdin` content for any reasoning, comparison, or decision-making tasks.
- For some cases where the output (`stdout`/`expected_output`) is very long, the `stdout`/`expected_output` field may be replaced by `[output]`/`[expected output]`. When you see `[output]`/`[expected output]`, in this case, if the solution's `passed` field is False, you should rely only on the given solution content for reasoning.

Here is a clear and concise summary of the input constraints mentioned in the problem statement (e.g., input size limits, value ranges, format requirements, etc.): {input_constraints_summary}

Your tasks are:
1. Based on the above canonical solution results, identify any commands that generate invalid or unhelpful corner cases (i.e., those that fail when run with the canonical solution) and mark them for replacement.
2. Based on the correct solutions results, identify commands that generate corner cases which incorrectly classify correct solutions as wrong, and mark them for replacement.
3. Analyze the above results to determine which commands fail to effectively distinguish between correct and incorrect solutions.
4. Generate new additional commands that can better expose bugs in incorrect solutions and improve differentiation between correct and incorrect solutions.

**Strictly follow these output requirements:**
- Your response must be in JSON format matching this structure:
    {{
        "replace_command_list": ["old_command_1", "old_command_2", ...],
        "add_command_list": ["new_command_1", "new_command_2", ...]
    }}
- `replace_command_list` contains commands from the original list that should be removed/replaced due to generating invalid or unhelpful corner cases, or incorrectly classifying correct solutions as wrong.
- `add_command_list` contains new commands to be added to better distinguish correct and incorrect solutions, including improved versions of replaced commands and completely new adversarial commands.
- Each command should be a shell command starting with './gen' and followed by the appropriate arguments for the generation program.
- Do not generate the corner case inputs directly; only generate the command lines to run the generation program.
- The commands should be ready to execute in a Linux shell and should use proper argument formatting as required by the generation program.

Please focus on maximizing the adversarial value of the generated corner cases based on the feedback above.
"""

# ADD_PROMPT_TEMPLATE = """
# Now you need to refine the previously generated command list for the corner case generation program based on evaluation feedback.

# You previously generated a set of commands for the given programming problem. The process is as follows:
# 1. Each command is executed to generate one or more corner case inputs.
# 2. For each generated corner case input, the canonical solution is executed to obtain the corresponding output, thus forming a complete corner case (input + output).
# 3. These corner cases are then used to evaluate both correct solutions and incorrect solutions.

# Current command list: {current_command_list}

# If any command failed to execute or produced errors when generating input, here are the error messages (if any):
# {command_run_errors} (ideally, this should be empty)

# The evaluation results are as follows (ideally, all three should be empty):
# - Outputs from correct solutions: These are cases where the generated corner cases incorrectly cause correct solutions to fail (i.e., the correct solution is judged as wrong on these cases). {correct_results}
# - Outputs from incorrect solutions: These are cases where the generated corner cases fail to expose bugs in incorrect solutions (i.e., the incorrect solution is judged as correct on these cases). {incorrect_results}
# - Outputs from the canonical solution (only includes results for cases that failed when run with the canonical solution): These are cases where the canonical solution itself fails or produces errors on the generated corner cases. {outputs}

# Please note: the `stdin` field in `correct_results`/`incorrect_results`/`outputs` is replaced by the corresponding command string, and the field will have a trailing ` [command]` tag to indicate this substitution. When you see such a `stdin` value, you should use the provided mapping between commands and generated inputs to implicitly convert the command back to its actual `stdin` content for any reasoning, comparison, or decision-making tasks.

# Here is a clear and concise summary of the input constraints mentioned in the problem statement (e.g., input size limits, value ranges, format requirements, etc.): {input_constraints_summary}

# Your tasks are:
# 1. Based on the above canonical solution results, identify any commands that generate invalid or unhelpful corner cases (i.e., those that fail when run with the canonical solution) and mark them for replacement.
# 2. Based on the correct solutions results, identify commands that generate corner cases which incorrectly classify correct solutions as wrong, and mark them for replacement.
# 3. Analyze the above results to determine which commands fail to effectively distinguish between correct and incorrect solutions.
# 4. Generate new additional commands that can better expose bugs in incorrect solutions and improve differentiation between correct and incorrect solutions.

# **Strictly follow these output requirements:**
# - Your response must be in JSON format matching this structure:
#     {{
#         "replace_command_list": ["old_command_1", "old_command_2", ...],
#         "add_command_list": ["new_command_1", "new_command_2", ...]
#     }}
# - `replace_command_list` contains commands from the original list that should be removed/replaced due to generating invalid or unhelpful corner cases, or incorrectly classifying correct solutions as wrong.
# - `add_command_list` contains new commands to be added to better distinguish correct and incorrect solutions, including improved versions of replaced commands and completely new adversarial commands.
# - Each command should be a shell command starting with './gen' and followed by the appropriate arguments for the generation program.
# - Do not generate the corner case inputs directly; only generate the command lines to run the generation program.
# - The commands should be ready to execute in a Linux shell and should use proper argument formatting as required by the generation program.

# Please focus on maximizing the adversarial value of the generated corner cases based on the feedback above.
# """

INIT_RESPONSE_TEMPLATE = """{{
    "input_constraints_summary": [{input_constraints_summary}],
    "command_list": [{command_list}]
}}"""

RESPONSE_TEMPLATE = """{{
    "replace_command_list": [{replace_command_list}],
    "add_command_list": [{add_command_list}]
}}"""

SOLUTION_RESULT_TEMPLATE = """
language: {language},
solution: {solution},
output: {output}
"""

TEST_CASE_RESULT_TEMPLATE = "passed: {passed}; stdin: {stdin}; stdout: {stdout}; expected_output: {expected_output}"

CANONICAL_SOLUTION_TEMPLATE = """
canonical_solution_language: {language}, 
canonical_solution: {solution},
stdin: {stdin},
output: {output}
"""

TEMPLATE = """
Here is a {language} solution to the problem: 
```{language}
{solution}
```
"""

REASONING_MODELS = ["o3-mini", "o4-mini", "gpt-5"]
NO_REASONING_MODELS = ["gpt-4o"]

# %% 核心功能类
class OpenAIClient:
    """OpenAI API 客户端封装"""
    
    def __init__(self, api_base: str, api_key: str, model: str = "gpt-4o", max_tokens: int = 1000, no_reasoning: bool = True, max_attempts: int = 3):
        
        
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.no_reasoning = no_reasoning
        self.max_attempts = max_attempts
        self.extra_args = {}
        if not no_reasoning:
            assert model in REASONING_MODELS, f"Model {model} does not support reasoning mode."
            # self.extra_args = {
            #     "reasoning": {
            #         "effort": "low",
            #         # "summary": "auto",
            #     },
            #     "text": {
            #         "verbosity": "medium"
            #     }
            # }
            
        else:
            assert model in NO_REASONING_MODELS, f"Model {model} does not support no reasoning mode."


    def generate_command(self, messages: List[Dict], first: bool, sample_id: str) -> Union[Init_Command_Model, Command_Model]:
        """
        使用 OpenAI API 生成 commands。
        """
        if first:
            command_model = Init_Command_Model
        else:
            command_model = Command_Model
        for _ in range(self.max_attempts):
            try:
                if self.no_reasoning:
                    response = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        response_format=command_model,
                        timeout=100,
                    )
                else:
                    response = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        response_format=command_model,
                        reasoning_effort="low",
                        verbosity="medium",
                        timeout=100,
                    )
                return response.choices[0].message.parsed  
            except Exception as e:
                log_sample(sample_id, f"Error generating corner case: {e}")
                continue
    
        return None


class SandboxClient:
    """Sandbox API 客户端封装"""
    
    def __init__(self):
        self.session = requests.Session()
    
    def call_api(self, api_path: str, json_data: Dict, max_retries: int = 3, retry_delay: float = 1, sample_id=None) -> Dict:
        """
        调用远程的 sandbox API，支持 TimeLimitExceeded 处理
        当遇到 TimeLimitExceeded 时会进行智能重试
        
        根据不同的API端点和请求类型，适配不同的TimeLimitExceeded检测逻辑：
        1. /submit 端点: 完整的解决方案验证，有tests数组
        2. /run_code 端点: 代码执行，直接返回执行结果
        """
        MAX_TIME = 1000  # 最大超时时间限制
        
        # 识别API类型
        is_submit_api = api_path.endswith('/submit')
        is_run_code_api = api_path.endswith('/run_code')
        
        # 记录原始超时配置
        if is_submit_api:
            config = json_data.get('config', {})
            original_run_timeout = config.get('run_timeout', 20)
            original_compile_timeout = config.get('compile_timeout', 20)
        else:
            # run_code API 没有 config 结构
            original_run_timeout = 20
            original_compile_timeout = 20
            
        timeout_increase_attempts = 0
        max_timeout_attempts = 3
        
        for attempt in range(max_retries + 1):
            try:
                with self.session.post(api_path, json=json_data) as res:
                    response = res.json()
                    
                    # 根据API类型检查TimeLimitExceeded
                    tle_type = self._detect_timeout_by_api_type(response, is_submit_api, is_run_code_api)
                    
                    if tle_type:
                        if tle_type == "real_timeout" and timeout_increase_attempts < max_timeout_attempts:
                            # 智能分析输出并调整超时时间
                            should_retry, new_timeout_config = self._analyze_timeout_and_adjust(
                                response, json_data, original_run_timeout, original_compile_timeout,
                                is_submit_api, is_run_code_api, MAX_TIME, sample_id
                            )
                            
                            if should_retry:
                                # 应用新的超时配置
                                if is_submit_api and new_timeout_config:
                                    config.update(new_timeout_config)
                                    json_data['config'] = config
                                # run_code API 暂时无法调整超时，直接重试
                                
                                timeout_increase_attempts += 1
                                continue
                            else:
                                # 没有找到合适的重试条件，尝试使用特定API
                                return self._retry_with_specific_api(json_data, max_retries=3, sample_id=sample_id)
                                
                        elif tle_type == "sandbox_blocked":
                            # Sandbox 内部阻塞，尝试使用特定API重试
                            if attempt < max_retries:
                                if sample_id:
                                    log_sample(sample_id, f"Sandbox blocked detected {api_path}, retrying... (attempt {attempt + 1}/{max_retries})")
                                else:
                                    log_global(f"Sandbox blocked detected {api_path}, retrying... (attempt {attempt + 1}/{max_retries})")
                                if retry_delay > 0:
                                    time.sleep(retry_delay)
                                continue
                            else:
                                if sample_id:
                                    log_sample(sample_id, f"Sandbox blocking {api_path} persists after {max_retries} retries, trying specific API")
                                else:
                                    log_global(f"Sandbox blocking {api_path} persists after {max_retries} retries, trying specific API")
                                return self._retry_with_specific_api(json_data, max_retries=3, sample_id=sample_id)
                        else:
                            # 其他超时情况或已达到最大超时尝试次数
                            if sample_id:
                                log_sample(sample_id, f"TimeLimitExceeded persists after attempts ({api_path}), trying specific API")
                            else:
                                log_global(f"TimeLimitExceeded persists after attempts ({api_path}), trying specific API")
                            return self._retry_with_specific_api(json_data, max_retries=3, sample_id=sample_id)
                    
                    return response
                    
            except Exception as e:
                if attempt < max_retries:
                    if sample_id:
                        log_sample(sample_id, f"Request failed ({api_path}), retrying... (attempt {attempt + 1}/{max_retries}): {e}")
                    else:
                        log_global(f"Request failed ({api_path}), retrying... (attempt {attempt + 1}/{max_retries}): {e}")
                    if retry_delay > 0:
                        time.sleep(retry_delay)
                    continue
                else:
                    if sample_id:
                        log_sample(sample_id, f"Request failed ({api_path}) after {max_retries} retries: {e}")
                    else:
                        log_global(f"Request failed ({api_path}) after {max_retries} retries: {e}")
                    return {"error": str(e)}
        
        return response
    
    def _detect_timeout_by_api_type(self, response: Dict, is_submit_api: bool, is_run_code_api: bool) -> Union[bool, str]:
        """
        根据API类型检测TimeLimitExceeded
        """
        if is_submit_api:
            # /submit API: 使用原有的is_time_limit_exceeded逻辑
            return is_time_limit_exceeded(response)
        elif is_run_code_api:
            # /run_code API: 检查顶层的执行结果
            if response.get('status') != 'Success':
                compile_result = response.get('compile_result', {})
                run_result = response.get('run_result', {})
                
                # 检查编译或运行是否超时
                if (isinstance(compile_result, dict) and compile_result.get('status') == 'TimeLimitExceeded') or (isinstance(run_result, dict) and run_result.get('status') == 'TimeLimitExceeded'):
                    
                    # 检查是否有实际输出
                    if run_result.get('stdout'):
                        return "real_timeout"
                    else:
                        return "sandbox_blocked"
        
        return False
    
    def _analyze_timeout_and_adjust(self, response: Dict, json_data: Dict, 
                                  original_run_timeout: int, original_compile_timeout: int,
                                  is_submit_api: bool, is_run_code_api: bool, 
                                  MAX_TIME: int, sample_id: str = None) -> Tuple[bool, Optional[Dict]]:
        """
        分析超时情况并计算新的超时配置
        """
        should_retry = False
        new_timeout_config = None
        
        if is_submit_api:
            # /submit API: 分析tests中的超时情况
            for test in response.get("tests", []):
                exec_info = test.get('exec_info', {})
                run_result = exec_info.get('run_result', {})
                
                if isinstance(run_result, dict) and run_result.get('status') == 'TimeLimitExceeded':
                    actual_stdout = run_result.get('stdout', '')
                    
                    # 从config中获取期望输出
                    config = json_data.get('config', {})
                    provided_data = config.get('provided_data', {})
                    test_cases = provided_data.get('test', [])
                    
                    # 找到对应的测试用例期望输出
                    if test_cases:
                        test_index = response.get("tests", []).index(test)
                        if test_index < len(test_cases):
                            expected_stdout = test_cases[test_index].get('output', {}).get('stdout', '')
                            
                            is_partial, multiplier = calculate_timeout_multiplier(actual_stdout, expected_stdout)
                            
                            if is_partial:
                                new_run_timeout = int(original_run_timeout * multiplier)
                                new_compile_timeout = int(original_compile_timeout * multiplier)
                                
                                if new_run_timeout <= MAX_TIME:
                                    new_timeout_config = {
                                        'run_timeout': new_run_timeout,
                                        'compile_timeout': new_compile_timeout
                                    }
                                    if sample_id:
                                        log_sample(sample_id, f"Submit API: Real timeout with partial output, increasing timeout by {multiplier}x to {new_run_timeout}s")
                                    else:
                                        log_global(f"Submit API: Real timeout with partial output, increasing timeout by {multiplier}x to {new_run_timeout}s")
                                    should_retry = True
                                    break
                                else:
                                    if sample_id:
                                        log_sample(sample_id, f"Submit API: Timeout would exceed MAX_TIME ({MAX_TIME}s), not retrying")
                                    else:
                                        log_global(f"Submit API: Timeout would exceed MAX_TIME ({MAX_TIME}s), not retrying")
                                    should_retry = False
                            else:
                                if sample_id:
                                    log_sample(sample_id, f"Submit API: Real timeout with mismatched output, not retrying")
                                else:
                                    log_global(f"Submit API: Real timeout with mismatched output, not retrying")
                                return False, None
                                
        elif is_run_code_api:
            # /run_code API: 分析顶层运行结果
            run_result = response.get('run_result', {})
            if isinstance(run_result, dict) and run_result.get('status') == 'TimeLimitExceeded':
                actual_stdout = run_result.get('stdout', '')
                
                # 对于run_code API，我们没有期望输出进行比较
                # 如果有输出，说明程序在运行，可以重试
                if actual_stdout and actual_stdout.strip():
                    if sample_id:
                        log_sample(sample_id, f"Run code API: Real timeout with output, will retry with specific API")
                    else:
                        log_global(f"Run code API: Real timeout with output, will retry with specific API")
                    should_retry = True
                else:
                    if sample_id:
                        log_sample(sample_id, f"Run code API: Real timeout without output, likely sandbox issue")
                    else:
                        log_global(f"Run code API: Real timeout without output, likely sandbox issue")
                    should_retry = False
        
        return should_retry, new_timeout_config
    
    def _retry_with_specific_api(self, json_data: Dict, max_retries: int = 3, sample_id: str = None) -> Dict:
        """使用特定API队列重试请求"""
        for attempt in range(max_retries):
            specific_api_path = get_specific_api_path(timeout=0.1)
            if specific_api_path is None:
                if sample_id:
                    log_sample(sample_id, f"No specific API available for retry attempt {attempt + 1}")
                else:
                    log_global(f"No specific API available for retry attempt {attempt + 1}")
                time.sleep(1)
                continue
            
            try:
                if sample_id:
                    log_sample(sample_id, f"Retrying with specific API: {specific_api_path} (attempt {attempt + 1}/{max_retries})")
                else:
                    log_global(f"Retrying with specific API: {specific_api_path} (attempt {attempt + 1}/{max_retries})")
                with self.session.post(specific_api_path, json=json_data) as res:
                    response = res.json()
                    
                    # 检查新的响应是否还有TimeLimitExceeded
                    tle_type = is_time_limit_exceeded(response)
                    if not tle_type:
                        return response
                    else:
                        if sample_id:
                            log_sample(sample_id, f"Specific API also has TimeLimitExceeded: {tle_type}")
                        else:
                            log_global(f"Specific API also has TimeLimitExceeded: {tle_type}")
                        
            except Exception as e:
                if sample_id:
                    log_sample(sample_id, f"Specific API request failed: {e}")
                else:
                    log_global(f"Specific API request failed: {e}")
            finally:
                # 将API路径归还到队列
                return_specific_api_path(specific_api_path)
        
        # 如果所有特定API重试都失败，返回错误
        return {"error": "All retry attempts failed with TimeLimitExceeded"}


class DatasetProcessor:
    """数据集处理器"""
    
    @staticmethod
    def read_dataset(data_path: str, split: str, transform: bool = False) -> List[Dict]:
        """读取 codecontests 数据集"""
        try:
            if os.path.isdir(data_path):
                data = Dataset.load_from_disk(data_path)
            else:
                data = load_dataset(data_path, split=split)
            # data = list(data)
            
            if transform:
                # 将其转化为 CommonOJ 格式
                format_data = []
                for sample in tqdm(data, desc="Processing dataset"):
                    # format_sample = DatasetProcessor.transform_codecontents(sample)
                    format_sample = sample
                    format_data.append(format_sample)
                return format_data
            else:
                return data
        except Exception as e:
            log_global(f"Error reading dataset: {e}")
            return []
    
    @staticmethod
    def transform_codecontents(sample: Dict) -> Dict:
        """转换数据格式"""
        format_sample = dict(sample)
        
        # 处理 ID
        if 'name' in sample:
            format_sample['id'] = "/".join(['Codeforces'] + sample['name'].split('.')[0].split('_'))
            format_sample['content'] = '# ' + sample['name'].split('.')[-1].strip() + '\n\n' + sample['description']
            format_sample['labels'] = {
                "tag": sample.get('cf_tags', []),
                "title": sample['name'].split('.')[-1].strip()
            }
        
        # 处理 canonical solution
        canonical_solution = {}
        if 'solutions' in sample:
            for i in range(len(LANGUAGE)):
                if i in sample['solutions']['language']:
                    lang = LANGUAGE[i]
                    if "PYTHON" in lang:
                        canonical_solution['python'] = sample['solutions']['solution'][sample['solutions']['language'].index(i)]
                    else:
                        canonical_solution[lang.lower()] = sample['solutions']['solution'][sample['solutions']['language'].index(i)]
        
        format_sample['canonical_solution'] = canonical_solution
        
        # 处理测试用例
        test_cases = []
        for test_type in ['public_tests', 'private_tests', 'generated_tests']:
            if test_type in sample:
                DatasetProcessor.transform_test_cases(sample[test_type], test_cases)
        
        format_sample['test'] = test_cases
        return format_sample
    
    @staticmethod
    def transform_test_cases(test_cases: Dict, format_test_cases: List):
        """转换测试用例格式"""
        for input_data, output_data in zip(test_cases['input'], test_cases['output']):
            format_test_cases.append({
                'input': {'stdin': input_data},
                'output': {'stdout': output_data}
            })


class CornerCaseGenerator:
    """Corner case 生成器"""
    
    def __init__(self, openai_client: OpenAIClient, sandbox_client: SandboxClient, config_manager=None):
        self.openai_client = openai_client
        self.sandbox_client = sandbox_client
        self.config_manager = config_manager
        self.max_iterations = config_manager.processing_config.max_iterations if config_manager else 3
        self.max_sample_solutions = config_manager.processing_config.max_sample_solutions if config_manager else 3
        self.use_all_solutions = config_manager.processing_config.use_all_solutions if config_manager else False

    def parse_corner_cases(self, case_inputs_original: str, sample_id: str = None) -> Optional[List[str]]:
        """解析生成的 corner cases"""
        try:
            case_inputs = ast.literal_eval(case_inputs_original)
        except (SyntaxError, ValueError) as e:
            if sample_id:
                log_sample(sample_id, f"Error parsing corner case with ast.literal_eval: {e}")
                log_sample(sample_id, f"Generated content: {case_inputs_original}")
            else:
                log_global(f"Error parsing corner case with ast.literal_eval: {e}")
                log_global(f"Generated content: {case_inputs_original}")
            
            try:
                # 创建受限的执行环境
                restricted_globals = {
                    "__builtins__": {},
                    "range": range,
                    "len": len,
                    "str": str,
                    "int": int,
                    "join": str.join,
                }
                case_inputs = eval(case_inputs_original, restricted_globals, {})
                if sample_id:
                    log_sample(sample_id, f"Successfully parsed using restricted eval: {len(case_inputs)} cases")
                else:
                    log_global(f"Successfully parsed using restricted eval: {len(case_inputs)} cases")
            except Exception as eval_error:
                if sample_id:
                    log_sample(sample_id, f"Error parsing with restricted eval: {eval_error}")
                else:
                    log_global(f"Error parsing with restricted eval: {eval_error}")
                return None
        
        # 验证解析结果
        if not isinstance(case_inputs, list) or not all(isinstance(item, str) for item in case_inputs):
            if sample_id:
                log_sample(sample_id, "Parsed result is not a valid list of strings")
            else:
                log_global("Parsed result is not a valid list of strings")
            return None
        
        return case_inputs

    def parse_commands_from_Command_Model(self, command_original: List[str], command_new: Union[Init_Command_Model, Command_Model], sample_id: str = None, first: bool = False) -> Optional[List[str]]:
        if first:
            return self.parse_commands_from_Command_Model_first(command_original, command_new, sample_id)
        else:
            return self.parse_commands_from_Command_Model_continue(command_original, command_new, sample_id)

    def parse_commands_from_Command_Model_first(self, command_original: List[str], command_new: Init_Command_Model, sample_id: str = None) -> Optional[List[str]]:
        """解析生成的 commands"""
        # 添加新的 commands
        command_original.extend(command_new.command_list)
        return command_original

    def parse_commands_from_Command_Model_continue(self, command_original: List[str], command_new: Command_Model, sample_id: str = None) -> Optional[List[str]]:
        """解析生成的 corner cases，删除需要替换的并添加新的"""
        # 检查需要替换的 corner case 是否都在原始列表中
        missing_commands = [command for command in command_new.replace_command_list if command not in command_original]
        if missing_commands:
            if sample_id:
                log_sample(sample_id, f"Warning: The following commands to replace are not in the original list: {missing_commands}")
            else:
                log_global(f"Warning: The following commands to replace are not in the original list: {missing_commands}")

        # 删除需要替换的 commands
        commands = [command for command in command_original if command not in command_new.replace_command_list]
        # 添加新的 corner case
        commands.extend(command_new.add_command_list)
        return commands
    
    def generate_test_inputs(self, commands: List[str], sample: Dict, api_paths: List[str], max_workers: int = 4) -> Tuple[List[Dict], List[str]]:
        """给定 commands 生成输入 - 使用全局API队列的并行版本"""
        sample_id = sample.get('id', 'unknown')
        corner_case_inputs = []
        corner_case_inputs_error = []

        generator = sample['generator']
        
        # 创建任务队列
        task_queue = queue.Queue()
        for i, command in enumerate(commands):
            task_queue.put((i, command))
        
        # 结果存储
        results_lock = threading.Lock()
        results = {}
        
        def worker(worker_id: str):
            """工作线程函数 - 使用全局API队列"""
            processed_count = 0
            current_api_path = None
            
            while True:
                try:
                    i, command = task_queue.get(timeout=1)
                except queue.Empty:
                    # 任务完成后，将使用的 API 路径归还到全局队列
                    if current_api_path:
                        return_api_path(current_api_path)
                    break
                
                # 从全局队列获取 API 路径
                if current_api_path is None:
                    current_api_path = get_api_path(timeout=0.1)
                    if current_api_path is None:
                        # 如果所有 API 都在使用中，等待一下再重试
                        task_queue.put((i, command))  # 将任务放回队列
                        time.sleep(0.1)
                        continue
                
                try:
                    sandbox_run_api_path = current_api_path + "run_code"
                    
                    error = None
                    success = False
                    result_case_input = None

                    try:
                        global _global_files
                        
                        payload = {
                            "code": generator,
                            "language": 'cpp',
                            "extra_args": command.replace("./gen ", ""),
                            "files": _global_files,
                            "run_timeout": 40
                        }
                        case_response_input = self.sandbox_client.call_api(sandbox_run_api_path, payload, sample_id=sample_id)
                        if "error" in case_response_input:
                            error = case_response_input
                        
                        elif case_response_input.get('status') == "Success":
                            result_case_input = case_response_input.get('run_result', {}).get('stdout', '')
                            success = True
                    except Exception as gen_error:
                        log_sample(sample_id, f"Error running generator for command {command} on {current_api_path}: {gen_error}")
                    
                    with results_lock:
                        if success:
                            results[i] = ('success', [command, result_case_input])
                        else:
                            if error:
                                if "error" in error:
                                    error_msg = f"command: {command}; API Error: {error['error']}"
                                else:
                                    compile_result = error.get('compile_result', {})
                                    run_result = error.get('run_result', {})
                                    error_msg = (
                                        f"command: {command}; "
                                        f"Compile error (code {compile_result.get('return_code', '')}): {compile_result.get('stderr', '')}; "
                                        f"Runtime error (code {run_result.get('return_code', '')}): {run_result.get('stderr', '')}"
                                    )
                            else:
                                error_msg = f"command: {command}; Unknown error occurred"
                            results[i] = ('error', error_msg)
                    
                    processed_count += 1
                    
                except Exception as e:
                    log_sample(sample_id, f"Error processing command {command} on {current_api_path}: {e}")
                    with results_lock:
                        results[i] = ('error', f"Exception processing command {command}: {str(e)}")
                finally:
                    task_queue.task_done()
            
            if processed_count > 0:
                api_name = current_api_path.split('/')[-2] if current_api_path and '/' in current_api_path else str(current_api_path)
                log_sample(sample_id, f"Output generation worker {worker_id} processed {processed_count} case inputs using API {api_name}")
        
        # 创建工作线程
        threads = []
        for i in range(min(max_workers, len(commands))):
            thread = threading.Thread(
                target=worker,
                args=(f"output_gen_input_{i}",)
            )
            thread.start()
            threads.append(thread)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 整理结果
        for i in sorted(results.keys()):
            result_type, result_data = results[i]
            if result_type == 'success':
                corner_case_inputs.append(result_data)
            else:
                corner_case_inputs_error.append(result_data)
        
        return corner_case_inputs, corner_case_inputs_error
    
    def generate_test_outputs(self, case_inputs: List[str], sample: Dict, api_paths: List[str], max_workers: int = 4) -> Tuple[List[Dict], List[str]]:
        """为生成的输入生成对应的输出 - 使用全局API队列的并行版本"""
        sample_id = sample.get('id', 'unknown')
        corner_cases = []
        corner_cases_error = []
        
        # 创建任务队列
        task_queue = queue.Queue()
        for i, case_input in enumerate(case_inputs):
            task_queue.put((i, case_input))
        
        # 结果存储
        results_lock = threading.Lock()
        results = {}
        
        def worker(worker_id: str):
            """工作线程函数 - 使用全局API队列"""
            processed_count = 0
            current_api_path = None
            
            while True:
                try:
                    i, case_input = task_queue.get(timeout=1)
                except queue.Empty:
                    # 任务完成后，将使用的 API 路径归还到全局队列
                    if current_api_path:
                        return_api_path(current_api_path)
                    break
                
                # 从全局队列获取 API 路径
                if current_api_path is None:
                    current_api_path = get_api_path(timeout=0.1)
                    if current_api_path is None:
                        # 如果所有 API 都在使用中，等待一下再重试
                        task_queue.put((i, case_input))  # 将任务放回队列
                        time.sleep(0.1)
                        continue
                
                try:
                    sandbox_run_api_path = current_api_path + "run_code"
                    
                    # 优先获取 python，其次 cpp，再次 java
                    code_lang_pairs = []
                    canonical_solution = sample.get('canonical_solution', {})
                    
                    for lang in ['python', 'cpp', 'java']:
                        if lang in canonical_solution and canonical_solution[lang]:
                            code_lang_pairs.append((lang, canonical_solution[lang]))
                    
                    if not code_lang_pairs:
                        # 如果没有可用的 canonical solution，记录错误
                        error_msg = f"No canonical solution available for case {i}"
                        log_sample(sample_id, error_msg)
                        with results_lock:
                            results[i] = ('error', error_msg)
                        processed_count += 1
                        continue
                    
                    last_error = None
                    success = False
                    result_case = None
                    
                    for language, code in code_lang_pairs:
                        try:
                            payload = {
                                "code": code,
                                "language": language,
                                "stdin": case_input,
                                # "run_timeout": 40,
                            }
                            log_sample(sample_id, f"Running case {i} with {language} code ({sandbox_run_api_path})...")
                            case_response_output = self.sandbox_client.call_api(sandbox_run_api_path, payload, sample_id=sample_id)
                            
                            # 检查是否有错误字段
                            if "error" in case_response_output:
                                last_error = [language, code, case_response_output]
                                continue
                                
                            if case_response_output.get('status') == "Success":
                                result_case = {
                                    "input": {"stdin": case_input},
                                    "output": {"stdout": case_response_output.get('run_result', {}).get('stdout', '')}
                                }
                                success = True
                                break
                            else:
                                last_error = [language, code, case_response_output]
                                
                        except Exception as lang_error:
                            log_sample(sample_id, f"Error running {language} code for case {i}: {lang_error}")
                            last_error = {"error": [language, code, str(lang_error)]}
                    
                    with results_lock:
                        if success:
                            results[i] = ('success', result_case)
                        else:
                            if last_error:
                                if "error" in last_error:
                                    error_msg = f"Stdin: {case_input}; API Error: {last_error['error'][-1]}"
                                    code = last_error['error'][1]
                                    language = last_error['error'][0]
                                elif (isinstance(last_error, list) and "error" in last_error[-1]):
                                    language = last_error[0]
                                    code = last_error[1]
                                    error_msg = f"Stdin: {case_input}; API Error: {last_error[-1]['error']}"
                                else:
                                    compile_result = last_error[-1].get('compile_result', {})
                                    run_result = last_error[-1].get('run_result', {})
                                    error_msg = (
                                        f"Stdin: {case_input}; "
                                        f"Compile error (code {compile_result.get('return_code', '')}): {compile_result.get('stderr', '')}; "
                                        f"Runtime error (code {run_result.get('return_code', '')}): {run_result.get('stderr', '')}"
                                    )
                                    language = last_error[0]
                                    code = last_error[1]
                            else:
                                error_msg = f"Stdin: {case_input}; Unknown error occurred"
                            results[i] = ('error', (language, code, case_input, error_msg))

                    processed_count += 1
                    
                except Exception as e:
                    log_sample(sample_id, f"Error processing case {i} on {current_api_path}: {e}")
                    with results_lock:
                        results[i] = ('error', f"Exception processing case {i}: {str(e)}")
                finally:
                    task_queue.task_done()
            
            if processed_count > 0:
                api_name = current_api_path.split('/')[-2] if current_api_path and '/' in current_api_path else str(current_api_path)
                log_sample(sample_id, f"Output generation worker {worker_id} processed {processed_count} cases using API {api_name}")
        
        # 创建工作线程
        threads = []
        for i in range(min(max_workers, len(case_inputs))):
            thread = threading.Thread(
                target=worker,
                args=(f"output_gen_{i}",)
            )
            thread.start()
            threads.append(thread)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 整理结果
        for i in sorted(results.keys()):
            result_type, result_data = results[i]
            if result_type == 'success':
                corner_cases.append(result_data)
            else:
                corner_cases_error.append(result_data)
        
        return corner_cases, corner_cases_error
    
    def validate_corner_cases(self, sample: Dict, api_paths: List[str], dataset_type: str, max_workers: int = 4) -> Dict:
        """验证生成的 corner cases"""
        validator = SolutionValidator(self.sandbox_client)
        return validator.validate_sample(sample, api_paths, dataset_type, max_workers)

    def format_test_cases(self, test_cases: List[Dict], corner_case_inputs_dict_reverse: Dict) -> List[str]:
        """格式化测试用例结果"""
        global _global_truncated_length
        test_cases_results = []
        for test_case in test_cases:
            stdin = test_case['test_info']['input']['stdin']
            if stdin in corner_case_inputs_dict_reverse:
                stdin = corner_case_inputs_dict_reverse[stdin] + " [command]"
            stdout = test_case['test_info']['output']['stdout']
            if len(stdout) >= _global_truncated_length:
                stdout = "[output]"
            expected_output=test_case['exec_info']['run_result']['stdout']
            if len(expected_output) >= _global_truncated_length:
                expected_output = "[expected output]"
            test_case_result = TEST_CASE_RESULT_TEMPLATE.format(
                passed=test_case.get('passed', False),
                stdin=stdin,
                stdout=stdout,
                expected_output=expected_output
            )
            test_cases_results.append(test_case_result)
        return test_cases_results
    
    def generate_for_sample(self, sample: Dict, api_paths: List[str], dataset_type: str = "code_contests_test", max_workers: int = None) -> Tuple[List[Dict], List[Dict]]:
        global _global_truncated_length
        """为单个样本生成 corner cases"""
        sample_id = sample.get('id', 'unknown')

        if max_workers is None:
            max_workers = self.config_manager.processing_config.output_generation_workers if self.config_manager else 4
        
        problem_statement = sample['name'].split('. ')[-1].strip() + '\n\n' + sample['description']
        generator = sample['generator']
        prompt = PROMPT_TEMPLATE.format(problem_statement=problem_statement, generator=generator)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. You must strictly follow the user's instructions."},
            {"role": "user", "content": prompt}
        ]
        
        commands = []
        corner_cases = []
        all_results = []
        input_constraints_summary = ""
        begin = 0

        for idx in range(begin, self.max_iterations):  # 使用配置的迭代次数
            log_sample(sample_id, "Generating commands...")
            
            commands_original = self.openai_client.generate_command(messages, first=(idx==0), sample_id=sample_id)

            log_sample(sample_id, f"Generated commands: {commands_original}")
            if not commands_original:
                break
            
            log_sample(sample_id, "parsing commands...")
            commands = self.parse_commands_from_Command_Model(commands, commands_original, sample_id, first=(idx==0))
            
            if not commands:
                continue

            log_sample(sample_id, f"Successfully parsed {len(commands)} commands")
            log_sample(sample_id, f"Current commands: {commands!r}")

            # 执行 commands 生成 corner case inputs
            log_sample(sample_id, "Generating corner case inputs...")
            corner_case_inputs_map, corner_case_inputs_error = self.generate_test_inputs(commands, sample, api_paths, max_workers)
            corner_case_inputs = [item[1] for item in corner_case_inputs_map]
            success_commands = [item[0] for item in corner_case_inputs_map]
            corner_case_inputs_dict = {item[0]: item[1] for item in corner_case_inputs_map}
            corner_case_inputs_dict_reverse = {item[1]: item[0] for item in corner_case_inputs_map if len(item[1]) >= _global_truncated_length}  # 仅对较长的输入进行反向映射，避免长度过大
            # corner_case_inputs_dict_reverse = {item[1]: item[0] for item in corner_case_inputs_map}

            # 将 corner_case_inputs 中的 "\..\n" 统一为 "\n"，同时保证每个 corner_case_input 的结尾为 "\n"
            # corner_case_inputs = [re.sub(r'\\+n', '\n', case) for case in corner_case_inputs]
            # corner_case_inputs = [case if case.endswith('\n') else case + '\n' for case in corner_case_inputs]

            log_sample(sample_id, rf"Successfully parsed {len(corner_case_inputs)} corner case inputs")
            # log_sample(sample_id, rf"Current corner case inputs: {corner_case_inputs!r}")

            # 生成测试输出 - 使用并行版本
            log_sample(sample_id, "Generating corner case outputs...")
            corner_cases, corner_cases_error = self.generate_test_outputs(corner_case_inputs, sample, api_paths, max_workers)
            log_sample(sample_id, f"Successfully generated {len(corner_cases)} corner cases with outputs")
            log_sample(sample_id, f"Failed to generate outputs for {len(corner_cases_error)} corner cases")

            # 更新样本的测试用例
            sample_copy = sample.copy()
            sample_copy['test'] = corner_cases
            
            # 验证 corner cases - 使用并行版本
            validation_workers = self.config_manager.processing_config.solution_validation_workers if self.config_manager else 4
            log_sample(sample_id, "Validating corner cases...")
            result = self.validate_corner_cases(sample_copy, api_paths, dataset_type, validation_workers)
            log_sample(sample_id, f"Validated corner cases")
            if idx == 0:
                input_constraints_summary = commands_original.input_constraints_summary
                all_results.append({
                    'corner_cases': corner_cases,
                    'corner_cases_error': corner_cases_error,
                    'result': result,
                    'input_constraints_summary': commands_original.input_constraints_summary,
                    'generate_commands': commands_original.command_list,
                    'generate_case_inputs': corner_case_inputs,
                    'generate_case_inputs_error': corner_case_inputs_error,
                    'messages': messages
                })
            else:
                all_results.append({
                    'corner_cases': corner_cases,
                    'corner_cases_error': corner_cases_error,
                    'result': result,
                    'commands_add': commands_original.add_command_list,
                    'commands_replace': commands_original.replace_command_list,
                    'generate_case_inputs': corner_case_inputs,
                    'generate_case_inputs_error': corner_case_inputs_error,
                    'messages': messages
                })
            
            # 如果结果都为空，则停止迭代
            if not result['solution_result'] and not result['incorrect_solution_result'] and not corner_cases_error and not corner_case_inputs_error:
                break
            
            # 准备下一轮迭代的反馈
            log_sample(sample_id, "Preparing feedback for next iteration...")
            self._prepare_feedback_for_next_iteration(result, commands, corner_case_inputs_dict, corner_case_inputs_dict_reverse, corner_case_inputs_error, corner_cases_error, messages, commands_original, sample, first=(idx==0), input_constraints_summary=input_constraints_summary)
            log_sample(sample_id, "Feedback prepared for next iteration.")

        return corner_cases, all_results

    def _prepare_feedback_for_next_iteration(self, result: Dict, commands: List, corner_case_inputs_dict: Dict, corner_case_inputs_dict_reverse: Dict, corner_case_inputs_error: List, corner_cases_error: List, messages: List, commands_original: Union[Init_Command_Model, Command_Model], sample: Dict, first: bool, input_constraints_summary: str):
        """为下一轮迭代准备反馈"""
        # 随机采样结果（使用配置的采样数量）
        max_samples = self.max_sample_solutions
        log_sample(sample.get('id', 'unknown'), f"Sampling up to {max_samples} solutions and incorrect solutions for feedback")
        solution_result = random.sample(result['solution_result'], min(max_samples, len(result['solution_result'])))
        incorrect_solution_result = random.sample(result['incorrect_solution_result'], min(max_samples, len(result['incorrect_solution_result'])))

        log_sample(sample.get('id', 'unknown'), f"Sampled {len(solution_result)} correct solutions and {len(incorrect_solution_result)} incorrect solutions for feedback")
        
        solution_results = [
            SOLUTION_RESULT_TEMPLATE.format(
                language=res['language'], 
                solution=res['solution'], 
                output=self.format_test_cases(res['result']['tests'], corner_case_inputs_dict_reverse)
            ) for res in solution_result if isinstance(res['result'], dict) and 'tests' in res['result']
        ]
        
        incorrect_solution_results = [
            SOLUTION_RESULT_TEMPLATE.format(
                language=res['language'], 
                solution=res['solution'], 
                output=self.format_test_cases(res['result']['tests'], corner_case_inputs_dict_reverse)
            ) for res in incorrect_solution_result if isinstance(res['result'], dict) and 'tests' in res['result']
        ]
        
        # 获取最后使用的语言和代码（用于错误报告）
        if corner_cases_error:
            # language, code = "", ""
            # canonical_solution = sample.get('canonical_solution', {})
            # for lang in ['python', 'cpp', 'java']:
            #     if lang in canonical_solution and canonical_solution[lang]:
            #         language, code = lang, canonical_solution[lang]
            canonical_solution_results = []
            for error in corner_cases_error:
                stdin = error[2]
                if stdin in corner_case_inputs_dict_reverse:
                    stdin = corner_case_inputs_dict_reverse[stdin] + " [command]"
                canonical_solution_result = CANONICAL_SOLUTION_TEMPLATE.format(
                    language=error[0],
                    solution=error[1],
                    stdin=stdin,
                    output=error[3]
                )
                canonical_solution_results.append(canonical_solution_result)
        else:
            canonical_solution_results = "No errors in generating outputs for corner cases."
        
        log_sample(sample.get('id', 'unknown'), f"Prepared {len(solution_results)} correct solution results, {len(incorrect_solution_results)} incorrect solution results, and {len(canonical_solution_results) if corner_cases_error else 0} canonical solution errors for feedback")
        # 构造新的 prompt
        corner_case_inputs_dict_replace = {}
        for command, stdin in corner_case_inputs_dict.items():
            if stdin in corner_case_inputs_dict_reverse:
                stdin = "[input]"
            corner_case_inputs_dict_replace[command] = stdin
        add_prompt = ADD_PROMPT_TEMPLATE.format(
            current_command_list=commands,
            # command_to_input_map=corner_case_inputs_dict,
            command_to_input_map=corner_case_inputs_dict_replace,
            command_run_errors=corner_case_inputs_error,
            input_constraints_summary=input_constraints_summary,
            correct_results=solution_results,
            incorrect_results=incorrect_solution_results,
            outputs=canonical_solution_results
        )

        log_sample(sample.get('id', 'unknown'), "Adding new messages to conversation for next iteration")

        if first:
            assistant_response = INIT_RESPONSE_TEMPLATE.format(
                input_constraints_summary=commands_original.input_constraints_summary, 
                command_list=json.dumps(commands_original.command_list)
            )
        else:
            assistant_response = RESPONSE_TEMPLATE.format(
                replace_command_list=json.dumps(commands_original.replace_command_list), 
                add_command_list=json.dumps(commands_original.add_command_list)
            )
        messages.append({"role": "assistant", "content": assistant_response})
        messages.append({"role": "user", "content": add_prompt})
        
        log_sample(sample.get('id', 'unknown'), "Messages updated for next iteration")

        if not self.use_all_solutions:
            # 更新样本的 solutions 和 incorrect_solutions
            solutions = {
                "language": [LANGUAGE.index(res['language']) for res in solution_result],
                "solution": [res['solution'] for res in solution_result]
            }
            
            incorrect_solutions = {
                "language": [LANGUAGE.index(res['language']) for res in incorrect_solution_result],
                "solution": [res['solution'] for res in incorrect_solution_result]
            }
            
            sample['solutions'] = solutions
            sample['incorrect_solutions'] = incorrect_solutions


class SolutionValidator:
    """解决方案验证器"""
    
    def __init__(self, sandbox_client: SandboxClient):
        self.sandbox_client = sandbox_client
    
    def validate_sample(self, sample: Dict, api_paths: List[str], dataset_type: str, max_workers: int = 4) -> Dict:
        """验证单个样本的解决方案"""
        id = sample['id']
        config = {
            'language': None,
            'locale': "en",
            'compile_timeout': 20,
            'run_timeout': 20,
            'dataset_type': "CommonOJDataset",
            'extra': {'run_all_cases': True},
        }
        
        key_list = ['id', 'content', 'test', 'labels', 'checker', 'canonical_solution']
        provided_data = {key: sample[key] for key in key_list if key in sample}
        config['provided_data'] = provided_data
        
        solutions = sample.get('solutions', {'language': [], 'solution': []})
        incorrect_solutions = sample.get('incorrect_solutions', {'language': [], 'solution': []})

        log_sample(sample['id'], "Validating solutions...")
        res = self._validate_solutions(config, solutions, api_paths, id, dataset_type, flag=True, max_workers=max_workers)
        incorrect_res = self._validate_solutions(config, incorrect_solutions, api_paths, id, dataset_type, flag=False, max_workers=max_workers)
        log_sample(sample['id'], "Solutions validated.")

        return {
            'id': id,
            'solution_result': res,
            'incorrect_solution_result': incorrect_res,
        }
    
    def _validate_solutions(self, config: Dict, solutions: Dict, api_paths: List[str], id: str, dataset_type: str, flag: bool = False, max_workers: int = 4) -> List[Dict]:
        """验证解决方案列表 - 使用全局API队列的并行版本"""
        sample_id = id  # 使用样本ID进行日志记录
        if not solutions['language'] or not solutions['solution']:
            return []
        
        # 创建任务队列
        task_queue = queue.Queue()
        for idx, (language_index, solution) in enumerate(zip(solutions['language'], solutions['solution'])):
            language = LANGUAGE[language_index]
            if language != "UNKNOWN_LANGUAGE":
                task_queue.put((idx, language_index, language, solution))
        
        if task_queue.empty():
            return []
        
        # 结果存储
        results_lock = threading.Lock()
        results = []
        
        def worker(worker_id: str):
            """工作线程函数 - 使用全局API队列"""
            processed_count = 0
            current_api_path = None
            
            while True:
                try:
                    idx, language_index, language, solution = task_queue.get(timeout=1)
                except queue.Empty:
                    # 任务完成后，将使用的 API 路径归还到全局队列
                    if current_api_path:
                        return_api_path(current_api_path)
                    break
                
                # 从全局队列获取 API 路径
                if current_api_path is None:
                    current_api_path = get_api_path(timeout=0.1)
                    if current_api_path is None:
                        # 如果所有 API 都在使用中，等待一下再重试
                        task_queue.put((idx, language_index, language, solution))  # 将任务放回队列
                        time.sleep(0.1)
                        continue
                
                try:
                    if "PYTHON" in language:
                        language = "PYTHON"
                    
                    config_copy = config.copy()
                    config_copy['language'] = language.lower()
                    completion = TEMPLATE.format(language=language.lower(), solution=solution)
                    
                    payload = {
                        'dataset': dataset_type,
                        'id': id,
                        'completion': completion,
                        'config': config_copy
                    }
                    log_sample(sample_id, f"Submitting {idx} ({language}) code to {current_api_path}...")
                    resp = self.sandbox_client.call_api(current_api_path + "submit", payload, sample_id=sample_id)
                    log_sample(sample_id, f"Received response from {current_api_path}: {idx} ({language})")

                    # 检查是否有API错误
                    if "error" in resp:
                        log_sample(sample_id, f"API {current_api_path} error for solution validation {idx} ({language}): {resp['error']}")
                        processed_count += 1
                        continue
                    
                    # 处理 checker 验证逻辑
                    if not resp.get('accepted', False):
                        log_sample(sample_id, f"Running checker validation for {idx} ({language}) {current_api_path}...")
                        resp = self._run_checker_validation(resp, config_copy, current_api_path, sample_id)
                        log_sample(sample_id, f"Checker validation completed for {idx} ({language}) {current_api_path}")

                    # 根据 flag 决定是否保留结果
                    if flag != resp.get('accepted', False):
                        result = {
                            'language': language,
                            'solution': solution,
                            'result': resp
                        }
                        
                        with results_lock:
                            results.append(result)
                    
                    processed_count += 1
                    
                except Exception as e:
                    log_sample(sample_id, f"Error validating solution on {current_api_path}: {e}")
                    if 'resp' in locals():
                        log_sample(sample_id, f"Response: {resp}")
                    else:
                        log_sample(sample_id, "No response received")
                finally:
                    task_queue.task_done()
            
            if processed_count > 0:
                api_name = current_api_path.split('/')[-2] if current_api_path and '/' in current_api_path else str(current_api_path)
                log_sample(sample_id, f"Solution validation worker {worker_id} processed {processed_count} solutions using API {api_name}")
        
        # 创建工作线程
        threads = []
        for i in range(min(max_workers, task_queue.qsize())):
            thread = threading.Thread(
                target=worker,
                args=(f"solution_val_{i}",)
            )
            thread.start()
            threads.append(thread)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        return results
    
    def _run_checker_validation(self, resp: Dict, config: Dict, api_path: str, sample_id: str = None) -> Dict:
        """运行 checker 验证"""
        checker = config.get('provided_data', {}).get('checker', None)
        if checker is None:
            return resp
        
        accepted = True
        for test_case in resp.get('tests', []):
            if not test_case.get('passed', False):
                stdout = test_case['exec_info']['run_result'].get('stdout', '')
                expected_output = test_case['test_info']['output'].get('stdout', '')
                stdin = test_case['test_info']['input'].get('stdin', '')
                
                try:
                    # 编码为 base64
                    stdin_b64 = base64.b64encode(stdin.encode('utf-8')).decode('utf-8')
                    stdout_b64 = base64.b64encode(stdout.encode('utf-8')).decode('utf-8')
                    expected_output_b64 = base64.b64encode(expected_output.encode('utf-8')).decode('utf-8')
                    
                    with open("testlib.h", 'rb') as file:
                        testlib_data_b64 = base64.b64encode(file.read()).decode('utf-8')
                except Exception as e:
                    if sample_id:
                        log_sample(sample_id, f"Error encoding base64 for test case {test_case}: {e}")
                    else:
                        log_global(f"Error encoding base64 for test case {test_case}: {e}")
                    continue
                
                files = {
                    "input.txt": stdin_b64,
                    "output.txt": stdout_b64,
                    "answer.txt": expected_output_b64,
                    "testlib.h": testlib_data_b64
                }
                
                payload = {
                    "code": checker,
                    "language": "cpp",
                    "extra_args": "input.txt output.txt answer.txt",
                    "files": files,
                }
                
                try:
                    response = self.sandbox_client.call_api(api_path + "run_code", payload, sample_id=sample_id)
                    test_case['checker_info'] = response
                    
                    # 检查是否有API错误
                    if "error" in response:
                        if sample_id:
                            log_sample(sample_id, f"API error in checker validation: {response['error']}")
                        else:
                            log_global(f"API error in checker validation: {response['error']}")
                        accepted = False
                        break
                    
                    if response.get('status') == "Success":
                        if "ok" not in response['run_result'].get("stderr", ""):
                            accepted = False
                            break
                    else:
                        accepted = False
                        break
                except Exception as e:
                    if sample_id:
                        log_sample(sample_id, f"Error calling sandbox API for test case {test_case}: {e}")
                    else:
                        log_global(f"Error calling sandbox API for test case {test_case}: {e}")
                    accepted = False
                    break
                    
        
        resp['accepted'] = accepted
        return resp


class ParallelProcessor:
    """并行处理器"""

    def __init__(self, api_paths: List[str], specific_api_paths: List[str], max_workers: int = 1, config_manager=None):
        self.api_paths = api_paths
        self.specific_api_paths = specific_api_paths
        self.max_workers = max_workers
        self.config_manager = config_manager
        
        # 初始化全局 API 队列
        initialize_global_api_queue(api_paths)
        initialize_global_specific_api_queue(specific_api_paths)
        
        # 初始化客户端
        openai_config = config_manager.openai_config if config_manager else None
        if openai_config:
            self.openai_client = OpenAIClient(openai_config.api_base, openai_config.api_key, openai_config.model, openai_config.max_tokens, openai_config.no_reasoning)
        else:
            self.openai_client = OpenAIClient(API_BASE, API_KEY)
        
        self.sandbox_client = SandboxClient()
        self.corner_case_generator = CornerCaseGenerator(
            self.openai_client, 
            self.sandbox_client, 
            config_manager
        )
        
        # 创建 session 池
        self.sandbox_client.session.mount('http://', requests.adapters.HTTPAdapter(
            pool_maxsize=len(api_paths) * max_workers
        ))
    
    def process_dataset(self, dataset: List[Dict], dataset_type: str = "code_contests_test", 
                       results_dir: Optional[str] = None, debug: bool = False) -> None:
        """并行处理数据集 - 现在每个 sample 内部的操作也是并行的"""
        # 确保日志系统已经初始化
        if get_global_logger_manager() is None and results_dir:
            initialize_logger_manager(results_dir)
        
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            
            # 获取已处理的样本 ID
            existing_ids = set()
            for fname in os.listdir(results_dir):
                if fname.endswith('.json'):
                    existing_ids.add(os.path.splitext(fname)[0])
            
            # 筛选出需要处理的新样本
            samples_to_process = [
                sample for sample in dataset 
                if sample['id'].replace('/', '_') not in existing_ids
            ]
        else:
            samples_to_process = dataset
        
        if not samples_to_process:
            log_global("All samples have already been processed. No new tasks to run.")
            return
        
        log_global(f"Found {len(samples_to_process)} new samples to process out of {len(dataset)} total.")
        log_global(f"Using {len(self.api_paths)} API endpoints")
        log_global(f"Parallel processing will be applied at both sample level and operation level")
        
        # 线程同步
        results_lock = threading.Lock()
        global completed_tasks
        completed_tasks = 0
        
        # 使用 ThreadPoolExecutor 进行 sample 级别的并行
        sample_workers = self.config_manager.processing_config.sample_level_workers if self.config_manager else 2
        max_sample_workers = min(sample_workers, len(samples_to_process), len(self.api_paths))
        
        def process_single_sample(sample: Dict) -> Dict:
            """处理单个样本"""
            sample_id = sample.get('id', 'unknown')
            result_data = {
                'id': sample_id,
                'status': 'processing',
                'error': None,
                'corner_cases': [],
                'result': []
            }
            
            try:
                sample['filter_solutions'] = sample['solutions']
                sample['filter_incorrect_solutions'] = sample['incorrect_solutions']
                
                # 每个样本使用全部 API 端点进行内部并行处理
                output_workers = self.config_manager.processing_config.output_generation_workers if self.config_manager else len(self.api_paths)
                log_sample(sample['id'], "Generating corner cases...")
                corner_cases, all_results = self.corner_case_generator.generate_for_sample(
                    sample, self.api_paths, dataset_type, max_workers=output_workers
                )
                log_sample(sample['id'], "Corner cases generated.")

                result_data['corner_cases'] = corner_cases
                result_data['result'] = all_results
                result_data['status'] = 'completed'
                result_data.update(sample)  # 包含原始样本数据
                
            except Exception as e:
                error_msg = f"Error processing sample {sample_id}: {str(e)}"
                log_sample(sample_id, error_msg)
                result_data['status'] = 'error'
                result_data['error'] = error_msg
                result_data.update(sample)  # 仍然包含原始样本数据
            
            finally:
                # 确保每个样本都有结果文件保存
                if results_dir:
                    try:
                        with results_lock:
                            result_error_folder = os.path.join(results_dir, "error")
                            os.makedirs(result_error_folder, exist_ok=True)
                            sample_name = sample_id.replace('/', '_') + '.json' if result_data['status'] == 'completed' else os.path.join("error", sample_id.replace('/', '_') + '.json')
                            if result_data['status'] == 'completed' and not result_data['corner_cases']:
                                log_sample(sample_id, f"Warning: Sample {sample_id} completed but no corner cases generated.")
                                sample_name = os.path.join("empty", sample_id.replace('/', '_') + '.json')
                                result_empty_folder = os.path.join(results_dir, "empty")
                                os.makedirs(result_empty_folder, exist_ok=True)
                            result_path = os.path.join(results_dir, sample_name)
                            with open(result_path, 'w') as f:
                                json.dump(result_data, f, indent=4)
                    except Exception as save_error: 
                        log_sample(sample_id, f"Error saving result for sample {sample_id}: {save_error}")
                        # 尝试保存到错误文件
                        try:
                            error_sample_name = sample_id.replace('/', '_') + '_error.json'
                            result_empty_folder = os.path.join(results_dir, "error")
                            os.makedirs(result_empty_folder, exist_ok=True)
                            error_result_path = os.path.join(result_empty_folder, error_sample_name)
                            error_data = {
                                'id': sample_id,
                                'status': 'save_error',
                                'error': str(save_error),
                                'original_data': sample
                            }
                            with open(error_result_path, 'w') as f:
                                json.dump(error_data, f, indent=4)
                        except Exception as final_error:
                            log_sample(sample_id, f"Failed to save even error file for sample {sample_id}: {final_error}")
                
                global completed_tasks
                with results_lock:
                    completed_tasks += 1
            
            return result_data
        
        # 使用线程池处理所有样本
        with ThreadPoolExecutor(max_workers=max_sample_workers) as executor:
            # 提交所有任务
            future_to_sample = {
                executor.submit(process_single_sample, sample): sample
                for sample in samples_to_process
            }
            
            # 使用进度条跟踪进度
            with tqdm(total=len(samples_to_process), desc="Processing samples") as pbar:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    sample_id = sample.get('id', 'unknown')
                    
                    try:
                        result = future.result()
                        if result is not None:
                            if debug:
                                status = result.get('status', 'unknown')
                                log_global(f"Sample {sample_id} processed with status: {status}")
                                if status == 'error':
                                    log_global(f"  Error: {result.get('error', 'Unknown error')}")
                        else:
                            log_global(f"Sample {sample_id} returned None result")
                            
                    except Exception as e:
                        log_global(f"Future execution failed for sample {sample_id}: {e}")
                        
                        # 尝试保存错误信息到文件
                        if results_dir:
                            try:
                                error_sample_name = sample_id.replace('/', '_') + '_future_error.json'
                                error_result_path = os.path.join(results_dir, error_sample_name)
                                error_data = {
                                    'id': sample_id,
                                    'status': 'future_error',
                                    'error': str(e),
                                    'original_data': sample
                                }
                                with open(error_result_path, 'w') as f:
                                    json.dump(error_data, f, indent=4)
                                log_global(f"Saved future error data for sample {sample_id}")
                            except Exception as save_error:
                                log_global(f"Failed to save future error data for sample {sample_id}: {save_error}")
                                
                    finally:
                        pbar.update(1)
        
        log_global("All tasks completed.")


# %% 主函数
def main():
    """主函数"""
    # 导入配置
    from config import ConfigManager, PRODUCTION_CONFIG
    
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 验证配置
    if not config_manager.validate_config():
        print("Configuration validation failed. Exiting.")
        return
    
    # 初始化日志系统
    initialize_logger_manager(config_manager.dataset_config.results_dir)
    
    # 获取运行时信息
    runtime_info = config_manager.get_runtime_info()
    log_global("=== Runtime Configuration ===")
    for key, value in runtime_info.items():
        if key != "api_paths":  # API paths 太长，不打印
            log_global(f"{key}: {value}")
    log_global("=" * 30)
    
    # 读取数据集
    dataset_processor = DatasetProcessor()
    dataset = dataset_processor.read_dataset(
        config_manager.dataset_config.data_path, 
        config_manager.dataset_config.split
    )
    
    if not dataset:
        log_global("Failed to load dataset")
        return
    
    log_global(f"Loaded {len(dataset)} samples from dataset")
    
    # 创建并行处理器
    api_paths = config_manager.sandbox_config.get_api_paths()
    # 创建特定API路径（用于TimeLimitExceeded重试）
    specific_api_paths = config_manager.sandbox_config.get_specific_api_paths() if hasattr(config_manager.sandbox_config, 'get_specific_api_paths') else api_paths[:min(10, len(api_paths))]
    processor = ParallelProcessor(
        api_paths, 
        specific_api_paths,
        max_workers=config_manager.processing_config.max_workers_per_api,
        config_manager=config_manager
    )
    
    # 开始处理
    start_time = time.time()
    processor.process_dataset(
        dataset, 
        config_manager.dataset_config.dataset_type, 
        config_manager.dataset_config.results_dir, 
        debug=config_manager.processing_config.debug
    )
    end_time = time.time()
    
    # 输出统计信息和保存结果
    _save_processing_stats(start_time, end_time, len(dataset), api_paths, runtime_info, config_manager)
    
    # 清理日志系统
    logger_manager = get_global_logger_manager()
    if logger_manager:
        logger_manager.cleanup()


def main_with_custom_config(config_type: str = "production"):
    """使用自定义配置运行主函数"""
    from config import ConfigManager, PRODUCTION_CONFIG, DEVELOPMENT_CONFIG
    
    config_manager = ConfigManager()
    
    # 根据配置类型更新配置
    if config_type == "production":
        config_dict = PRODUCTION_CONFIG
    elif config_type == "development":
        config_dict = DEVELOPMENT_CONFIG
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    # 更新配置（这里简化处理，实际可以做更复杂的配置合并）
    if "sandbox_config" in config_dict:
        for key, value in config_dict["sandbox_config"].items():
            setattr(config_manager.sandbox_config, key, value)
    
    if "processing_config" in config_dict:
        for key, value in config_dict["processing_config"].items():
            setattr(config_manager.processing_config, key, value)

    if "dataset_config" in config_dict:
        for key, value in config_dict["dataset_config"].items():
            setattr(config_manager.dataset_config, key, value)

    if "openai_config" in config_dict:
        for key, value in config_dict["openai_config"].items():
            setattr(config_manager.openai_config, key, value)

    # 使用更新后的配置运行
    main_with_config_manager(config_manager)


def main_with_config_manager(config_manager):
    """使用指定的配置管理器运行主函数"""
    # 验证配置
    if not config_manager.validate_config():
        print("Configuration validation failed. Exiting.")
        return
    
    # 初始化日志系统
    initialize_logger_manager(config_manager.dataset_config.results_dir)
    
    # 获取运行时信息
    runtime_info = config_manager.get_runtime_info()
    log_global("=== Runtime Configuration ===")
    for key, value in runtime_info.items():
        if key != "api_paths":
            log_global(f"{key}: {value}")
    log_global("=" * 30)
    
    # 读取数据集
    dataset_processor = DatasetProcessor()
    dataset = dataset_processor.read_dataset(
        config_manager.dataset_config.data_path, 
        config_manager.dataset_config.split
    )
    
    if not dataset:
        log_global("Failed to load dataset")
        return
    
    log_global(f"Loaded {len(dataset)} samples from dataset")
    
    # 创建并行处理器
    api_paths = config_manager.sandbox_config.get_api_paths()
    specific_api_paths = config_manager.sandbox_config.get_specific_api_paths()
    processor = ParallelProcessor(
        api_paths, 
        specific_api_paths,
        max_workers=config_manager.processing_config.max_workers_per_api,
        config_manager=config_manager
    )
    
    # 开始处理
    start_time = time.time()
    processor.process_dataset(
        dataset, 
        config_manager.dataset_config.dataset_type, 
        config_manager.dataset_config.results_dir, 
        debug=config_manager.processing_config.debug
    )
    end_time = time.time()
    
    # 输出统计信息和保存结果
    _save_processing_stats(start_time, end_time, len(dataset), api_paths, runtime_info, config_manager)
    
    # 清理日志系统
    logger_manager = get_global_logger_manager()
    if logger_manager:
        logger_manager.cleanup()


def _save_processing_stats(start_time: float, end_time: float, dataset_size: int, 
                          api_paths: List[str], runtime_info: Dict, config_manager):
    """保存处理统计信息"""
    total_time = end_time - start_time
    log_global(f"Total processing time: {total_time:.2f} seconds")
    if dataset_size > 0:
        log_global(f"Average time per sample: {total_time/dataset_size:.2f} seconds")
    
    # 保存详细统计
    time_file = os.path.join(config_manager.dataset_config.results_dir, "processing_stats.txt")
    with open(time_file, "w") as f:
        f.write("=== Processing Statistics ===\n")
        f.write(f"Total processing time: {total_time:.2f} seconds\n")
        f.write(f"Sample number: {dataset_size}\n")
        if dataset_size > 0:
            f.write(f"Average time per sample: {total_time/dataset_size:.2f} seconds\n")
        f.write(f"API endpoints used: {len(api_paths)}\n")
        f.write(f"Total workers: {runtime_info['total_workers']}\n")
        f.write(f"Max iterations per sample: {config_manager.processing_config.max_iterations}\n")
        f.write("\n=== Configuration ===\n")
        f.write(f"Dataset path: {config_manager.dataset_config.data_path}\n")
        f.write(f"Results directory: {config_manager.dataset_config.results_dir}\n")
        f.write(f"Debug mode: {config_manager.processing_config.debug}\n")
        f.write("\n=== Parallel Processing Details ===\n")
        f.write("Parallel processing is applied at multiple levels:\n")
        f.write("1. Sample level: Multiple samples processed concurrently\n")
        f.write("2. Output generation: Multiple test case outputs generated in parallel\n")
        f.write("3. Solution validation: Multiple solutions validated in parallel\n")


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数选择配置类型
    if len(sys.argv) > 1:
        config_type = sys.argv[1]
        if config_type in ["production", "development"]:
            main_with_custom_config(config_type)
        else:
            log_global(f"Unknown config type: {config_type}")
            log_global("Available types: production, development")
    else:
        main()
