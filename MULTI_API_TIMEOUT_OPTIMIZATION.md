# SandboxClient 多API类型TimeLimitExceeded处理优化

## 优化背景

之前的 `call_api` 方法只针对 `/submit` API 进行了TimeLimitExceeded优化，但实际上它还服务于其他两种不同的使用场景：

1. **`_validate_solutions`** - 使用 `/submit` API 进行完整解决方案验证
2. **`_run_checker_validation`** - 使用 `/run_code` API 运行checker程序
3. **`generate_test_outputs`** - 使用 `/run_code` API 运行代码获取输出

每种API的输入格式、响应结构和超时处理策略都不同，需要分别优化。

## 优化内容

### 1. API类型自动识别

```python
def call_api(self, api_path: str, json_data: Dict, max_retries: int = 3, retry_delay: float = 1) -> Dict:
    # 识别API类型
    is_submit_api = api_path.endswith('/submit')
    is_run_code_api = api_path.endswith('/run_code')
```

根据API路径自动识别调用类型，采用不同的处理策略。

### 2. 不同API的超时检测

#### `/submit` API (解决方案验证)
- **响应结构**: `{"tests": [{"exec_info": {"run_result": {...}}}]}`
- **检测方式**: 遍历 `tests` 数组检查每个测试的执行状态
- **优势**: 有完整的测试用例信息，可以进行精确的输出比较

#### `/run_code` API (代码执行)
- **响应结构**: `{"run_result": {"status": "TimeLimitExceeded", "stdout": "..."}}`
- **检测方式**: 直接检查顶层的 `run_result` 状态
- **特点**: 没有期望输出进行比较，主要看是否有实际输出

```python
def _detect_timeout_by_api_type(self, response: Dict, is_submit_api: bool, is_run_code_api: bool):
    if is_submit_api:
        return is_time_limit_exceeded(response)  # 使用原有逻辑
    elif is_run_code_api:
        # 检查顶层执行结果
        if response.get('status') != 'Success':
            run_result = response.get('run_result', {})
            if run_result.get('status') == 'TimeLimitExceeded':
                return "real_timeout" if run_result.get('stdout') else "sandbox_blocked"
    return False
```

### 3. 智能超时分析和调整

#### `/submit` API 
- **策略**: 比较实际输出和期望输出，计算输出完整性
- **调整方式**: 根据输出比例动态调整超时时间
- **配置**: 直接修改 `config` 中的 `run_timeout` 和 `compile_timeout`

#### `/run_code` API
- **策略**: 检查是否有输出，有输出说明程序在运行
- **调整方式**: 暂时无法调整超时参数，直接使用特定API重试
- **原因**: `/run_code` API 没有 config 结构来传递超时参数

```python
def _analyze_timeout_and_adjust(self, response, json_data, original_run_timeout, original_compile_timeout,
                               is_submit_api, is_run_code_api, MAX_TIME):
    if is_submit_api:
        # 详细分析测试用例输出，计算超时倍数
        # 返回新的超时配置
    elif is_run_code_api:
        # 简单检查是否有输出
        # 有输出就重试，无法调整超时参数
```

### 4. 重试策略优化

| API类型 | real_timeout处理 | sandbox_blocked处理 |
|---------|------------------|---------------------|
| `/submit` | 调整超时时间重试 → 特定API | 特定API重试 |
| `/run_code` | 直接特定API重试 | 特定API重试 |

## 使用场景对比

### 场景1: generate_test_outputs
```python
payload = {
    "code": code,
    "language": language,  
    "stdin": case_input,
}
response = self.sandbox_client.call_api(api_path + "run_code", payload)
```
- **目标**: 运行canonical solution获取corner case的预期输出
- **优化**: 检测 `/run_code` API 的超时，有输出时使用特定API重试

### 场景2: _validate_solutions
```python
payload = {
    'dataset': dataset_type,
    'id': id,
    'completion': completion,
    'config': config_copy
}
response = self.sandbox_client.call_api(api_path + "submit", payload)
```
- **目标**: 验证solution/incorrect_solution的正确性
- **优化**: 分析测试用例输出完整性，动态调整超时时间

### 场景3: _run_checker_validation
```python
payload = {
    "code": checker,
    "language": "cpp",
    "extra_args": "input.txt output.txt answer.txt",
    "files": files,
}
response = self.sandbox_client.call_api(api_path + "run_code", payload)
```
- **目标**: 运行checker程序验证输出正确性
- **优化**: 检测checker程序的超时，使用特定API重试

## 测试验证

运行 `test_enhanced_sandbox_client.py` 验证：

```bash
$ python3 test_enhanced_sandbox_client.py
Testing enhanced SandboxClient...
✓ Submit API timeout detection: False
✓ Run code API timeout detection: real_timeout  
✓ Run code API blocked detection: sandbox_blocked
✓ Submit API timeout analysis: should_retry=True, config={'run_timeout': 38, 'compile_timeout': 38}
✓ Run code API timeout analysis: should_retry=True, config=None
🎉 All enhanced SandboxClient tests passed!
```

## 关键改进点

1. **API类型自适应**: 根据API路径自动选择合适的处理策略
2. **响应格式适配**: 针对不同API的响应结构采用不同的解析方式
3. **超时策略优化**: `/submit` API支持动态超时调整，`/run_code` API使用特定API重试
4. **错误处理增强**: 每种API类型都有对应的错误检测和处理逻辑
5. **向后兼容**: 保持原有接口不变，内部自动适配不同场景

这样的优化确保了 `call_api` 方法能够智能地处理所有使用场景的TimeLimitExceeded问题，提高了整个系统的稳定性和成功率。
