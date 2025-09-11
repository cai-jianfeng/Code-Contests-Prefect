# TimeLimitExceeded 处理改进总结

## 改进概览

根据 `solutions_eval_plus_test.py` 的实现，我们对 `corner_case_gen_parallel.py` 进行了以下改进来处理 TimeLimitExceeded 错误：

## 1. 新增功能函数

### 1.1 特定API队列管理
```python
def get_specific_api_path(timeout: float = 0.1) -> Optional[str]
def return_specific_api_path(api_path: str)
```
- 管理特定API路径队列，用于TimeLimitExceeded重试

### 1.2 超时检测
```python
def is_time_limit_exceeded(response) -> Union[False, str]
```
- 检测响应中的TimeLimitExceeded状态
- 返回值：
  - `False`: 没有TimeLimitExceeded
  - `"sandbox_blocked"`: Sandbox内部阻塞，无实际运行结果
  - `"real_timeout"`: 代码真实运行超时，有部分输出结果

### 1.3 超时倍数计算
```python
def calculate_timeout_multiplier(actual_stdout, expected_stdout) -> Tuple[bool, float]
```
- 基于输出完整性计算超时时间倍数
- 用于智能调整重试时的超时时间

## 2. SandboxClient 改进

### 2.1 智能重试机制
- 原始的 `call_api` 方法只调用一次API
- 新版本支持最多3次重试，包括：
  1. **real_timeout 处理**: 根据输出完整性调整超时时间后重试
  2. **sandbox_blocked 处理**: 使用特定API队列重试
  3. **异常处理**: 网络错误等异常的重试

### 2.2 特定API重试
```python
def _retry_with_specific_api(self, json_data: Dict, max_retries: int = 3) -> Dict
```
- 当常规API遇到TimeLimitExceeded时，使用特定API队列重试
- 每次重试使用不同的特定API路径，最大化避免sandbox错误

## 3. 错误处理改进

### 3.1 generate_test_outputs 方法
- 添加了对canonical solution不存在的检查
- 改进了API错误的处理
- 确保每个测试用例都有明确的成功/失败状态

### 3.2 _validate_solutions 方法
- 添加了对API返回错误的检查
- 改进了异常处理和日志输出
- 避免未定义变量的访问

### 3.3 process_dataset 方法
- **确保每个sample都有结果文件保存**，即使出错也会保存错误信息
- 多层错误处理：
  1. 处理过程中的异常 → 保存错误状态的结果文件
  2. 文件保存失败 → 保存到 `*_error.json` 文件
  3. Future执行失败 → 保存到 `*_future_error.json` 文件

## 4. 主要改进点对比

| 方面 | 原始版本 | 改进版本 |
|------|----------|----------|
| API调用 | 单次调用，失败即返回错误 | 智能重试，区分不同错误类型 |
| 超时处理 | 无特殊处理 | 分析输出完整性，动态调整超时时间 |
| API选择 | 固定使用分配的API | 遇到问题时切换到特定API队列 |
| 错误恢复 | 样本失败可能无结果文件 | 确保每个样本都有结果文件 |
| 错误信息 | 简单错误日志 | 详细的错误分类和保存 |

## 5. 使用方式

### 5.1 配置要求
需要在配置中提供两种API路径：
- `api_paths`: 常规API路径
- `specific_api_paths`: 特定API路径（用于重试）

### 5.2 调用示例
```python
# 创建处理器时需要提供两种API路径
processor = ParallelProcessor(
    api_paths=regular_api_paths,
    specific_api_paths=specific_api_paths,
    max_workers=max_workers,
    config_manager=config_manager
)
```

## 6. 测试验证

运行 `verify_timeout_handling.py` 可以验证核心功能：
```bash
python3 verify_timeout_handling.py
```

预期输出：
```
✓ Sandbox blocked detection: sandbox_blocked
✓ Real timeout detection: real_timeout  
✓ Timeout multiplier: partial=True, multiplier=2.40
✓ SandboxClient created: <class 'corner_case_gen_parallel.SandboxClient'>
All function tests passed!
```

## 7. 注意事项

1. **特定API队列**: 确保有足够的特定API用于重试
2. **超时限制**: 设置合理的最大超时时间（MAX_TIME = 1000秒）
3. **重试次数**: 每种错误类型最多重试3次
4. **文件保存**: 所有样本都会有结果文件，便于后续分析和恢复
