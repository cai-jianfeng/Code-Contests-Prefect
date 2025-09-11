#!/usr/bin/env python3
"""
演示负载均衡改进效果的脚本
展示新的队列机制如何平衡 API 端点的使用
"""

import time
import threading
import queue
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import random

def demonstrate_old_approach():
    """演示旧方法的 API 使用分布"""
    print("=== 旧方法: API 端点使用分布 ===")
    
    api_paths = ["/api1", "/api2", "/api3", "/api4"]
    tasks = list(range(20))  # 20个任务
    max_workers = 8
    api_usage = Counter()
    
    # 模拟旧方法：每个API分配固定数量的worker
    workers_per_api = max_workers // len(api_paths)
    workers_per_api = max(1, workers_per_api)
    
    def old_worker(api_path: str, worker_id: str, task_subset: List[int]):
        """旧方法的工作函数"""
        for task_id in task_subset:
            # 模拟API调用
            time.sleep(0.01)
            api_usage[api_path] += 1
    
    # 将任务分配给各个API的worker
    threads = []
    tasks_per_worker = len(tasks) // (len(api_paths) * workers_per_api)
    task_index = 0
    
    for api_path in api_paths:
        for i in range(workers_per_api):
            worker_tasks = tasks[task_index:task_index + tasks_per_worker]
            if worker_tasks:
                thread = threading.Thread(
                    target=old_worker,
                    args=(api_path, f"{api_path}_{i}", worker_tasks)
                )
                thread.start()
                threads.append(thread)
                task_index += tasks_per_worker
    
    # 处理剩余任务
    if task_index < len(tasks):
        remaining_tasks = tasks[task_index:]
        thread = threading.Thread(
            target=old_worker,
            args=(api_paths[0], "extra", remaining_tasks)
        )
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    print("API 使用统计:")
    for api_path, count in api_usage.items():
        print(f"  {api_path}: {count} 次 ({count/sum(api_usage.values())*100:.1f}%)")
    
    variance = sum((count - sum(api_usage.values())/len(api_usage))**2 for count in api_usage.values()) / len(api_usage)
    print(f"使用分布方差: {variance:.2f} (方差越大越不均衡)")
    print()


def demonstrate_new_approach():
    """演示新方法的 API 使用分布"""
    print("=== 新方法: 队列负载均衡的 API 使用分布 ===")
    
    api_paths = ["/api1", "/api2", "/api3", "/api4"]
    max_workers = 8
    api_usage = Counter()
    usage_lock = threading.Lock()
    
    # 创建任务队列
    task_queue = queue.Queue()
    for i in range(20):
        task_queue.put(i)
    
    # 创建API队列
    api_queue = queue.Queue()
    for api_path in api_paths:
        api_queue.put(api_path)
    
    def new_worker(worker_id: str):
        """新方法的工作函数"""
        processed_count = 0
        current_api_path = None
        
        while True:
            try:
                task_id = task_queue.get(timeout=0.1)
            except queue.Empty:
                # 任务完成后，将使用的API路径放回队列
                if current_api_path:
                    api_queue.put(current_api_path)
                break
            
            # 获取API路径
            if current_api_path is None:
                try:
                    current_api_path = api_queue.get(timeout=0.05)
                except queue.Empty:
                    # 如果所有API都在使用中，等待一下再重试
                    task_queue.put(task_id)
                    time.sleep(0.01)
                    continue
            
            try:
                # 模拟API调用
                time.sleep(0.01)
                
                with usage_lock:
                    api_usage[current_api_path] += 1
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error in worker {worker_id}: {e}")
            finally:
                task_queue.task_done()
        
        if processed_count > 0:
            api_name = current_api_path.split('/')[-1] if current_api_path else "unknown"
            print(f"Worker {worker_id} processed {processed_count} tasks using {api_name}")
    
    # 创建工作线程
    threads = []
    for i in range(max_workers):
        thread = threading.Thread(target=new_worker, args=(f"worker_{i}",))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    print("\nAPI 使用统计:")
    if api_usage:
        for api_path, count in api_usage.items():
            print(f"  {api_path}: {count} 次 ({count/sum(api_usage.values())*100:.1f}%)")
        
        variance = sum((count - sum(api_usage.values())/len(api_usage))**2 for count in api_usage.values()) / len(api_usage)
        print(f"使用分布方差: {variance:.2f} (方差越小越均衡)")
    else:
        print("  没有API调用记录")
    print()


def demonstrate_with_different_task_sizes():
    """演示不同任务数量下的负载均衡效果"""
    print("=== 不同任务数量的负载均衡对比 ===")
    
    api_paths = ["/api1", "/api2", "/api3", "/api4"]
    task_sizes = [10, 50, 100]
    
    for task_size in task_sizes:
        print(f"\n任务数量: {task_size}")
        
        # 新方法测试
        api_usage = Counter()
        usage_lock = threading.Lock()
        
        task_queue = queue.Queue()
        for i in range(task_size):
            task_queue.put(i)
        
        api_queue = queue.Queue()
        for api_path in api_paths:
            api_queue.put(api_path)
        
        def balanced_worker(worker_id: str):
            current_api_path = None
            while True:
                try:
                    task_id = task_queue.get(timeout=0.01)
                except queue.Empty:
                    if current_api_path:
                        api_queue.put(current_api_path)
                    break
                
                if current_api_path is None:
                    try:
                        current_api_path = api_queue.get(timeout=0.01)
                    except queue.Empty:
                        task_queue.put(task_id)
                        continue
                
                # 模拟处理
                time.sleep(0.001)
                with usage_lock:
                    api_usage[current_api_path] += 1
                task_queue.task_done()
        
        threads = []
        for i in range(8):
            thread = threading.Thread(target=balanced_worker, args=(f"worker_{i}",))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        # 计算负载均衡度
        total_calls = sum(api_usage.values())
        expected_per_api = total_calls / len(api_paths)
        variance = sum((count - expected_per_api)**2 for count in api_usage.values()) / len(api_usage)
        balance_score = 1 / (1 + variance)  # 分数越高越均衡
        
        print(f"  负载均衡分数: {balance_score:.3f} (1.0为完全均衡)")
        for api_path, count in sorted(api_usage.items()):
            print(f"    {api_path}: {count} 次")


def main():
    """主演示函数"""
    print("=" * 60)
    print("API 端点负载均衡改进演示")
    print("=" * 60)
    
    # 演示旧方法的问题
    demonstrate_old_approach()
    
    # 演示新方法的改进
    demonstrate_new_approach()
    
    # 演示不同任务量下的效果
    demonstrate_with_different_task_sizes()
    
    print("=" * 60)
    print("总结:")
    print("1. 旧方法: 固定API分配导致负载不均")
    print("2. 新方法: 队列机制实现动态负载均衡")
    print("3. 改进效果: 显著降低API使用分布的方差")
    print("4. 优势: 更好的资源利用率和系统稳定性")
    print("=" * 60)


if __name__ == "__main__":
    main()
