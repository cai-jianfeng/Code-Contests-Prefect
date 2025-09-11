#!/usr/bin/env python3
"""
测试队列打乱功能的脚本
"""

import queue
import threading
import random
import time
from solutions_eval_original_test import shuffle_queue_safely

def test_queue_shuffle():
    """测试队列打乱功能"""
    
    # 创建测试队列和锁
    test_queue = queue.Queue()
    queue_lock = threading.Lock()
    
    # 添加一些测试任务
    test_tasks = []
    for i in range(10):
        task = {
            'sample_id': f'sample_{i}',
            'type': 'solution',
            'task_order': i
        }
        test_tasks.append(task)
        test_queue.put(task)
    
    print("Original task order:")
    # 记录原始顺序（不移除任务）
    original_order = []
    temp_tasks = []
    while not test_queue.empty():
        task = test_queue.get()
        original_order.append(task['task_order'])
        temp_tasks.append(task)
    
    # 将任务重新放回队列
    for task in temp_tasks:
        test_queue.put(task)
    
    print(f"Before shuffle: {original_order}")
    
    # 测试打乱功能
    shuffle_queue_safely(test_queue, queue_lock)
    
    # 检查打乱后的顺序
    shuffled_order = []
    while not test_queue.empty():
        task = test_queue.get()
        shuffled_order.append(task['task_order'])
    
    print(f"After shuffle: {shuffled_order}")
    
    # 验证所有任务都还在
    if sorted(original_order) == sorted(shuffled_order):
        print("✅ All tasks preserved after shuffle")
    else:
        print("❌ Some tasks lost during shuffle")
    
    # 验证顺序确实改变了
    if original_order != shuffled_order:
        print("✅ Task order successfully changed")
    else:
        print("⚠️ Task order unchanged (could happen by chance)")

def test_concurrent_access():
    """测试并发访问时的线程安全性"""
    
    test_queue = queue.Queue()
    queue_lock = threading.Lock()
    
    # 添加更多任务用于并发测试
    for i in range(50):
        task = {'id': i, 'data': f'task_{i}'}
        test_queue.put(task)
    
    results = []
    results_lock = threading.Lock()
    
    def worker():
        """模拟worker获取任务"""
        local_results = []
        for _ in range(5):  # 每个worker获取5个任务
            try:
                with queue_lock:
                    if not test_queue.empty():
                        task = test_queue.get()
                        local_results.append(task['id'])
                time.sleep(0.01)  # 模拟处理时间
            except:
                break
        
        with results_lock:
            results.extend(local_results)
    
    def shuffler():
        """模拟shuffle操作"""
        time.sleep(0.05)  # 等待一些任务被处理
        shuffle_queue_safely(test_queue, queue_lock)
        print("Queue shuffled during concurrent access")
    
    # 启动多个worker和一个shuffler
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)
    
    shuffle_thread = threading.Thread(target=shuffler)
    shuffle_thread.start()
    threads.append(shuffle_thread)
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print(f"Processed {len(results)} tasks with concurrent shuffle")
    print(f"Remaining tasks in queue: {test_queue.qsize()}")
    
    if len(set(results)) == len(results):
        print("✅ No duplicate tasks processed")
    else:
        print("❌ Duplicate tasks detected")

if __name__ == "__main__":
    print("=== Testing Queue Shuffle Functionality ===")
    test_queue_shuffle()
    print("\n=== Testing Concurrent Access Safety ===")
    test_concurrent_access()
