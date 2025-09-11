#!/usr/bin/env python3
"""
测试日志系统功能的简单脚本
"""

import os
import tempfile
import shutil
from corner_case_gen_parallel import LoggerManager, initialize_logger_manager, log_global, log_sample

def test_logging_system():
    """测试日志系统的基本功能"""
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="log_test_")
    
    try:
        print(f"Testing logging system in: {temp_dir}")
        
        # 初始化日志管理器
        initialize_logger_manager(temp_dir)
        
        # 测试全局日志
        log_global("这是一条全局日志消息")
        log_global("Testing global logging system")
        log_global("全局日志测试 - 中文支持")
        
        # 测试样本日志
        sample_id1 = "Codeforces/123/A"
        sample_id2 = "test/sample_456"
        
        log_sample(sample_id1, "这是样本1的第一条日志")
        log_sample(sample_id1, "样本1的第二条日志 - 包含特殊字符: !@#$%")
        log_sample(sample_id2, "Sample 2 logging test")
        log_sample(sample_id2, "样本2的日志 - 测试中文字符")
        
        # 再次记录全局日志
        log_global("处理完成，共处理了2个样本")
        
        # 检查日志文件是否创建
        log_dir = os.path.join(temp_dir, "log")
        
        print(f"\n检查日志目录: {log_dir}")
        if os.path.exists(log_dir):
            print("✓ 日志目录已创建")
            
            # 检查全局日志文件
            global_log = os.path.join(log_dir, "global.log")
            if os.path.exists(global_log):
                print("✓ 全局日志文件已创建")
                with open(global_log, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"全局日志内容 ({len(content.splitlines())} 行):")
                    print(content)
            else:
                print("✗ 全局日志文件未找到")
            
            # 检查样本日志文件
            sample1_log = os.path.join(log_dir, "Codeforces_123_A.log")
            sample2_log = os.path.join(log_dir, "test_sample_456.log")
            
            for sample_log, sample_name in [(sample1_log, "样本1"), (sample2_log, "样本2")]:
                if os.path.exists(sample_log):
                    print(f"✓ {sample_name}日志文件已创建")
                    with open(sample_log, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"{sample_name}日志内容 ({len(content.splitlines())} 行):")
                        print(content)
                else:
                    print(f"✗ {sample_name}日志文件未找到")
            
            # 列出所有文件
            print(f"\n日志目录中的所有文件:")
            for file in os.listdir(log_dir):
                print(f"  - {file}")
                
        else:
            print("✗ 日志目录未创建")
        
        print("\n测试完成!")
        
    finally:
        # 清理临时目录
        print(f"清理临时目录: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_logging_system()
