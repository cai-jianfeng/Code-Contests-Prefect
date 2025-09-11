#!/usr/bin/env python3
"""
独立测试日志系统功能的脚本
"""

import os
import tempfile
import shutil
import logging
import threading
from datetime import datetime

# 复制日志系统代码进行测试
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
            # 如果没有设置日志器，回退到print
            print(f"[GLOBAL] {message}")
    
    def log_sample(self, sample_id: str, message: str):
        """记录样本日志"""
        logger = self.get_sample_logger(sample_id)
        if logger:
            logger.info(message)
        else:
            # 如果没有设置日志器，回退到print
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

def test_logging_system():
    """测试日志系统的基本功能"""
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="log_test_")
    
    try:
        print(f"Testing logging system in: {temp_dir}")
        
        # 创建日志管理器
        logger_manager = LoggerManager(temp_dir)
        
        # 测试全局日志
        logger_manager.log_global("这是一条全局日志消息")
        logger_manager.log_global("Testing global logging system")
        logger_manager.log_global("全局日志测试 - 中文支持")
        
        # 测试样本日志
        sample_id1 = "Codeforces/123/A"
        sample_id2 = "test/sample_456"
        
        logger_manager.log_sample(sample_id1, "这是样本1的第一条日志")
        logger_manager.log_sample(sample_id1, "样本1的第二条日志 - 包含特殊字符: !@#$%")
        logger_manager.log_sample(sample_id2, "Sample 2 logging test")
        logger_manager.log_sample(sample_id2, "样本2的日志 - 测试中文字符")
        
        # 再次记录全局日志
        logger_manager.log_global("处理完成，共处理了2个样本")
        
        # 清理日志器以确保文件写入
        logger_manager.cleanup()
        
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
