#!/usr/bin/env python3
"""
简单的主程序测试 - 检查导入和基本功能
"""

import sys
sys.path.append('/aiarena/gpfs')

def test_imports():
    """测试关键模块的导入"""
    try:
        print("Testing imports...")
        
        # 测试主要类的导入
        from corner_case_gen_parallel import (
            LoggerManager, 
            OpenAIClient, 
            SandboxClient, 
            CornerCaseGenerator, 
            Corner_Case_Model,
            initialize_logger_manager,
            log_global,
            log_sample
        )
        
        print("✓ All imports successful!")
        
        # 测试日志系统
        print("\nTesting logger system...")
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        initialize_logger_manager(temp_dir)
        log_global("Test global message")
        log_sample("test_sample", "Test sample message")
        
        print("✓ Logger system works!")
        
        # 测试OpenAI客户端创建
        print("\nTesting OpenAI client creation...")
        API_BASE = "https://lonlie.plus7.plus/v1"
        API_KEY = "sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"
        
        client = OpenAIClient(API_BASE, API_KEY)
        print("✓ OpenAI client created successfully!")
        
        # 测试Pydantic模型
        print("\nTesting Pydantic model...")
        model = Corner_Case_Model(
            replace_corner_case_list=[],
            add_corner_case_list=["test1", "test2"]
        )
        print(f"✓ Pydantic model works! Add list: {model.add_corner_case_list}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All basic tests passed! The system should be ready to run.")
    else:
        print("\n❌ Basic tests failed!")
