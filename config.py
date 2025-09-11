"""
配置文件 - 用于管理并行 corner case 生成的各种参数
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class OpenAIConfig:
    """OpenAI API 配置"""
    api_base: str = "https://lonlie.plus7.plus/v1"
    api_key: str = "sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"
    model: str = "gpt-4o"
    max_tokens: int = 1000
    no_reasoning: bool = True  # 是否使用 reasoning 参数
    max_attempts: int = 3  # 最大尝试次数


@dataclass
class SandboxConfig:
    """Sandbox API 配置"""
    hosts: List[str] = None
    specific_hosts: List[str] = None  # 特定的主机列表
    base_port: int = 8080
    port_range: int = 4
    compile_timeout: int = 20
    run_timeout: int = 20
    
    def __post_init__(self):
        if self.hosts is None:
            self.hosts = ["10.244.230.127", "10.244.213.170"]
    
    def get_api_paths(self) -> List[str]:
        """生成所有 API 路径"""
        api_paths = []
        # ports_per_host = self.port_range // len(self.hosts)
        ports_per_host = self.port_range
        
        for host in self.hosts:
            for i in range(ports_per_host):
                port = self.base_port + i
                api_paths.append(f"http://{host}:{port}/")
        
        return api_paths
    
    def get_specific_api_paths(self) -> List[str]:
        """获取特定主机的 API 路径"""
        if not self.specific_hosts:
            return []
        
        api_paths = []
        for host in self.specific_hosts:
            for i in range(self.port_range):
                port = self.base_port + i
                api_paths.append(f"http://{host}:{port}/")
        
        return api_paths


@dataclass
class ProcessingConfig:
    """处理配置"""
    max_workers_per_api: int = 1
    max_iterations: int = 3
    max_sample_solutions: int = 3
    use_all_solutions: bool = False
    debug: bool = False
    save_intermediate_results: bool = True
    # 新增细粒度并行配置
    output_generation_workers: int = 4  # 输出生成的并行工作线程数
    solution_validation_workers: int = 4  # 解决方案验证的并行工作线程数
    sample_level_workers: int = 4  # sample 级别的并行度


@dataclass
class DatasetConfig:
    """数据集配置"""
    data_path: str = "/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered"
    split: str = "test"
    dataset_type: str = "code_contests_test"
    results_dir: str = "/aiarena/gpfs/Code-Contests-Ours/test/"
    
    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = None):
        self.openai_config = OpenAIConfig()
        self.sandbox_config = SandboxConfig()
        self.processing_config = ProcessingConfig()
        self.dataset_config = DatasetConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """从文件加载配置（可以实现为 JSON/YAML 格式）"""
        # 这里可以扩展为从 JSON 或 YAML 文件加载配置
        pass
    
    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        # 这里可以扩展为保存到 JSON 或 YAML 文件
        pass
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """获取运行时信息"""
        api_paths = self.sandbox_config.get_api_paths()
        total_workers = len(api_paths) * self.processing_config.max_workers_per_api
        
        return {
            "api_endpoints": len(api_paths),
            "total_workers": total_workers,
            "api_paths": api_paths,
            "dataset_path": self.dataset_config.data_path,
            "results_dir": self.dataset_config.results_dir,
            "max_iterations": self.processing_config.max_iterations,
        }
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查必要的路径
            if not os.path.exists(self.dataset_config.data_path):
                print(f"Dataset path does not exist: {self.dataset_config.data_path}")
                return False
            
            # 检查 API 配置
            if not self.openai_config.api_key:
                print("OpenAI API key is not set")
                return False
            
            # 检查 sandbox 配置
            if not self.sandbox_config.hosts:
                print("No sandbox hosts configured")
                return False
            
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False


# 预定义的配置模板
DEVELOPMENT_CONFIG = {
    "sandbox_config": {
        "hosts": ["localhost"],
        "port_range": 2,
    },
    "processing_config": {
        "max_workers_per_api": 1,
        "max_iterations": 2,
        "debug": True,
    }
}

PRODUCTION_CONFIG = {
    "sandbox_config": {
        "hosts": [
            "10.244.188.149",
            "10.244.40.134",
            "10.244.204.96",
            "10.244.128.68",
            "10.244.81.216",
            "10.244.166.233",
        ],
        "specific_hosts": [
            "10.244.179.6",

        ],            
        "port_range": 128,  # 每个 host 使用 128 个端口
    },
    "processing_config": {
        "max_workers_per_api": 1,
        "max_iterations": 3,
        "max_sample_solutions": 10,
        "use_all_solutions": True,  # 是否使用所有解决方案
        "sample_level_workers": 4,  # sample 级别的并行度
        "output_generation_workers": 128,
        "solution_validation_workers": 128 * 4,
        "debug": False,
    },
    "dataset_config": {
        "data_path": "/aiarena/gpfs/Code-Contests-Uni/test/test_dataset_filtered",
        "split": "test",
        "dataset_type": "code_contests_test",
        "results_dir": "/aiarena/group/llmgroup/caijf/Code-Contests-Ours/test_all_solutions_gpt5_new_replace_add_feedback_gen_command_replace_new_new/",
    },
    "openai_config": {
        "model": "gpt-5",
        "max_tokens": 4000,
        "no_reasoning": False,  # 是否使用 reasoning 参数
    }
}
