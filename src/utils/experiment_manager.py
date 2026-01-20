import json
import os
from datetime import datetime
from pathlib import Path


class ExperimentManager:
    """
    实验管理器：用于管理多个实验，方便对比和分析
    
    功能：
    - 列出所有实验
    - 对比不同实验的结果
    - 生成实验报告
    """
    
    def __init__(self, base_dir="./experiments"):
        """
        初始化实验管理器
        
        :param base_dir: str, 实验根目录
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def list_experiments(self):
        """
        列出所有实验
        
        :return: list, 实验名称列表
        """
        if not os.path.exists(self.base_dir):
            return []
        
        experiments = []
        for item in os.listdir(self.base_dir):
            exp_path = os.path.join(self.base_dir, item)
            if os.path.isdir(exp_path):
                config_path = os.path.join(exp_path, "config.json")
                if os.path.exists(config_path):
                    experiments.append(item)
        
        return sorted(experiments)
    
    def get_experiment_info(self, experiment_name):
        """
        获取实验信息
        
        :param experiment_name: str, 实验名称
        :return: dict, 实验信息
        """
        exp_dir = os.path.join(self.base_dir, experiment_name)
        config_path = os.path.join(exp_dir, "config.json")
        
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 读取训练数据
        train_dir = os.path.join(exp_dir, "train")
        test_dir = os.path.join(exp_dir, "test")
        
        train_files = [f for f in os.listdir(train_dir) if f.endswith('.json')]
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.json')]
        
        # 读取最新的训练数据
        latest_train = None
        if train_files:
            latest_train_file = sorted(train_files)[-1]
            with open(os.path.join(train_dir, latest_train_file), 'r', encoding='utf-8') as f:
                latest_train = json.load(f)
        
        # 读取最新的测试数据
        latest_test = None
        if test_files:
            latest_test_file = sorted(test_files)[-1]
            with open(os.path.join(test_dir, latest_test_file), 'r', encoding='utf-8') as f:
                latest_test = json.load(f)
        
        return {
            "experiment_name": experiment_name,
            "config": config,
            "latest_training_data": latest_train,
            "latest_test_data": latest_test,
            "train_files": train_files,
            "test_files": test_files
        }
    
    def compare_experiments(self, experiment_names):
        """
        对比多个实验的结果
        
        :param experiment_names: list, 实验名称列表
        :return: dict, 对比结果
        """
        comparison = {
            "experiments": [],
            "summary": {}
        }
        
        for exp_name in experiment_names:
            info = self.get_experiment_info(exp_name)
            if info:
                comparison["experiments"].append({
                    "name": exp_name,
                    "method": info["config"].get("method", "Unknown"),
                    "final_running_score": None,
                    "best_score": None,
                    "final_c1_score": None,
                    "final_c2_score": None,
                    "mean_target_score": None
                })
                
                # 提取关键指标
                if info["latest_training_data"]:
                    train_data = info["latest_training_data"]
                    if train_data.get("running_scores"):
                        comparison["experiments"][-1]["final_running_score"] = train_data["running_scores"][-1]
                    if train_data.get("training_records"):
                        comparison["experiments"][-1]["best_score"] = max(train_data["training_records"])
                
                if info["latest_test_data"] and info["latest_test_data"].get("test_results"):
                    last_test = info["latest_test_data"]["test_results"][-1]
                    results = last_test.get("results", {})
                    comparison["experiments"][-1]["final_c1_score"] = results.get("target_env_c1_score")
                    comparison["experiments"][-1]["final_c2_score"] = results.get("target_env_c2_score")
                    comparison["experiments"][-1]["mean_target_score"] = results.get("mean_target_score")
        
        # 生成摘要
        if comparison["experiments"]:
            methods = [exp["method"] for exp in comparison["experiments"]]
            comparison["summary"] = {
                "total_experiments": len(comparison["experiments"]),
                "methods": list(set(methods)),
                "best_final_score": max([exp["final_running_score"] for exp in comparison["experiments"] if exp["final_running_score"] is not None], default=None),
                "best_target_score": max([exp["mean_target_score"] for exp in comparison["experiments"] if exp["mean_target_score"] is not None], default=None)
            }
        
        return comparison
    
    def generate_report(self, output_file="experiment_report.json"):
        """
        生成所有实验的报告
        
        :param output_file: str, 输出文件名
        :return: str, 报告文件路径
        """
        experiments = self.list_experiments()
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_experiments": len(experiments),
            "experiments": []
        }
        
        for exp_name in experiments:
            info = self.get_experiment_info(exp_name)
            if info:
                report["experiments"].append({
                    "name": exp_name,
                    "method": info["config"].get("method", "Unknown"),
                    "start_time": info["config"].get("start_time"),
                    "config": info["config"]
                })
        
        report_path = os.path.join(self.base_dir, output_file)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report_path
