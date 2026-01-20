import json
import os
from datetime import datetime
import numpy as np
import torch


class ExperimentLogger:
    """
    实验记录器：保存训练和测试数据
    
    功能：
    - 创建实验目录
    - 保存训练指标（loss, score等）
    - 保存测试结果（JSON格式）
    - 保存模型检查点
    """
    
    def __init__(self, experiment_name, base_dir="./experiments"):
        """
        初始化实验记录器
        
        :param experiment_name: str, 实验名称（如 "dann_exp1", "baseline_exp1"）
        :param base_dir: str, 实验根目录
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        
        # 创建实验目录结构
        self.train_dir = os.path.join(self.experiment_dir, "train")
        self.test_dir = os.path.join(self.experiment_dir, "test")
        self.checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        
        for dir_path in [self.train_dir, self.test_dir, self.checkpoints_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 初始化数据存储
        self.train_data = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "episodes": [],
            "training_records": [],
            "running_scores": [],
            "source_domain_acc": [],
            "target_domain_acc": [],
            "losses": {
                "ppo_loss": [],
                "domain_loss": [],
                "total_loss": []
            }
        }
        
        self.test_data = {
            "experiment_name": experiment_name,
            "test_results": []
        }
        
        # 保存实验配置
        self.config = {}
        
    def log_config(self, config):
        """
        记录实验配置
        
        :param config: dict, 实验配置参数
        """
        self.config = config
        self.config["start_time"] = datetime.now().isoformat()
        config_path = os.path.join(self.experiment_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def log_training_step(self, episode, score, running_score, 
                         source_acc=None, target_acc=None,
                         ppo_loss=None, domain_loss=None, total_loss=None,
                         c1_score=None, c2_score=None):
        """
        记录训练步骤数据
        
        :param episode: int, episode 编号
        :param score: float, 当前 episode 得分
        :param running_score: float, 移动平均得分
        :param source_acc: float, 源域分类准确率（可选）
        :param target_acc: float, 目标域分类准确率（可选）
        :param ppo_loss: float, PPO 损失（可选）
        :param domain_loss: float, 域分类损失（可选）
        :param total_loss: float, 总损失（可选）
        :param c1_score: float, c1 环境测试得分（可选）
        :param c2_score: float, c2 环境测试得分（可选）
        """
        step_data = {
            "episode": episode,
            "score": float(score),
            "running_score": float(running_score),
            "timestamp": datetime.now().isoformat()
        }
        
        if source_acc is not None:
            step_data["source_domain_acc"] = float(source_acc)
            self.train_data["source_domain_acc"].append(float(source_acc))
        
        if target_acc is not None:
            step_data["target_domain_acc"] = float(target_acc)
            self.train_data["target_domain_acc"].append(float(target_acc))
        
        if ppo_loss is not None:
            step_data["ppo_loss"] = float(ppo_loss)
            self.train_data["losses"]["ppo_loss"].append(float(ppo_loss))
        
        if domain_loss is not None:
            step_data["domain_loss"] = float(domain_loss)
            self.train_data["losses"]["domain_loss"].append(float(domain_loss))
        
        if total_loss is not None:
            step_data["total_loss"] = float(total_loss)
            self.train_data["losses"]["total_loss"].append(float(total_loss))
        
        if c1_score is not None:
            step_data["c1_score"] = float(c1_score)
        
        if c2_score is not None:
            step_data["c2_score"] = float(c2_score)
        
        self.train_data["episodes"].append(step_data)
        self.train_data["training_records"].append(float(score))
        self.train_data["running_scores"].append(float(running_score))
    
    def save_training_data(self, filename=None):
        """
        保存训练数据到 JSON 文件
        
        :param filename: str, 文件名（可选，默认使用时间戳）
        """
        if filename is None:
            filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        self.train_data["end_time"] = datetime.now().isoformat()
        filepath = os.path.join(self.train_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.train_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def log_test_result(self, test_name, results, metadata=None):
        """
        记录测试结果
        
        :param test_name: str, 测试名称（如 "final_test", "episode_100"）
        :param results: dict, 测试结果，例如：
            {
                "source_env_score": 100.5,
                "target_env_c1_score": 80.3,
                "target_env_c2_score": 75.2,
                "mean_score": 85.0
            }
        :param metadata: dict, 额外的元数据（可选）
        """
        test_result = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        if metadata:
            test_result["metadata"] = metadata
        
        self.test_data["test_results"].append(test_result)
    
    def save_test_data(self, filename=None):
        """
        保存测试数据到 JSON 文件
        
        :param filename: str, 文件名（可选，默认使用时间戳）
        """
        if filename is None:
            filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.test_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_checkpoint(self, model, optimizer, episode, filename=None):
        """
        保存模型检查点
        
        :param model: torch.nn.Module, 模型
        :param optimizer: torch.optim.Optimizer, 优化器
        :param episode: int, episode 编号
        :param filename: str, 文件名（可选）
        """
        if filename is None:
            filename = f"checkpoint_ep{episode:04d}.pth"
        
        checkpoint = {
            "episode": episode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        filepath = os.path.join(self.checkpoints_dir, filename)
        torch.save(checkpoint, filepath)
        
        return filepath
    
    def load_checkpoint(self, filepath, model, optimizer=None):
        """
        加载模型检查点
        
        :param filepath: str, 检查点文件路径
        :param model: torch.nn.Module, 模型
        :param optimizer: torch.optim.Optimizer, 优化器（可选）
        :return: int, episode 编号
        """
        checkpoint = torch.load(filepath, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint["episode"]
    
    def get_experiment_summary(self):
        """
        获取实验摘要
        
        :return: dict, 实验摘要
        """
        summary = {
            "experiment_name": self.experiment_name,
            "experiment_dir": self.experiment_dir,
            "start_time": self.train_data.get("start_time"),
            "end_time": self.train_data.get("end_time"),
            "total_episodes": len(self.train_data["episodes"]),
            "final_running_score": self.train_data["running_scores"][-1] if self.train_data["running_scores"] else None,
            "best_score": max(self.train_data["training_records"]) if self.train_data["training_records"] else None,
            "test_count": len(self.test_data["test_results"])
        }
        
        return summary
