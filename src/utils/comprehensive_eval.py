"""
全面评估系统：用于进行统计严谨的评估，符合学术标准

功能：
- 多次运行取平均和标准差
- 计算成功率、域间隙等指标
- 支持多随机种子实验
- 生成统计报告
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import json

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    # 如果没有 scipy，使用 numpy 的简单实现
    import warnings
    warnings.warn("scipy not available, some statistical functions may be limited")


class ComprehensiveEvaluator:
    """
    全面评估器：进行统计严谨的评估
    """
    
    def __init__(self, num_runs: int = 10, max_steps: int = 1000):
        """
        初始化评估器
        
        :param num_runs: int, 每次评估运行的次数（用于计算统计量）
        :param max_steps: int, 每个 episode 的最大步数
        """
        self.num_runs = num_runs
        self.max_steps = max_steps
    
    def evaluate_agent(self, agent, env, num_episodes: int = 1) -> Dict:
        """
        评估智能体在环境中的表现（多次运行取统计量）
        
        :param agent: Agent, 智能体
        :param env: Env, 环境
        :param num_episodes: int, 每个运行包含的 episode 数
        :return: dict, 包含均值、标准差、置信区间等统计信息
        """
        scores = []
        episode_lengths = []
        success_count = 0  # 成功完成任务的次数
        
        for run in range(self.num_runs):
            total_score = 0
            episode_length = 0
            
            for ep in range(num_episodes):
                score = 0
                state = env.reset()
                
                for t in range(self.max_steps):
                    action, a_logp = agent.select_action(state)
                    state_, reward, done, _ = env.step_eval(
                        action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])
                    )
                    score += reward
                    state = state_
                    episode_length += 1
                    
                    if done:
                        break
                
                total_score += score
            
            avg_score = total_score / num_episodes
            scores.append(avg_score)
            episode_lengths.append(episode_length / num_episodes)
            
            # 定义成功：得分大于某个阈值（例如 > 0）
            if avg_score > 0:
                success_count += 1
        
        # 计算统计量
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        median_score = np.median(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # 计算 95% 置信区间
        if len(scores) > 1:
            if HAS_SCIPY:
                confidence_interval = stats.t.interval(
                    0.95, len(scores) - 1,
                    loc=mean_score,
                    scale=stats.sem(scores)
                )
            else:
                # 使用简化的正态分布近似
                sem = std_score / np.sqrt(len(scores))
                confidence_interval = (
                    mean_score - 1.96 * sem,
                    mean_score + 1.96 * sem
                )
        else:
            confidence_interval = (mean_score, mean_score)
        
        success_rate = success_count / self.num_runs
        
        return {
            "mean_score": float(mean_score),
            "std_score": float(std_score),
            "median_score": float(median_score),
            "min_score": float(min_score),
            "max_score": float(max_score),
            "confidence_interval_95": [float(confidence_interval[0]), float(confidence_interval[1])],
            "success_rate": float(success_rate),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "num_runs": self.num_runs,
            "all_scores": [float(s) for s in scores]  # 保存所有分数用于进一步分析
        }
    
    def evaluate_domain_adaptation(
        self,
        agent,
        source_env,
        target_envs: List,
        num_episodes: int = 1
    ) -> Dict:
        """
        评估域适应效果：计算源域和目标域之间的性能差距
        
        :param agent: Agent, 智能体
        :param source_env: Env, 源域环境
        :param target_envs: List[Env], 目标域环境列表
        :param num_episodes: int, 每个运行包含的 episode 数
        :return: dict, 包含域适应评估结果
        """
        # 评估源域性能
        source_results = self.evaluate_agent(agent, source_env, num_episodes)
        
        # 评估每个目标域性能
        target_results = []
        for i, target_env in enumerate(target_envs):
            target_result = self.evaluate_agent(agent, target_env, num_episodes)
            target_result["domain_id"] = f"target_{i+1}"
            target_results.append(target_result)
        
        # 计算域间隙（Domain Gap）
        source_score = source_results["mean_score"]
        target_scores = [r["mean_score"] for r in target_results]
        mean_target_score = np.mean(target_scores)
        
        domain_gap = source_score - mean_target_score
        domain_gap_reduction = None  # 需要与 baseline 对比才能计算
        
        # 计算泛化率（Generalization Ratio）
        generalization_ratio = mean_target_score / source_score if source_score > 0 else 0
        
        return {
            "source_domain": source_results,
            "target_domains": target_results,
            "domain_gap": float(domain_gap),
            "mean_target_score": float(mean_target_score),
            "generalization_ratio": float(generalization_ratio),
            "domain_gap_reduction": domain_gap_reduction
        }
    
    def compare_methods(
        self,
        methods: Dict[str, Tuple],  # {"method_name": (agent, source_env, target_envs)}
        num_episodes: int = 1
    ) -> Dict:
        """
        对比多个方法（例如 DANN vs Baseline）
        
        :param methods: dict, 方法名称到 (agent, source_env, target_envs) 的映射
        :param num_episodes: int, 每个运行包含的 episode 数
        :return: dict, 对比结果
        """
        comparison_results = {}
        
        for method_name, (agent, source_env, target_envs) in methods.items():
            results = self.evaluate_domain_adaptation(
                agent, source_env, target_envs, num_episodes
            )
            comparison_results[method_name] = results
        
        # 计算相对改进（如果有 baseline）
        if "baseline" in comparison_results and len(comparison_results) > 1:
            baseline_target_score = comparison_results["baseline"]["mean_target_score"]
            
            for method_name, results in comparison_results.items():
                if method_name != "baseline":
                    improvement = results["mean_target_score"] - baseline_target_score
                    relative_improvement = (improvement / baseline_target_score * 100) if baseline_target_score != 0 else 0
                    
                    results["improvement_over_baseline"] = float(improvement)
                    results["relative_improvement_percent"] = float(relative_improvement)
                    
                    # 计算域间隙减少
                    baseline_gap = comparison_results["baseline"]["domain_gap"]
                    method_gap = results["domain_gap"]
                    gap_reduction = baseline_gap - method_gap
                    results["domain_gap_reduction"] = float(gap_reduction)
                    results["domain_gap_reduction_percent"] = float(
                        (gap_reduction / baseline_gap * 100) if baseline_gap != 0 else 0
                    )
        
        return comparison_results
    
    def statistical_significance_test(
        self,
        method1_scores: List[float],
        method2_scores: List[float],
        alpha: float = 0.05
    ) -> Dict:
        """
        进行统计显著性检验（t-test）
        
        :param method1_scores: List[float], 方法1的分数列表
        :param method2_scores: List[float], 方法2的分数列表
        :param alpha: float, 显著性水平
        :return: dict, 检验结果
        """
        if len(method1_scores) < 2 or len(method2_scores) < 2:
            return {
                "p_value": None,
                "significant": False,
                "note": "Insufficient data for statistical test"
            }
        
        # 进行独立样本 t 检验
        if HAS_SCIPY:
            t_stat, p_value = stats.ttest_ind(method1_scores, method2_scores)
        else:
            # 简化的 t 检验实现
            n1, n2 = len(method1_scores), len(method2_scores)
            mean1, mean2 = np.mean(method1_scores), np.mean(method2_scores)
            var1, var2 = np.var(method1_scores, ddof=1), np.var(method2_scores, ddof=1)
            
            pooled_std = np.sqrt((var1/n1 + var2/n2))
            if pooled_std == 0:
                t_stat, p_value = 0, 1.0
            else:
                t_stat = (mean1 - mean2) / pooled_std
                # 简化的 p-value 计算（使用正态分布近似）
                # 简化的 p-value 近似（使用正态分布）
                # 对于大样本，t 分布接近正态分布
                z_score = abs(t_stat)
                # 使用经验公式近似 p-value
                if z_score > 3:
                    p_value = 0.001
                elif z_score > 2:
                    p_value = 0.05
                elif z_score > 1.96:
                    p_value = 0.1
                else:
                    p_value = 0.5
        
        is_significant = p_value < alpha
        
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(is_significant),
            "alpha": alpha,
            "interpretation": "Statistically significant" if is_significant else "Not statistically significant"
        }
    
    def generate_evaluation_report(
        self,
        evaluation_results: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """
        生成评估报告
        
        :param evaluation_results: dict, 评估结果
        :param output_path: str, 输出文件路径（可选）
        :return: str, 报告文本
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("Comprehensive Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        if "source_domain" in evaluation_results:
            # 单方法评估报告
            source = evaluation_results["source_domain"]
            report_lines.append("Source Domain Performance:")
            report_lines.append(f"  Mean Score: {source['mean_score']:.2f} ± {source['std_score']:.2f}")
            report_lines.append(f"  95% CI: [{source['confidence_interval_95'][0]:.2f}, {source['confidence_interval_95'][1]:.2f}]")
            report_lines.append(f"  Success Rate: {source['success_rate']:.2%}")
            report_lines.append("")
            
            for target in evaluation_results["target_domains"]:
                report_lines.append(f"Target Domain {target['domain_id']} Performance:")
                report_lines.append(f"  Mean Score: {target['mean_score']:.2f} ± {target['std_score']:.2f}")
                report_lines.append(f"  95% CI: [{target['confidence_interval_95'][0]:.2f}, {target['confidence_interval_95'][1]:.2f}]")
                report_lines.append(f"  Success Rate: {target['success_rate']:.2%}")
                report_lines.append("")
            
            report_lines.append("Domain Adaptation Metrics:")
            report_lines.append(f"  Domain Gap: {evaluation_results['domain_gap']:.2f}")
            report_lines.append(f"  Generalization Ratio: {evaluation_results['generalization_ratio']:.2%}")
            
            if evaluation_results.get("domain_gap_reduction"):
                report_lines.append(f"  Domain Gap Reduction: {evaluation_results['domain_gap_reduction']:.2f}")
        
        else:
            # 多方法对比报告
            for method_name, results in evaluation_results.items():
                report_lines.append(f"{method_name.upper()} Results:")
                report_lines.append(f"  Source Score: {results['source_domain']['mean_score']:.2f} ± {results['source_domain']['std_score']:.2f}")
                report_lines.append(f"  Mean Target Score: {results['mean_target_score']:.2f}")
                report_lines.append(f"  Domain Gap: {results['domain_gap']:.2f}")
                
                if "improvement_over_baseline" in results:
                    report_lines.append(f"  Improvement over Baseline: {results['improvement_over_baseline']:.2f}")
                    report_lines.append(f"  Relative Improvement: {results['relative_improvement_percent']:.2f}%")
                    report_lines.append(f"  Domain Gap Reduction: {results['domain_gap_reduction']:.2f} ({results['domain_gap_reduction_percent']:.2f}%)")
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
                # 同时保存 JSON 格式
                json_path = output_path.replace('.txt', '.json')
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump(evaluation_results, jf, indent=2, ensure_ascii=False)
        
        return report_text
