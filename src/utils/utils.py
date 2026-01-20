import numpy as np
import torch


def get_random_buffer(buffer, batch_size, similarity_threshold=0.2, device=None, used_colors=None):
    """
    生成随机背景颜色的缓冲区（用于域适应）
    
    :param buffer: torch.Tensor, 源域缓冲区
    :param batch_size: int, 批次大小
    :param similarity_threshold: float, 相似度阈值
    :param device: torch.device, 设备
    :param used_colors: set, 已使用的颜色集合，用于避免重复
    :return: tuple, (修改后的缓冲区, 新的颜色集合)
    """
    if device is None:
        device = buffer.device
    
    if used_colors is None:
        used_colors = set()
    
    ORG_B0 = torch.tensor([100, 202, 100], device=device).float()
    ORG_B1 = torch.tensor([100, 228, 100], device=device).float()

    ORG_B0_norm = ORG_B0 / 128. - 1
    ORG_B1_norm = ORG_B1 / 128. - 1

    # 生成新的背景颜色，确保不与已使用的颜色重复
    max_attempts = 100
    for attempt in range(max_attempts):
        new_background_color_0 = 2 * torch.rand(buffer.shape[-1]).double() - 1
        new_background_color_1 = 2 * torch.rand(buffer.shape[-1]).double() - 1
        
        # 将颜色转换为可哈希的元组
        color_0_tuple = tuple(new_background_color_0.cpu().numpy().round(decimals=3))
        color_1_tuple = tuple(new_background_color_1.cpu().numpy().round(decimals=3))
        
        # 检查是否与已使用的颜色相似（避免生成过于相似的颜色）
        is_similar = False
        for used_color in used_colors:
            if len(used_color) == len(color_0_tuple):
                diff_0 = sum(abs(a - b) for a, b in zip(used_color, color_0_tuple))
                diff_1 = sum(abs(a - b) for a, b in zip(used_color, color_1_tuple))
                if diff_0 < 0.1 or diff_1 < 0.1:  # 如果颜色差异小于阈值，认为相似
                    is_similar = True
                    break
        
        if not is_similar:
            used_colors.add(color_0_tuple)
            used_colors.add(color_1_tuple)
            break
    
    new_background_color_0 = new_background_color_0.to(device)
    new_background_color_1 = new_background_color_1.to(device)

    color_difference_B0 = torch.abs(buffer - ORG_B0_norm)
    background_mask = torch.all(color_difference_B0 <= similarity_threshold, dim=-1)
    buffer[background_mask] = new_background_color_0

    color_difference_B1 = torch.abs(buffer - ORG_B1_norm)
    background_mask = torch.all(color_difference_B1 <= similarity_threshold, dim=-1)
    buffer[background_mask] = new_background_color_1

    return buffer, used_colors


def eval(agent, env):
    """
    评估智能体在环境中的表现
    
    :param agent: Agent, 智能体
    :param env: Env, 环境
    :return: float, 得分
    """
    score = 0
    state = env.reset()

    for t in range(1000):
        action, a_logp = agent.select_action(state)
        state_, reward, done, _ = env.step_eval(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
        score += reward
        state = state_

        if done:
            break

    return score
