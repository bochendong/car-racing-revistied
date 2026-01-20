import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler, BatchSampler
from torch.distributions import Beta
import matplotlib.pyplot as plt
from ..utils.utils import get_random_buffer


class Agent:
    max_grad_norm = 0.5
    clip_param = 0.1

    transition = np.dtype([
        ('s', np.float64, (4, 96, 96, 3)),
        ('a', np.float64, (3,)),
        ('a_logp', np.float64),
        ('r', np.float64),
        ('s_', np.float64, (4, 96, 96, 3))
    ])

    def __init__(self, net, criterion, optimizer, buffer_capacity=2000, batch_size=128, device=None):
        self.net = net
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.ppo_epoch = 10
        self.device = device

        self.source_buffer = np.empty(self.buffer_capacity, dtype=self.transition)
        self.counter = 0
        self.used_colors = set()  # 用于跟踪已使用的颜色，避免重复

    def _prepare_tensors(self):
        """准备张量数据"""
        s = torch.tensor(self.source_buffer['s'], dtype=torch.double).to(self.device)
        a = torch.tensor(self.source_buffer['a'], dtype=torch.double).to(self.device)
        r = torch.tensor(self.source_buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
        s_ = torch.tensor(self.source_buffer['s_'], dtype=torch.double).to(self.device)
        old_a_logp = torch.tensor(self.source_buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        return s, a, r, s_, old_a_logp

    def select_action(self, state):
        """选择动作"""
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.net.sketch(state)
            out = torch.squeeze(out)
            out = self.net.feature_extractor(out)
            out = self.net.cnn_base(out)
            out = out.view(-1, 256)
            out = self.net.fc(out)
            alpha = self.net.alpha_head(out) + 1
            beta = self.net.beta_head(out) + 1

        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def store(self, transition):
        """存储转换数据"""
        self.source_buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        return False
    
    def _prepare_domain_labels(self):
        """准备域标签"""
        source_domain_label = torch.zeros(self.batch_size).long().to(self.device)
        target_domain_label = torch.ones(self.batch_size).long().to(self.device)
        return source_domain_label, target_domain_label

    def _compute_advantage(self, s, r, s_):
        """计算优势函数"""
        with torch.no_grad():
            target_v = r + 0.99 * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
        return target_v, adv

    def _save_images(self, epoch, image_array):
        """保存图像"""
        f, axs = plt.subplots(10, 4, figsize=(8, 12))
        axs = axs.flatten()
        for img, ax in zip(image_array, axs):
            ax.imshow(img)
        f.savefig('./output_r/%04d.png' % epoch)
        plt.close(f)

    def PPO_Loss(self, a, alpha, beta, adv, v, target_v, old_a_logp, index):
        """计算 PPO 损失"""
        dist = Beta(alpha, beta)
        a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
        ratio = torch.exp(a_logp - old_a_logp[index])
        surr1 = ratio * adv[index]
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
        action_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.smooth_l1_loss(v, target_v[index])
        return action_loss + 2. * value_loss

    def update(self, epoch, eta=0.1):
        """更新智能体"""
        # 准备张量
        s, a, r, s_, old_a_logp = self._prepare_tensors()
        target_v, adv = self._compute_advantage(s, r, s_)
        source_label, target_label = self._prepare_domain_labels()

        image_array, source_acc_array, target_acc_array = [], [], []

        for _ in range(self.ppo_epoch):
            total, source_domain_correct, target_domain_correct = 0, 0, 0
            add_image = True
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True):
                total += self.batch_size
                source = s[index]
                target, self.used_colors = get_random_buffer(source.clone(), self.batch_size, device=self.device, used_colors=self.used_colors)

                # 域损失
                (alpha, beta), v, domain_out, s_sketch = self.net(source, eta)
                source_loss = self.criterion(domain_out, source_label)
                source_domain_correct += ((torch.argmax(domain_out, dim=1) == source_label).sum().item())

                _, _, domain_out, t_sketch = self.net(target, eta)
                target_loss = self.criterion(domain_out, target_label)
                target_domain_correct += ((torch.argmax(domain_out, dim=1) == target_label).sum().item())

                domain_classify_loss = source_loss + target_loss

                # RL 智能体损失
                ppo_loss = self.PPO_Loss(a, alpha, beta, adv, v, target_v, old_a_logp, index)

                # 更新智能体
                loss = ppo_loss + domain_classify_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if add_image:
                    s_sketch = s_sketch[0][0].reshape(96, 96).cpu().detach().numpy()
                    t_sketch = t_sketch[0][0].reshape(96, 96).cpu().detach().numpy()
                    s_sketch, t_sketch = (s_sketch + 1) / 2, (t_sketch + 1) / 2

                    s_img = source[0][0].cpu().detach().numpy()
                    t_img = target[0][0].cpu().detach().numpy()
                    s_img, t_img = (s_img + 1) / 2, (t_img + 1) / 2

                    image_array.extend([s_img, s_sketch, t_img, t_sketch])
                    add_image = False

            source_acc_array.append(source_domain_correct / total)
            target_acc_array.append(target_domain_correct / total)

        self._save_images(epoch, image_array)
        mean_source_acc = np.mean(source_acc_array)
        mean_target_acc = np.mean(target_acc_array)
        print('Source Acc:', mean_source_acc, 'Target Acc:', mean_target_acc)
        
        # 返回准确率和损失值，方便记录
        return {
            'source_acc': mean_source_acc,
            'target_acc': mean_target_acc
        }
