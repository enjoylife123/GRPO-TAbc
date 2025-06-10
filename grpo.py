import torch
import torch.nn as nn
from copy import deepcopy


class Memory:
    """
    用于存储强化学习过程中的转换数据。
    """

    def __init__(self):
        self.states = []  # 状态列表
        self.history_states = []  # 历史状态列表（上一次的嵌入向量）
        self.actions = []  # 动作列表
        self.rewards = []  # 奖励列表
        self.logprobs = []  # 动作的对数概率列表
        self.dones = []  # 完成状态列表
        self.edge_indices = []  # 边索引列表

    def clear_memory(self):
        """清空存储的数据"""
        self.states.clear()
        self.history_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.logprobs.clear()
        self.dones.clear()
        self.edge_indices.clear()

    def get_batch(self, device):
        """
        将存储的数据转换为批量张量。
        """
        states = torch.stack(self.states).to(device).detach()
        history_states = torch.stack(self.history_states).to(device).detach() if self.history_states else None
        actions = torch.stack(self.actions).to(device).detach()
        logprobs = torch.stack(self.logprobs).to(device).detach()
        edge_indices = torch.stack(self.edge_indices).to(device).detach()
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device).detach()
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device).detach()

        return states, history_states, actions, logprobs, edge_indices, rewards, dones


class GRPO:
    """
    GRPO 算法实现，使用所有批次的平均损失计算梯度后更新。
    """

    def __init__(self,
                 policy_net,  # 策略网络
                 lr=1e-4,  # 学习率
                 gamma=0.99,  # 折扣因子
                 k_epochs=4,  # 更新次数
                 eps_clip=0.2,  # GRPO 的 clip 参数
                 entropy_coef=0.01,  # 熵正则化系数
                 device="cpu"):  # 设备
        self.last_loss = None  # 记录上一次的损失值
        self.max_delta = 0.1  # 允许的最大变化范围

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.device = device

        self.policy_net = policy_net.to(device)
        self.policy_old = deepcopy(self.policy_net).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()),
            lr=lr
        )
        self.MseLoss = nn.MSELoss()

    def update(self, memory, group_advantages, batch_size=32):
        """
        使用 GRPO 算法更新策略网络，先计算所有批次的平均损失，再计算梯度并更新。

        参数:
            memory (Memory): 存储的转换数据。
            group_advantages (torch.Tensor): 组间优势值。
            batch_size (int): 每个批次的样本数量，默认为 32。
        """
        # 将 memory 中的数据转换为批量张量
        states, history_states, actions, old_logprobs, edge_indices, rewards, dones = memory.get_batch(self.device)
        group_advantages = group_advantages.to(self.device).detach()

        # 计算总样本数
        num_samples = states.size(0)

        # 存储所有批次的损失
        total_policy_loss = 0
        total_entropy_loss = 0
        num_valid_batches = 0

        # 遍历所有批次，计算损失但不反向传播
        for i in range(0, num_samples, batch_size):
            # 获取当前批次的数据
            batch_states = states[i:i + batch_size]
            batch_history_states = history_states[i:i + batch_size] if history_states is not None else None
            batch_actions = actions[i:i + batch_size]
            batch_old_logprobs = old_logprobs[i:i + batch_size]
            batch_edge_indices = edge_indices[i:i + batch_size]
            batch_group_advantages = group_advantages[i:i + batch_size]

            # 如果剩余样本不足两个，从前面批次拿一个样本过来
            if batch_states.size(0) < 2 and i > 0:
                # 从前面批次拿一个样本
                prev_batch_states = states[i - 1:i]
                prev_batch_history_states = history_states[i - 1:i] if history_states is not None else None
                prev_batch_actions = actions[i - 1:i]
                prev_batch_old_logprobs = old_logprobs[i - 1:i]
                prev_batch_edge_indices = edge_indices[i - 1:i]
                prev_batch_group_advantages = group_advantages[i - 1:i]

                # 将当前批次与前面批次的样本合并
                batch_states = torch.cat([prev_batch_states, batch_states], dim=0)
                batch_history_states = torch.cat([prev_batch_history_states, batch_history_states],
                                                 dim=0) if history_states is not None else None
                batch_actions = torch.cat([prev_batch_actions, batch_actions], dim=0)
                batch_old_logprobs = torch.cat([prev_batch_old_logprobs, batch_old_logprobs], dim=0)
                batch_edge_indices = torch.cat([prev_batch_edge_indices, batch_edge_indices], dim=0)
                batch_group_advantages = torch.cat([prev_batch_group_advantages, batch_group_advantages], dim=0)

            # 如果当前批次仍然不足两个，跳过
            if batch_states.size(0) < 2:
                continue

            # 重新计算动作概率
            action_probs, _, _ = self.policy_net(batch_states, batch_history_states, batch_edge_indices)
            dist = torch.distributions.Categorical(action_probs)
            logprobs = dist.log_prob(batch_actions)
            entropy = dist.entropy()

            # 计算 ratios
            ratios = torch.exp(logprobs - batch_old_logprobs)

            # 计算 surr1 和 surr2
            surr1 = ratios * batch_group_advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_group_advantages

            # 计算策略损失和熵损失
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy.mean()

            # 累加损失（用 .detach() 断开计算图）
            total_policy_loss += policy_loss
            total_entropy_loss += entropy_loss
            num_valid_batches += 1

        # 如果没有有效批次，直接返回
        if num_valid_batches == 0:
            return 0

        # 计算平均损失
        avg_policy_loss = total_policy_loss / num_valid_batches
        avg_entropy_loss = total_entropy_loss / num_valid_batches
        avg_loss = avg_policy_loss + self.entropy_coef * avg_entropy_loss

        # 损失裁剪（可选）
        # if self.last_loss is not None:
        #     min_loss = self.last_loss - self.max_delta
        #     max_loss = self.last_loss + self.max_delta
        #     avg_loss = torch.clamp(avg_loss, min_loss, max_loss)
        # self.last_loss = avg_loss.item()

        # 反向传播 + 梯度裁剪
        self.optimizer.zero_grad()
        avg_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新旧策略网络
        self.policy_old.load_state_dict(self.policy_net.state_dict())

        return avg_loss.item()