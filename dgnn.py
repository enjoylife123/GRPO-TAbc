import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from params import *

class FirstTransformerLayer(torch.nn.Module):
    """
    改进后的Transformer层，支持batch_first并优化了维度处理
    """
    def __init__(self, input_dim, hidden_dim, num_heads=2, num_layers=1):
        super(FirstTransformerLayer, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            batch_first=True  # 添加batch_first参数
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # 输入形状处理 (保持batch_first=True)
        if x.dim() == 2:  # [nodes_num, features_dim]
            x = x.unsqueeze(0)  # [1, nodes_num, features_dim]

        # Transformer处理 (已经是batch_first格式)
        x = self.transformer(x)

        # 输出形状处理
        if x.size(0) == 1:  # 非批量模式
            x = x.squeeze(0)  # [nodes_num, features_dim]

        # 调整输出维度
        return self.output_layer(x)


class FirstLayerNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 gin_layers=1, transformer_layers=FIRST_LAYER_TRANSFORMER_LAYERS,
                 num_heads=FIRST_LAYER_NUM_HEADS, use_parallel=True):
        super(FirstLayerNetwork, self).__init__()
        self.use_parallel = use_parallel
        self.hidden_dim = hidden_dim

        # 前置MLP层
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Transformer分支
        self.transformer_layers = nn.ModuleList([
            FirstTransformerLayer(
                hidden_dim,  # 输入维度现在都是hidden_dim
                hidden_dim,
                num_heads
            )
            for _ in range(transformer_layers)
        ])

    def forward(self, x, edge_index):
        # 保存原始形状用于最后reshape
        original_shape = x.shape

        # 处理MLP部分
        if x.dim() == 3:  # [batch, nodes, features]
            # 合并batch和nodes维度以通过MLP
            x_flat = x.reshape(-1, x.size(-1))  # [batch*nodes, features]
            mlp_out = self.mlp(x_flat)
            # 恢复原始形状 [batch, nodes, hidden_dim]
            mlp_out = mlp_out.view(original_shape[0], original_shape[1], -1)
        else:  # [nodes, features]
            mlp_out = self.mlp(x)  # [nodes, hidden_dim]

        # Transformer处理
        x_trans = mlp_out  # 使用MLP输出作为Transformer输入
        for transformer_layer in self.transformer_layers:
            x_trans = transformer_layer(x_trans)

        # 拼接MLP和Transformer输出
        combined = torch.cat([mlp_out, x_trans], dim=-1)  # [..., hidden_dim * 2]

        return combined

class SecondLayerNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 gin_layers=1, transformer_layers=SECOND_LAYER_TRANSFORMER_LAYERS,
                 num_heads=SECOND_LAYER_NUM_HEADS, use_parallel=True):  # 新增use_parallel控制并行模式
        super(SecondLayerNetwork, self).__init__()

        # Transformer分支
        self.transformer_layers = nn.ModuleList([
            FirstTransformerLayer(
                input_dim if i == 0 else hidden_dim,  # 第一层输入维度对齐
                hidden_dim,
                num_heads
            )
            for i in range(transformer_layers)
        ])

    def forward(self, x, edge_index):
        for transformer_layer in self.transformer_layers:
            output = transformer_layer(x)
        return output


# class PPOPolicyNetworkWithDualTransformer(torch.nn.Module):
#     def __init__(self, first_layer_input_dim, first_layer_hidden_dim,
#                  second_layer_input_dim, second_layer_hidden_dim, mlp_layers=2):
#         super(PPOPolicyNetworkWithDualTransformer, self).__init__()
#         self.second_layer_hidden_dim = second_layer_hidden_dim
#
#         self.first_layer_network = FirstLayerNetwork(
#             first_layer_input_dim, first_layer_hidden_dim)
#
#         self.second_layer_network = SecondLayerNetwork(second_layer_input_dim, second_layer_hidden_dim)
#
#         # 构建MLP层
#         layers = []
#         for i in range(mlp_layers):
#             in_dim = first_layer_hidden_dim + second_layer_hidden_dim if i == 0 else second_layer_hidden_dim
#             layers.extend([nn.Linear(in_dim, second_layer_hidden_dim), nn.ReLU()])
#         layers.append(nn.Linear(second_layer_hidden_dim, 1))
#         self.mlp = nn.Sequential(*layers)
#
#     def forward(self, current_state, history_state, edge_index):
#         # First layer network processing
#         # output_1 = self.mlp0(current_state).squeeze(-1)
#         first_out = self.first_layer_network(current_state, edge_index)  # shape: [first_layer_hidden_dim] or [batch_size, first_layer_hidden_dim]
#
#         # Second layer network processing
#         if history_state is not None:
#             # second_out = self.second_layer_network(history_state)  # shape: [second_layer_hidden_dim] or [batch_size, second_layer_hidden_dim]
#             second_out = self.first_layer_network(history_state, edge_index)
#         else:
#             # Create zeros matching first_out's batch (if any) but with second_layer_hidden_dim
#             if first_out.dim() == 2:  # Non-batched
#                 second_out = torch.zeros(
#                     (first_out.size(0), first_out.size(1)),
#                     device=first_out.device,
#                     dtype=first_out.dtype
#                 )
#             else:  # Batched
#                 second_out = torch.zeros(
#                     (first_out.size(0), first_out.size(1), first_out.size(2)),
#                     device=first_out.device,
#                     dtype=first_out.dtype
#                 )
#
#         # Combine features (unsqueeze if non-batched to allow concatenation)
#         if first_out.dim() == 1 and second_out.dim() == 1:
#             combined = torch.cat([first_out, second_out], dim=-1)  # shape: [first + second hidden_dims]
#         else:
#             combined = torch.cat([first_out, second_out], dim=-1)  # shape: [batch_size, first + second hidden_dims]
#
#         # combined = torch.cat([first_out], dim=-1)
#         output = self.mlp(combined).squeeze(-1)
#
#         return F.softmax(output, dim=-1), combined.detach()

class PPOPolicyNetworkWithDualTransformer(torch.nn.Module):
    def __init__(self, first_layer_input_dim, first_layer_hidden_dim,
                 second_layer_input_dim, second_layer_hidden_dim, mlp_layers=2):
        super(PPOPolicyNetworkWithDualTransformer, self).__init__()
        # self.second_layer_hidden_dim = second_layer_hidden_dim
        # self.second_layer_input_dim = second_layer_input_dim
        self.first_layer_hidden_dim = first_layer_hidden_dim

        self.first_layer_network = FirstLayerNetwork(
            first_layer_input_dim, first_layer_hidden_dim)

        # self.second_layer_network = SecondLayerNetwork(second_layer_input_dim, second_layer_hidden_dim)

        # 构建MLP层
        layers = []
        for i in range(mlp_layers):
            in_dim = first_layer_hidden_dim * 4 if i == 0 else second_layer_hidden_dim
            layers.extend([nn.Linear(in_dim, second_layer_hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(second_layer_hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, current_state, history_state, edge_index):
        # First layer network processing
        first_out = self.first_layer_network(current_state,
                                             edge_index)  # shape: [nodes_num, feature_dim] or [batch_size, nodes_num, feature_dim]

        # Second layer network processing
        if history_state is not None:
            second_out = history_state
        else:
            history_state = torch.zeros(
                (first_out.size(0), first_out.size(1)),
                device=first_out.device,
                dtype=first_out.dtype
            )
            second_out = history_state

        # 拼接特征 - 现在可以安全地假设维度一致
        combined = torch.cat([first_out, second_out],
                             dim=-1)  # shape: [nodes_num, f1+f2] 或 [batch_size, nodes_num, f1+f2]

        # 通过MLP处理每个节点的特征
        original_shape = combined.shape
        if combined.dim() == 3:
            combined = combined.view(-1, combined.size(-1))  # [batch_size * nodes_num, features]

        output = self.mlp(combined)  # [batch_size * nodes_num, 1] 或 [nodes_num, 1]

        # 恢复原始形状
        if len(original_shape) == 3:
            output = output.view(original_shape[0], original_shape[1])  # [batch_size, nodes_num]
        else:
            output = output.squeeze(-1)  # [nodes_num]

        # 应用 softmax 确保输出是有效的概率分布
        probs = F.softmax(output, dim=-1)  # 沿最后一个维度（nodes_num）计算softmax

        return probs, first_out.detach(), first_out  # 返回概率分布和combined特征