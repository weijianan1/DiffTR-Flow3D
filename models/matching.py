import torch
import torch.nn as nn
import torch.nn.functional as F

def global_correlation_softmax_3d(feature0, feature1, xyzs1, xyzs2,):
    """
    Implements Equation 8: Motion estimation.
    D_c^e = softmax( (F_s^e * (F_t^e)^T) / sqrt(d) ) * P_t - P_s
    """
    # global correlation
    b, c, n = feature0.shape
    feature0 = feature0.permute(0, 2, 1)  # [B, N, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, N]

    # Calculate similarity scores: (F_s^e * (F_t^e)^T) / sqrt(d)
    correlation = torch.matmul(feature0, feature1).view(b, n, n) / (c ** 0.5)  # [B, N, N]

    # flow from softmax
    init_grid_1 = xyzs1.to(correlation.device) # [B, 3, N] (P_s)
    init_grid_2 = xyzs2.to(correlation.device) # [B, 3, N] (P_t)
    grid_2 = init_grid_2.permute(0, 2, 1)  # [B, N, 3]

    correlation = correlation.view(b, n, n)  # [B, N, N]

    prob = F.softmax(correlation, dim=-1)  # [B, N, N]

    # Calculate soft correspondence: softmax(...) * P_t
    correspondence = torch.matmul(prob, grid_2).view(b, n, 3).permute(0, 2, 1)  # [B, 3, N]

    # Calculate flow: D_c^e = correspondence - P_s
    flow = correspondence - init_grid_1

    return flow, prob

class SelfCorrelationSoftmax3D(nn.Module):
    """
    Implements Equation 9: Motion approximation.
    Refines flow approximation for missing particles using attention on features.
    D_{s->t} = softmax( (W_q F_s^e)(W_k F_s^e)^T / sqrt(d) ) D_c^e
    
    query: feature0 (transformed by W_q), key: feature0 (transformed by W_k), value: flow
    """

    def __init__(self, in_channels,
                 **kwargs,
                 ):
        super(SelfCorrelationSoftmax3D, self).__init__()

        # W_q and W_k learnable embedding matrices from Eq 9.
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.softmax = torch.nn.Softmax(dim=-1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow,
                **kwargs,
                ):
        # q, k: feature [B, C, N], v: flow [B, 3, N]

        b, c, n = feature0.size()

        query = feature0.permute(0, 2, 1)  # [B, N, C]

        # Apply W_q and W_k
        query = self.q_proj(query)  # [B, N, C]
        key = self.k_proj(query)  # [B, N, C]

        value = flow.view(b, flow.size(1), n).permute(0, 2, 1)  # [B, N, 2]

        # Attention mechanism
        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, N, N]
        prob = self.softmax(scores)

        out = torch.matmul(prob, value)  # [B, N, 2]
        out = out.view(b, n, value.size(-1)).permute(0, 2, 1)  # [B, 2, N]

        return out