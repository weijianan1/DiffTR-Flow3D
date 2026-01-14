import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from einops import rearrange

# ... [OMITTED: single_head_full_attention, square_distance, index_points helper functions] ...

class TransformerBlock_PT(nn.Module):
    """
    Implements the core block for "Local correlation" (Eq 10).
    Adopts Point Transformer logic to confine computation to localized areas.
    """
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        # Relative position embedding delta
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # Mapping function gamma for attention vectors
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # Linear projections phi_q, phi_k, phi_v
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    def forward(self, xyz, features): # xyz: b x 3 x n, features: b x n x f

        xyz = torch.permute(xyz, (0,2,1))
        dists = square_distance(xyz, xyz) # b x n x n
        # knn_idx selects the k nearest neighbor particles
        knn_idx = torch.topk(dists, self.k, dim = -1, largest = False, sorted = False).indices
        knn_xyz = index_points(xyz, knn_idx) # b x n x k x 3
        
        pre = features # b x n x f
        x = self.fc1(features) # b x n x C
        
        # phi_q, phi_k, phi_v transformations
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        # q: b x n x C, k: b x k x C, v: b x k x C
        
        # Delta: relative position embedding
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x C
        
        # Eq 10: softmax(gamma(phi_q - phi_k + delta))
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc) # b x n x k x C
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x C
        
        # Aggregation: sum(attn * (phi_v + delta))
        res = (attn * (v+pos_enc)).sum(dim=-2)
        res = self.fc2(res) + pre # b x n x f
        return res, attn

class FeatureTransformer3D_PT(nn.Module):
    """
    Wraps TransformerBlock_PT to apply Local Correlation (Eq 10) 
    to both source (feature0) and target (feature1).
    """
    def __init__(self,
                 num_layers=1,
                 d_points=128,
                 ffn_dim_expansion=4,
                 k=16,
                 ):
        super(FeatureTransformer3D_PT, self).__init__()
        # ... [OMITTED: Initialization code] ...

    def forward(self, xyz0, xyz1, feature0, feature1,
                **kwargs,
                ):
        # ... [OMITTED: Forward logic calling the layers] ...
        return feature0, feature1

# Global-Cross Transformer
# ... [OMITTED: TransformerLayer, TransformerBlock classes] ...

class FeatureTransformer3D(nn.Module):
    """
    Implements "Global correlation" (Eq 11).
    Calculates correlated dependencies using standard transformer blocks 
    (self-attention and cross-attention).
    """
    def __init__(self,
                 num_layers=6,
                 d_model=128,
                 ffn_dim_expansion=4,
                 ):
        super(FeatureTransformer3D, self).__init__()
        # ... [OMITTED: Initialization code] ...

    def forward(self, feature0, feature1,
                **kwargs,
                ):
        # ... [OMITTED: Forward logic] ...
        return feature0, feature1