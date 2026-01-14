import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import FeatureTransformer3D, FeatureTransformer3D_PT
from models.matching import global_correlation_softmax_3d, SelfCorrelationSoftmax3D
from models.backbone import DGCNN, PointNet, MLP

class DiffTR_Flow3D(nn.Module):
    def __init__(self,
                 backbone='DGCNN',
                 channels=128,
                 ffn_dim_expansion=4,
                 num_transformer_pt_layers=1,
                 num_transformer_layers=8,
                 self_condition = True,
                 ):
        super(DiffTR_Flow3D, self).__init__()

        self.backbone = backbone
        self.channels = channels
        self.self_condition = self_condition

        # PointNet
        if self.backbone=='PointNet':
            self.pointnet0 = PointNet(output_channels = self.channels)
            self.pointnet1 = PointNet(output_channels = self.channels)
            channels = self.channels
        # DGCNN
        if self.backbone=='DGCNN':
            # Implements the feature extractor described in "Feature Extraction".
            # Maps 3D coordinates to high-dimensional feature space.
            self.DGCNN0 = DGCNN(output_channels = self.channels, k=16)
            self.DGCNN1 = DGCNN(output_channels = self.channels, k=16)
            channels = self.channels

        self.num_transformer_layers = num_transformer_layers
        self.num_transformer_pt_layers = num_transformer_pt_layers

        # Transformer
        # Corresponds to "Global correlation" (Eq 11): Transformer-like blocks with self/cross-attention.
        if self.num_transformer_layers > 0:
            self.transformer1 = FeatureTransformer3D(num_layers=num_transformer_layers,
                                   d_model=channels,
                                    ffn_dim_expansion=ffn_dim_expansion,
                                    )
        
        # Corresponds to "Local correlation" (Eq 10): Point Transformer to enhance context descriptors.
        if self.num_transformer_pt_layers > 0:
            self.transformer_PT = FeatureTransformer3D_PT(num_layers=num_transformer_pt_layers,
                                                        d_points=channels,
                                                        ffn_dim_expansion=ffn_dim_expansion,
                                                        k=16,
                                                        )
        # self correlation with self-feature similarity
        # Corresponds to "Motion approximation" (Eq 9): Refines flow using learnable embeddings.
        self.feature_flow_attn0 = SelfCorrelationSoftmax3D(in_channels=self.channels)
        self.feature_flow_attn1 = SelfCorrelationSoftmax3D(in_channels=self.channels)


    def forward(self, pred, time, pcs, state='train', **kwargs):
        flow_preds = []

        xyzs1, xyzs2 = pcs[...,0:3,:], pcs[...,3:6,:]

        # ==========================================
        # Step 1: Flow motion estimation
        # ==========================================
        
        # 1. Feature Extraction (Eq 7)
        # "Roughly place the source particle within the vicinity of the target position"
        # Uses DGCNN to get context feature descriptors F_s^e and F_t^e.
        if self.backbone=='DGCNN':
            feature1 = self.DGCNN0(xyzs1 + pred) 
            feature2 = self.DGCNN0(xyzs2)
        if self.backbone=='PointNet':
            feature1 = self.pointnet0(xyzs1 + pred)
            feature2 = self.pointnet0(xyzs2)
            
        # 2. Motion estimation (Eq 8)
        # Computes coarse flow D_c^e via global feature similarity matrices.
        flow_feat = global_correlation_softmax_3d(feature1, feature2, xyzs1, xyzs2)[0]
        
        # 3. Motion approximation (Eq 9)
        # Generates flow approximation for missing particles ("ghost particles") 
        # using scene flow from adjacent non-occluded areas.
        pred = self.feature_flow_attn0(feature1, flow_feat)

        # ==========================================
        # Step 2: Flow motion refinement
        # ==========================================
        
        # Update input particle sets: \hat{P}_s = P_s + D_{s->t} (pred)
        # Extract context features F_s^r and F_t^r again on updated positions.
        if self.backbone=='DGCNN':
            feature1 = self.DGCNN1(xyzs1 + pred)
            feature2 = self.DGCNN1(xyzs2)
        if self.backbone=='PointNet':
            feature1 = self.pointnet1(xyzs1 + pred)
            feature2 = self.pointnet1(xyzs2)

        # 4. Local correlation (Eq 10)
        # Enhances descriptors using Point Transformer (confining soft attention to localized area).
        if self.num_transformer_pt_layers > 0:
            feature1, feature2 = self.transformer_PT(xyzs1, xyzs2, feature1, feature2)
            
        # 5. Global correlation (Eq 11)
        # Stacks self-, cross-attentions to calculate correlation features \tilde{F}_s^p.
        feature1, feature2 = self.transformer1(feature1, feature2)
        
        # Final Motion Estimation & Approximation
        # Descriptors are fed into motion estimation (Eq 8) and approximation (Eq 9) again.
        flow_feat = global_correlation_softmax_3d(feature1, feature2, xyzs1+pred, xyzs2)[0]
        pred = self.feature_flow_attn1(feature1, flow_feat) + pred
        flow_preds.append(pred)

        return flow_preds

