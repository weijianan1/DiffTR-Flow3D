import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from models.pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, CrossLayerLightFeatCosine as CrossLayer, FlowEmbeddingLayer, BidirectionalLayerFeatCosine
from models.pointconv_util import SceneFlowEstimatorResidual
from models.pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance, knn_point_cosine, knn_point
import time

scale = 1.0

class PointConvEncoder(nn.Module):
    def __init__(self, weightnet=8):
        super(PointConvEncoder, self).__init__()
        feat_nei = 32

        self.level0_lift = Conv1d(3, 32)
        self.level0 = PointConv(feat_nei, 32 + 3, 32, weightnet = weightnet) # out
        self.level0_1 = Conv1d(32, 64)
        
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64, weightnet = weightnet)
        self.level1_0 = Conv1d(64, 64)# out
        self.level1_1 = Conv1d(64, 128)

        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128, weightnet = weightnet)
        self.level2_0 = Conv1d(128, 128) # out
        self.level2_1 = Conv1d(128, 256)

        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256, weightnet = weightnet)
        self.level3_0 = Conv1d(256, 256) # out
        self.level3_1 = Conv1d(256, 512)

        self.level4 = PointConvD(64, feat_nei, 512 + 3, 128, weightnet = weightnet) # out

    def forward(self, xyz, color):
        feat_l0 = self.level0_lift(color)
        feat_l0 = self.level0(xyz, feat_l0)
        feat_l0_1 = self.level0_1(feat_l0)

        #l1
        pc_l1, feat_l1, fps_l1 = self.level1(xyz, feat_l0_1)
        feat_l1 = self.level1_0(feat_l1)
        feat_l1_2 = self.level1_1(feat_l1)

        #l2
        pc_l2, feat_l2, fps_l2 = self.level2(pc_l1, feat_l1_2)
        feat_l2 = self.level2_0(feat_l2)
        feat_l2_3 = self.level2_1(feat_l2)

        #l3
        pc_l3, feat_l3, fps_l3 = self.level3(pc_l2, feat_l2_3)
        feat_l3 = self.level3_0(feat_l3)
        feat_l3_4 = self.level3_1(feat_l3)

        #l4
        pc_l4, feat_l4, fps_l4 = self.level4(pc_l3, feat_l3_4)

        # return [xyz, pc_l1, pc_l2, pc_l3, pc_l4], \
        #         [feat_l0, feat_l1, feat_l2, feat_l3, feat_l4], \
        #         [fps_l1, fps_l2, fps_l3, fps_l4]
        return feat_l4


