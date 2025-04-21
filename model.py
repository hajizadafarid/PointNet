import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        self.mlp_block1 = nn.Sequential(nn.Conv1d(3, 64, kernel_size = 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp_block2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size = 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp_block3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size = 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp_block4 = nn.Sequential(nn.Conv1d(64, 128, kernel_size = 1), nn.BatchNorm1d(128), nn.ReLU())
        self.mlp_block5 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size = 1), nn.BatchNorm1d(1024), nn.ReLU())



    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # TODO : Implement forward function.

        pointcloud = pointcloud.transpose(1, 2).to(pointcloud.device) # [B, 3, N]

        if self.input_transform:
            transform_1 = self.stn3(pointcloud) # [B, 3, 3]
            pointcloud = pointcloud.transpose(1, 2).to(pointcloud.device) # [B, N, 3]

            pointcloud = torch.matmul( pointcloud, transform_1) # [B, N, 3]

        else:
            pointcloud = pointcloud.transpose(1, 2).to(pointcloud.device)
            transform_1 = False

        feature_pointcloud = pointcloud.transpose(1, 2).to(pointcloud.device) # [B, 3, N]
        # print("feature_pointcloud.size()", feature_pointcloud.size())
        # print("pointcloud.size()", pointcloud.size())
        pointcloud = self.mlp_block2(self.mlp_block1(feature_pointcloud)) # [B, 64, N]

        if self.feature_transform:
            transform_2 = self.stn64(pointcloud) # [B, 64, 64]
            pointcloud = pointcloud.transpose(1, 2).to(pointcloud.device) # [B, N, 64]
            pointcloud = torch.matmul( pointcloud, transform_2) # [B, N, 64]
        else:
            pointcloud = pointcloud.transpose(1, 2).to(pointcloud.device)
            transform_2 = False
        pointcloud = pointcloud.transpose(1, 2).to(pointcloud.device) # [B, 64, N]
        feature_pointcloud = pointcloud.transpose(1, 2)  # [B, N, 64]
        pointcloud = self.mlp_block5(self.mlp_block4(self.mlp_block3(pointcloud))) # [B, 1024, N]


        global_feature,_ = torch.max(pointcloud, dim =2) # [B, 1024]


        return global_feature, transform_1, transform_2, feature_pointcloud


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.mlp_block6 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.mlp_block7 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.mlp_block8 = nn.Sequential(nn.Linear(256, self.num_classes))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.\

        pointcloud, transform_1, transform_2, _ = self.pointnet_feat(pointcloud)
        
        pointcloud = self.mlp_block6(pointcloud) # First fully connected layer: [B,1024] → [B,512]

        pointcloud = self.mlp_block7(pointcloud) # Second fully connected layer: [B,512] → [B,256]

        pointcloud = self.mlp_block8(pointcloud) # Final classification layer: [B,256] → [B,num_classes]

        return pointcloud, transform_1, transform_2


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        self.pointnet_feat = PointNetFeat()
        self.mlp_block1 = nn.Sequential(nn.Conv1d(1088, 512, kernel_size = 1), nn.BatchNorm1d(512), nn.ReLU())
        self.mlp_block2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size = 1), nn.BatchNorm1d(256), nn.ReLU())
        self.mlp_block3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size = 1), nn.BatchNorm1d(128), nn.ReLU())
        self.mlp_block4 = nn.Sequential(nn.Conv1d(128, 128, kernel_size = 1), nn.BatchNorm1d(128), nn.ReLU())
        self.mlp_block5 = nn.Conv1d(128, m, kernel_size = 1)
        

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        # cls_results,_,_ = self.pointnet_cls(pointcloud)
        global_feature, transform_1, transform_2, feature_pointcloud = self.pointnet_feat(pointcloud)
        _, N, _  = pointcloud.size()
        
        global_feature = torch.stack([global_feature for _ in range(N)], dim=1)

        # print("feature_pointcloud.size()", feature_pointcloud.size()) # [B, N, 64]
        # print("global_feature.size()", global_feature.size()) # [B, N, 1024]
        global_feature_big = torch.cat((feature_pointcloud, global_feature), 2) # [B, N, 1088]

        global_local_seg = self.mlp_block5(self.mlp_block4(self.mlp_block3(self.mlp_block2(self.mlp_block1(global_feature_big.transpose(1, 2))))))
        # global_local_seg = self.mlp_block1(global_feature_big.transpose(1, 2))
        global_local_seg =  global_local_seg.transpose(1, 2)  
        # print("global_local_seg.size()", global_local_seg.size())

        return global_local_seg, transform_1, transform_2, feature_pointcloud


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()
        self.N = num_points
        N = self.N
        self.mlp_block1 = nn.Sequential(nn.Linear(1024, int(N/4)), nn.BatchNorm1d(int(N/4)), nn.ReLU())
        self.mlp_block2 = nn.Sequential(nn.Linear(int(N/4), int(N/2)), nn.BatchNorm1d(int(N/2)), nn.ReLU())
        self.mlp_block3 = nn.Sequential(nn.Linear(int(N/2), N), nn.Dropout(0.3), nn.BatchNorm1d(N), nn.ReLU())
        self.mlp_block4 = nn.Linear(N, N * 3)
        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        B = pointcloud.size()[0]
        device = pointcloud.device
        pointcloud, _, _, _ = self.pointnet_feat(pointcloud)
        pointcloud = pointcloud.to(device)
        pointcloud = self.mlp_block1(pointcloud).to(pointcloud.device)
        pointcloud = self.mlp_block2(pointcloud).to(pointcloud.device)
        pointcloud = self.mlp_block3(pointcloud).to(pointcloud.device)
        pointcloud = self.mlp_block4(pointcloud).to(pointcloud.device)
        pointcloud = torch.reshape(pointcloud, (B, self.N, 3)).to(pointcloud.device)

        # TODO : Implement forward function.
        return pointcloud


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
