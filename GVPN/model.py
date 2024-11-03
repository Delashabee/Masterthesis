import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

'''from torch_geometric.data import Data

class PointNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=64):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Transpose the input to (batch_size, num_features, num_points)
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global max pooling
        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return self.sigmoid(x)'''


class FeatureExtractor3D(nn.Module):
    def __init__(self):
        super(FeatureExtractor3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32768):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, feature_dim, nhead, num_encoder_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        self.feature_extractor = FeatureExtractor3D()
        self.flatten = nn.Flatten(start_dim=2)
        self.position_encoder = PositionalEncoding(feature_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)  # (N, C, D, H, W)
        x = self.flatten(x).permute(2, 0, 1)  # (S, N, E)
        x = self.position_encoder(x)  # (S, N, E)
        x = self.transformer_encoder(x)  # (S, N, E)
        x = x.mean(dim=0)  # (N, E)
        x = self.fc(x)  # (N, num_classes)
        return x


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv(x)


class RVPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual):
        super(RVPBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual

        self.conv1 = self._make_conv()
        self.conv2 = self._make_conv()
        self.relu = nn.ReLU(inplace=False)

    def _make_conv(self):
        layers = []

        for i in range(2):
            layers += [CNNBlock(in_channels=self.in_channels, out_channels=self.out_channels)]
            self.in_channels = self.out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        identity = x1
        x2 = self.conv2(x1)

        if self.residual:
            x2 = x2 + identity

        return self.relu(x2)


class MyNBVNetV1(nn.Module):

    def __init__(self, num_classes, residual=True, dropout_prob=0.5):
        super(MyNBVNetV1, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # self.poolA = nn.AvgPool3d(kernel_size=2, stride=2)
        self.residual = residual

        self.block1 = RVPBlock(32, 64, residual=self.residual)
        self.block2 = RVPBlock(64, 128, residual=self.residual)
        self.block3 = RVPBlock(128, 256, residual=self.residual)
        self.block4 = RVPBlock(256, 512, residual=self.residual)
        # self.block5 = RVPBlock(512, 1024, residual=self.residual)
        #self.bn = nn.BatchNorm1d(32 + 64 + 128 + 256 + 512)
        self.fc = nn.Linear(32 + 64 + 128 + 256 + 512, num_classes)
        # self.fc = nn.Linear(32 + 64 + 128 + 256 + 512, 64)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.pool(self.conv1(x))
        x2 = self.pool(self.block1(x1))
        x3 = self.pool(self.block2(x2))
        x4 = self.pool(self.block3(x3))
        x5 = self.pool(self.block4(x4))

        x2 = torch.cat([x2, self.pool(x1)], dim=1)
        x3 = torch.cat([x3, self.pool(x2)], dim=1)
        x4 = torch.cat([x4, self.pool(x3)], dim=1)
        x5 = torch.cat([x5, self.pool(x4)], dim=1)

        x5 = x5.view(x5.shape[0], -1)
        #x5 = self.bn(x5)
        self.flat_feature_map = x5.clone().detach()

        x5 = self.sig(self.fc(x5))
        # x = self.sig(self.fc5(x))
        self.final_output = x5.clone().detach()

        return x5


class MyNBVNetV2(nn.Module):

    def __init__(self, num_classes, residual=True):
        super(MyNBVNetV2, self).__init__()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.residual = residual

        self.block1 = RVPBlock(64, 128, residual=self.residual)
        self.block2 = RVPBlock(128, 256, residual=self.residual)
        self.block3 = RVPBlock(256, 512, residual=self.residual)
        self.block4 = RVPBlock(512, 1024, residual=self.residual)
        # self.block5 = RVPBlock(512, 1024, residual=self.residual)

        self.fc1 = nn.Linear(64 + 128 + 256 + 512 + 1024, 1024)

        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU(inplace=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.pool(self.conv1(x))
        x2 = self.pool(self.block1(x1))
        x3 = self.pool(self.block2(x2))
        x4 = self.pool(self.block3(x3))
        x5 = self.pool(self.block4(x4))

        x2 = torch.cat([x2, self.pool(x1)], dim=1)
        x3 = torch.cat([x3, self.pool(x2)], dim=1)
        x4 = torch.cat([x4, self.pool(x3)], dim=1)
        x5 = torch.cat([x5, self.pool(x4)], dim=1)

        x5 = x5.view(x5.shape[0], -1)

        x5 = self.relu(self.fc1(x5))
        x5 = self.sig(self.fc2(x5))
        # x = self.sig(self.fc5(x))

        return x5


class MyNBVNetV3(nn.Module):

    def __init__(self, num_classes, residual=True, dropout_prob=0.5):
        super(MyNBVNetV3, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.residual = residual

        self.block1 = RVPBlock(32, 64, residual=self.residual)
        self.block2 = RVPBlock(64, 128, residual=self.residual)
        self.block3 = RVPBlock(128, 256, residual=self.residual)
        self.block4 = RVPBlock(256, 512, residual=self.residual)
        # self.block5 = RVPBlock(512, 1024, residual=self.residual)

        self.fc1 = nn.Linear(32 + 64 + 128 + 256 + 512, 512)
        self.fc1_drop = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(512, 256)
        self.fc2_drop = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(256, 128)
        self.fc3_drop = nn.Dropout(dropout_prob)

        self.fc4 = nn.Linear(128, 64)
        self.fc4_drop = nn.Dropout(dropout_prob)

        self.fc5 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU(inplace=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.pool(self.conv1(x))
        x2 = self.pool(self.block1(x1))
        x3 = self.pool(self.block2(x2))
        x4 = self.pool(self.block3(x3))
        x5 = self.pool(self.block4(x4))

        x2 = torch.cat([x2, self.pool(x1)], dim=1)
        x3 = torch.cat([x3, self.pool(x2)], dim=1)
        x4 = torch.cat([x4, self.pool(x3)], dim=1)
        x5 = torch.cat([x5, self.pool(x4)], dim=1)

        x5 = x5.view(x5.shape[0], -1)

        x5 = self.fc1_drop(self.relu(self.fc1(x5)))
        x5 = self.fc2_drop(self.relu(self.fc2(x5)))
        x5 = self.fc3_drop(self.relu(self.fc3(x5)))
        x5 = self.fc4_drop(self.relu(self.fc4(x5)))
        x5 = self.sig(self.fc5(x5))
        # x = self.sig(self.fc5(x))

        return x5


class MyNBVNetV4(nn.Module):

    def __init__(self, residual=True, dropout_prob=0.5):
        super(MyNBVNetV4, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.residual = residual

        self.block1 = RVPBlock(32, 64, residual=self.residual)
        self.block2 = RVPBlock(64, 128, residual=self.residual)
        self.block3 = RVPBlock(128, 256, residual=self.residual)
        self.block4 = RVPBlock(256, 512, residual=self.residual)
        # self.block5 = RVPBlock(512, 1024, residual=self.residual)

        self.fc1 = nn.Linear(32 + 64 + 128 + 256 + 512, 512)
        self.fc1_drop = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(512, 256)
        self.fc2_drop = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(256, 128)
        self.fc3_drop = nn.Dropout(dropout_prob)

        self.fc4 = nn.Linear(128, 64)

        self.relu = nn.LeakyReLU(inplace=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.pool(self.conv1(x))
        x2 = self.pool(self.block1(x1))
        x3 = self.pool(self.block2(x2))
        x4 = self.pool(self.block3(x3))
        x5 = self.pool(self.block4(x4))

        x2 = torch.cat([x2, self.pool(x1)], dim=1)
        x3 = torch.cat([x3, self.pool(x2)], dim=1)
        x4 = torch.cat([x4, self.pool(x3)], dim=1)
        x5 = torch.cat([x5, self.pool(x4)], dim=1)

        x5 = x5.view(x5.shape[0], -1)

        x5 = self.fc1_drop(self.relu(self.fc1(x5)))
        x5 = self.fc2_drop(self.relu(self.fc2(x5)))
        x5 = self.fc3_drop(self.relu(self.fc3(x5)))
        x5 = self.sig(self.fc4(x5))
        # x = self.sig(self.fc5(x))

        return x5


class MyNBVNetV5(nn.Module):

    def __init__(self, residual=True):
        super(MyNBVNetV5, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=4, stride=4)
        self.pool1 = nn.MaxPool3d(kernel_size=8, stride=8)
        self.residual = residual

        self.block1 = RVPBlock(32, 64, residual=self.residual)

        self.fc1 = nn.Linear(64 + 32, 64)

        self.relu = nn.ReLU(inplace=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.pool(self.conv1(x))
        x2 = self.pool1(self.block1(x1))

        x2 = torch.cat([x2, self.pool1(x1)], dim=1)

        x2 = x2.view(x.shape[0], -1)

        x = self.sig(self.fc1(x2))

        return x


class MyNBVNetV6(nn.Module):

    def __init__(self, residual=True, dropout_prob=0.5):
        super(MyNBVNetV6, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool3d(kernel_size=16, stride=16)
        self.residual = residual

        self.block1 = RVPBlock(32, 64, residual=self.residual)

        self.fc1 = nn.Linear(32 * 32 * 32 + 32 + 64, 64)
        '''self.fc1 = nn.Linear(32*32*32 + 32 + 64, 1024)
        self.fc1_drop = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(1024, 64)


        self.relu = nn.ReLU(inplace=False)'''
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.pool(self.conv1(x))
        x2 = self.pool1(self.block1(x1))

        x2 = torch.cat([x2, self.pool1(x1)], dim=1)

        '''print("x shape: ", x.shape)
        print("x1 shape: ", x1.shape)
        print("x2 shape: ", x2.shape)'''

        x2 = torch.cat([x.view(x.shape[0], -1), x2.view(x.shape[0], -1)], dim=1)
        # print("x2 after view and concat shape: ", x2.shape)

        x2 = x2.view(x2.shape[0], -1)
        # print("x2 after final view shape: ", x2.shape)
        '''x_out = self.fc1_drop(self.relu(self.fc1(x2)))
        x_out = self.sig(self.fc2(x_out))'''
        x_out = self.sig(self.fc1(x2))

        return x_out


class MyNBVNetV7(nn.Module):

    def __init__(self, residual=True, dropout_prob=0.5):
        super(MyNBVNetV7, self).__init__()
        self.residual = residual
        self.conv1 = CNNBlock(in_channels=1, out_channels=32)
        self.conv2 = CNNBlock(in_channels=32, out_channels=64)
        self.conv3 = CNNBlock(in_channels=64, out_channels=128)
        self.block = RVPBlock(128, 256, residual=self.residual)
        # self.conv5 = CNNBlock(in_channels=256, out_channels=512)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool3d(kernel_size=4, stride=4)
        self.fc1 = nn.Linear(32 + 64 + 128 + 256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.sig = nn.Sigmoid()
        self.relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x1 = self.pool(self.conv1(x))
        x2 = self.pool(self.conv2(x1))
        x3 = self.pool(self.conv3(x2))
        x4 = self.pool1(self.block(x3))
        # x5 = self.pool(self.conv5(x4))
        x_out = torch.cat([x2, self.pool(x1)], dim=1)
        x_out = torch.cat([x3, self.pool(x_out)], dim=1)
        x_out = torch.cat([x4, self.pool1(x_out)], dim=1)
        # x_out = torch.cat([x5, self.pool(x_out)], dim=1)
        x_out = x_out.view(x_out.shape[0], -1)
        x_out = self.relu(self.fc1(x_out))
        x_out = self.sig(self.fc2(x_out))

        return x_out


class VoxResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VoxResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class VoxResNet(nn.Module):
    def __init__(self, num_classes=64):
        super(VoxResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.sig = nn.Sigmoid()

        self.layer1 = VoxResBlock(64, 64)
        self.layer2 = VoxResBlock(64, 128)
        self.layer3 = VoxResBlock(128, 256)
        self.layer4 = VoxResBlock(256, 512)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.sig(self.fc(x))

        return x


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality, stride=1, downsample=None, base_width=4):
        super(ResNeXtBottleneck, self).__init__()

        D = int(out_channels * (base_width / 64.0))  # 基于 baseWidth 计算 D
        C = cardinality  # Cardinality 分组数

        # 第一层 1x1 卷积
        self.conv1 = nn.Conv3d(in_channels, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(D * C)

        # 第二层 3x3 卷积，分组卷积
        self.conv2 = nn.Conv3d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm3d(D * C)

        # 第三层 1x1 卷积
        self.conv3 = nn.Conv3d(D * C, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self, layers, cardinality=32, num_classes=1000, base_width=4):
        super(ResNeXt, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], cardinality, base_width)
        self.layer2 = self._make_layer(128, layers[1], cardinality, base_width, stride=2)
        self.layer3 = self._make_layer(256, layers[2], cardinality, base_width, stride=2)
        self.layer4 = self._make_layer(512, layers[3], cardinality, base_width, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def _make_layer(self, out_channels, blocks, cardinality, base_width, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

        layers = []
        layers.append(ResNeXtBottleneck(self.in_channels, out_channels, cardinality, stride, downsample, base_width))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNeXtBottleneck(self.in_channels, out_channels, cardinality, base_width=base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.sigmoid(x)

        return x


def visualize_features(flat_features, final_output, flat_shape=(32, 31), final_shape=(6, 6)):
    flat_features = flat_features.cpu().numpy()
    final_output = final_output.cpu().numpy()

    # 将展平的特征重塑为二维图像尺寸
    flat_reshaped = flat_features[0, :flat_shape[0] * flat_shape[1]].reshape(flat_shape)

    # 将最后一层的输出重塑为6x6的矩阵
    final_reshaped = final_output[0].reshape(final_shape)

    # 归一化
    flat_normalized = (flat_reshaped - np.min(flat_reshaped)) / (np.max(flat_reshaped) - np.min(flat_reshaped) + 1e-6)
    final_normalized = (final_reshaped - np.min(final_reshaped)) / (
                np.max(final_reshaped) - np.min(final_reshaped) + 1e-6)

    # 并排显示两个独立图像
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(flat_normalized, cmap='gray')
    axs[0].set_title('Flattened Features')
    axs[0].axis('off')

    axs[1].imshow(final_normalized, cmap='gray')
    axs[1].set_title('Final Layer Output (6x6)')
    axs[1].axis('off')

    plt.show()

def visualize_features_vertical(flat_features, final_output, label):
    flat_features = flat_features.cpu().numpy().flatten()
    final_output = final_output.cpu().numpy().flatten()
    label = label.flatten()

    # 创建左右并排排列的图像
    fig, axs = plt.subplots(1, 3, figsize=(9, 12))  # 调整每个子图的宽度

    # 可视化展平特征
    axs[0].imshow(flat_features.reshape(-1, 1), cmap='gray', aspect='auto')
    axs[0].set_title('Flattened Features (Vertical)')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_ylabel('Feature Index')

    # 可视化最终输出
    axs[1].imshow(final_output.reshape(-1, 1), cmap='gray', aspect='auto')
    axs[1].set_title('Final Layer Output (Vertical)')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_ylabel('Output Index')

    # 可视化标签
    axs[2].imshow(label.reshape(-1, 1), cmap='gray', aspect='auto')
    axs[2].set_title('Label (Vertical)')
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].set_ylabel('Label Index')

    plt.tight_layout()
    plt.show()

def eval(datapath, model_path, num_classes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_data = np.genfromtxt(datapath).reshape(1, 1, 32, 32, 32)
    test_data = torch.from_numpy(test_data).to(torch.float32)
    '''label_list = np.genfromtxt(label_path, dtype=np.int32).tolist()
    label = np.zeros(num_classes)
    label[label_list] = 1'''

    model = MyNBVNetV1(num_classes=num_classes)
    #model = ResNeXt(layers=[3, 4, 23, 3], cardinality=32, num_classes=num_classes)
    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    print('EVALUATING')
    model.eval()
    grid = test_data.to(device)

    startTime = time.time()

    output = model(grid)

    endTime = time.time()
    print('run time is ' + str(endTime - startTime))
    #print('output:', output[0])
    output[output >= 0.3] = 1
    output[output < 0.3] = 0
    '''correct1 = 0
    wrong1 = 0
    cnt1 = 0
    recall = 0
    precision = 0
    for j in range(num_classes):
        if label[j] == 1 and output[0][j] == 1:
            correct1 += 1
            cnt1 += 1
        elif label[j] == 1 and output[0][j] == 0:
            cnt1 += 1
        elif label[j] == 0 and output[0][j] == 1:
            wrong1 += 1
    recall += (correct1 / cnt1)
    precision += (correct1 / (correct1 + wrong1 + 1e-6))
    print(output.shape)
    print('recall:', recall, 'precision:', precision)'''
    #visualize_features_vertical(model.flat_feature_map, model.final_output, label)

    return output





if __name__ == "__main__":
    '''model = MyNBVNetV6().to('cuda:0')
    # block = RVPBlock(64, 128, residual=True)
    x = torch.randn(64, 1, 32, 32, 32).to('cuda:0')
    print(model(x).shape)
    # # print(model) '''
    num_classes = 40
    #name_of_models = ['012', '032', '054', '062', '066', '077', '081', '085', '089', '099']
    #name_of_models = ['']
    #for name_of_model in name_of_models:

        # label_path = 'Boneviewsid000.txt'
        #model_path = 'D:/Programfiles/Myscvp/SCVPNet/trained_model/mydata40_ResNeXt_101_Adam_MyLoss_la0_1_la1_1_e150_b32_l0.0008520272550880615_gamma0.3classes40.pth.tar'
    name_of_model = 'input_voxel'
    model_path = 'D:/Programfiles/Myscvp/SCVPNet/trained_model/mydata40_NBVNet1_Adam_pro_MyLoss_pro1.5_e50_b128_l0.0009041642643461512_gamma0.3classes40.pth.tar'

    '''pred = eval(
        f'D:/Programfiles/Myscvp/industrial_label_data/test/40_views/novel/{name_of_model}/toward{t}_rotate{r}_view{v}/grid_toward{t}_rotate{r}_view{v}.txt',
        f'D:/Programfiles/Myscvp/industrial_label_data/test/40_views/novel/{name_of_model}/toward{t}_rotate{r}_view{v}/ids_toward{t}_rotate{r}_view{v}.txt',
        model_path, num_classes)'''
    pred = eval(f'D:/Programfiles/Myscvp/industrial_label_data/test/40_views/novel_txt/{name_of_model}.txt',model_path, num_classes)
    ans = []
    for i in range(pred.shape[1]):
        if pred[0][i] == 1:
            print(i)
            ans.append(i)
    print('ans:', ans)
    np.savetxt('./log/' + name_of_model + '_NBVNet.txt', ans, fmt='%d')
    print('All tests of objects finished.')