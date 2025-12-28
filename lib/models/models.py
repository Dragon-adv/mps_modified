#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.resnetcifar import *
import torchvision.models as models


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320/20*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)
        # Projector for contrastive learning
        dim = self.fc1.out_features  # 50
        self.projector = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        # 1. 低级特征阶段
        feat_low_raw = F.relu(F.max_pool2d(self.conv1(x), 2))
        feat_low_flat = feat_low_raw.view(-1, feat_low_raw.shape[1] * feat_low_raw.shape[2] * feat_low_raw.shape[3])
        # 保存未归一化版本用于原型聚合
        low_level_features_raw = feat_low_flat
        # 归一化版本用于后续处理（如果需要）
        low_level_features = F.normalize(feat_low_flat, dim=1)
        
        # 2. 高级特征阶段
        feat_high_raw = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(feat_low_raw)), 2))
        feat_high_flat = feat_high_raw.view(-1, feat_high_raw.shape[1]*feat_high_raw.shape[2]*feat_high_raw.shape[3])
        feat_high_encoded = F.relu(self.fc1(feat_high_flat))
        # 保存未归一化版本用于原型聚合
        high_level_features_raw = feat_high_encoded
        # 归一化版本用于分类器和投影器
        high_level_features = F.normalize(feat_high_encoded, dim=1)
        
        # 3. 分类与投影阶段（使用归一化版本）
        feat_for_classifier = F.dropout(high_level_features, training=self.training)
        logits = self.fc2(feat_for_classifier)
        log_probs = F.log_softmax(logits, dim=1)
        
        # 对比学习投影（使用归一化版本）
        proj_output = self.projector(high_level_features)
        projected_features = F.normalize(proj_output, dim=1)
        
        # 返回未归一化的特征用于原型聚合
        return logits, log_probs, high_level_features_raw, low_level_features_raw, projected_features


class CNNFemnist(nn.Module):
    def __init__(self, args):
        super(CNNFemnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(16820/20*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)
        # Projector for contrastive learning
        dim = self.fc1.out_features  # 50
        self.projector = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        # 1. 低级特征阶段
        feat_low_raw = F.relu(F.max_pool2d(self.conv1(x), 2))
        feat_low_flat = feat_low_raw.view(-1, feat_low_raw.shape[1] * feat_low_raw.shape[2] * feat_low_raw.shape[3])
        # 保存未归一化版本用于原型聚合
        low_level_features_raw = feat_low_flat
        # 归一化版本用于后续处理（如果需要）
        low_level_features = F.normalize(feat_low_flat, dim=1)
        
        # 2. 高级特征阶段
        feat_high_raw = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(feat_low_raw)), 2))
        feat_high_flat = feat_high_raw.view(-1, feat_high_raw.shape[1]*feat_high_raw.shape[2]*feat_high_raw.shape[3])
        feat_high_encoded = F.relu(self.fc1(feat_high_flat))
        # 保存未归一化版本用于原型聚合
        high_level_features_raw = feat_high_encoded
        # 归一化版本用于分类器和投影器
        high_level_features = F.normalize(feat_high_encoded, dim=1)
        
        # 3. 分类与投影阶段（使用归一化版本）
        feat_for_classifier = F.dropout(high_level_features, training=self.training)
        logits = self.fc2(feat_for_classifier)
        log_probs = F.log_softmax(logits, dim=1)
        
        # 对比学习投影（使用归一化版本）
        proj_output = self.projector(high_level_features)
        projected_features = F.normalize(proj_output, dim=1)
        
        # 返回未归一化的特征用于原型聚合
        return logits, log_probs, high_level_features_raw, low_level_features_raw, projected_features


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc0 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, args.num_classes)
        # Projector for contrastive learning
        dim = self.fc0.out_features  # 120 (high_level_features 来自 fc0 的输出)
        self.projector = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        # 1. 低级特征阶段
        feat_low_raw = self.pool(F.relu(self.conv1(x)))
        feat_low_raw = self.pool(F.relu(self.conv2(feat_low_raw)))
        feat_low_flat = feat_low_raw.view(-1, feat_low_raw.shape[1] * feat_low_raw.shape[2] * feat_low_raw.shape[3])
        # 保存未归一化版本用于原型聚合
        low_level_features_raw = feat_low_flat
        # 归一化版本用于后续处理（如果需要）
        low_level_features = F.normalize(feat_low_flat, dim=1)
        
        # 2. 高级特征阶段
        feat_high_flat = feat_low_raw.view(-1, 16 * 5 * 5)
        feat_high_encoded = F.relu(self.fc0(feat_high_flat))
        # 保存未归一化版本用于原型聚合
        high_level_features_raw = feat_high_encoded
        # 归一化版本用于分类器和投影器
        high_level_features = F.normalize(feat_high_encoded, dim=1)
        
        # 3. 分类与投影阶段（使用归一化版本）
        feat_for_classifier = F.relu(self.fc1(high_level_features))  # base encoder
        logits = self.fc2(feat_for_classifier)
        log_probs = F.log_softmax(logits, dim=1)
        
        # 对比学习投影（使用归一化版本）
        proj_output = self.projector(high_level_features)
        projected_features = F.normalize(proj_output, dim=1)
        
        # 返回未归一化的特征用于原型聚合
        return logits, log_probs, high_level_features_raw, low_level_features_raw, projected_features

class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        # Projector for contrastive learning
        dim = self.fc.in_features  # 7*7*32 = 1568
        self.projector = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        # 1. 低级特征阶段
        feat_low_raw = self.layer1(x)
        feat_low_flat = feat_low_raw.view(-1, feat_low_raw.shape[1] * feat_low_raw.shape[2] * feat_low_raw.shape[3])
        # 保存未归一化版本用于原型聚合
        low_level_features_raw = feat_low_flat
        # 归一化版本用于后续处理（如果需要）
        low_level_features = F.normalize(feat_low_flat, dim=1)
        
        # 2. 高级特征阶段
        feat_high_raw = self.layer2(feat_low_raw)
        feat_high_flat = feat_high_raw.view(feat_high_raw.size(0), -1)
        # 保存未归一化版本用于原型聚合
        high_level_features_raw = feat_high_flat
        # 归一化版本用于分类器和投影器
        high_level_features = F.normalize(feat_high_flat, dim=1)
        
        # 3. 分类与投影阶段（使用归一化版本）
        logits = self.fc(high_level_features)
        log_probs = F.log_softmax(logits, dim=1)
        
        # 对比学习投影（使用归一化版本）
        proj_output = self.projector(high_level_features)
        projected_features = F.normalize(proj_output, dim=1)
        
        # 返回未归一化的特征用于原型聚合
        return logits, log_probs, high_level_features_raw, low_level_features_raw, projected_features


class ResNetWithFeatures(nn.Module):
    def __init__(self, base='resnet18', num_classes=1000, pretrained=True):
        super().__init__()
        assert base in ['resnet18', 'resnet34', 'resnet50']
        if base == 'resnet18':
            net = models.resnet18(pretrained=pretrained)
        elif base == 'resnet34':
            net = models.resnet34(pretrained=pretrained)
        else:
            net = models.resnet50(pretrained=pretrained)

        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool
        )

        self.layer1 = net.layer1  # Low-level feature
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4  # High-level feature
        self.avgpool = net.avgpool
        self.fc = nn.Linear(net.fc.in_features, num_classes)
        # Projector for contrastive learning
        dim = net.fc.in_features  # 512 for ResNet18/34, 2048 for ResNet50
        self.projector = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        # 1. 低级特征阶段
        x = self.stem(x)
        feat_low_raw = self.layer1(x)  # Low-level feature
        feat_low_flat = feat_low_raw.view(-1, feat_low_raw.shape[1] * feat_low_raw.shape[2] * feat_low_raw.shape[3])
        # 保存未归一化版本用于原型聚合
        low_level_features_raw = feat_low_flat
        # 归一化版本用于后续处理（如果需要）
        low_level_features = F.normalize(feat_low_flat, dim=1)
        
        # 2. 高级特征阶段
        x = self.layer2(feat_low_raw)
        out = self.layer3(x)
        feat_high_raw = self.layer4(out)  # High-level feature (bs,512,7,7)
        pooled = self.avgpool(feat_high_raw)
        feat_high_flat = torch.flatten(pooled, 1)
        # 保存未归一化版本用于原型聚合
        high_level_features_raw = feat_high_flat
        # 归一化版本用于分类器和投影器
        high_level_features = F.normalize(feat_high_flat, dim=1)
        
        # 3. 分类与投影阶段（使用归一化版本）
        logits = self.fc(feat_high_flat)  # 注意：fc使用未归一化的feat_high_flat
        log_probs = F.log_softmax(logits, dim=1)
        
        # 对比学习投影（使用归一化版本）
        proj_output = self.projector(high_level_features)
        projected_features = F.normalize(proj_output, dim=1)
        
        # 返回未归一化的特征用于原型聚合
        return logits, log_probs, high_level_features_raw, low_level_features_raw, projected_features


class ModelCT(nn.Module):

    def __init__(self, out_dim, n_classes):
        super(ModelCT, self).__init__()
        basemodel = ResNet50_cifar10()
        self.features = nn.Sequential(*list(basemodel.children())[:-1])
        num_ftrs = basemodel.fc.in_features
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)  # out_dim=256

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)
        
        # Projector for contrastive learning
        dim = num_ftrs  # 使用 l1 的输入维度作为 projector 的输入
        self.projector = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        # 1. 低级特征阶段
        feat_low_raw = self.features(x)
        feat_low_flat = feat_low_raw.squeeze()
        # 保存未归一化版本用于原型聚合
        low_level_features_raw = feat_low_flat
        # 归一化版本用于后续处理（如果需要）
        low_level_features = F.normalize(feat_low_flat, dim=1)
        
        # 2. 高级特征阶段
        feat_high_encoded = self.l1(feat_low_flat)
        feat_high_encoded = F.relu(feat_high_encoded)
        # 保存未归一化版本用于原型聚合
        high_level_features_raw = feat_high_encoded
        # 归一化版本用于分类器和投影器
        high_level_features = F.normalize(feat_high_encoded, dim=1)
        
        # 3. 分类与投影阶段（使用归一化版本）
        feat_for_classifier = self.l2(high_level_features)
        logits = self.l3(feat_for_classifier)
        log_probs = F.log_softmax(logits, dim=1)
        
        # 对比学习投影（使用归一化版本）
        proj_output = self.projector(high_level_features)
        projected_features = F.normalize(proj_output, dim=1)
        
        # 返回未归一化的特征用于原型聚合
        return logits, log_probs, high_level_features_raw, low_level_features_raw, projected_features

class GlobalFedmps(nn.Module):
    def __init__(self, args):
        super(GlobalFedmps, self).__init__()
        self.dataset=args.dataset
        if args.dataset=='mnist' or args.dataset=='femnist':
            self.fc2 = nn.Linear(50, args.num_classes)
        elif args.dataset=='cifar10' or args.dataset=='realwaste' or args.dataset == 'flowers' or args.dataset == 'defungi':
            self.fc1 = nn.Linear(120, 84)
            self.fc2 = nn.Linear(84, args.num_classes)
        elif args.dataset=='cifar100' or args.dataset=='tinyimagenet':
            basemodel = ResNet50_cifar10()
            num_ftrs = basemodel.fc.in_features
            out_dim=256
            self.l2 = nn.Linear(num_ftrs, out_dim)  # out_dim=256
            self.l3 = nn.Linear(out_dim, args.num_classes)
        elif args.dataset=='fashion':
            self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x1):
        if self.dataset=='mnist' or self.dataset=='femnist':
            x = F.dropout(x1, training=self.training)
            y = self.fc2(x)
        elif self.dataset == 'cifar10' or self.dataset=='realwaste' or self.dataset == 'flowers' or self.dataset == 'defungi':
            x = F.relu(self.fc1(x1))  # 截止相当于base encoder
            y = self.fc2(x)
        elif self.dataset=='cifar100' or self.dataset=='tinyimagenet':
            x = self.l2(x1)
            y = self.l3(x)
        elif self.dataset == 'fashion':
            y = self.fc(x1)

        return y

