#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
from torch.utils.data import DataLoader, Dataset, TensorDataset
from lib.conloss import *
from lib.utils import *
from lib.ntdloss import *
from lib.sfd_loss import make_pi_sample_per_class, logit_adjustment_ce, a_scl_loss

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    """
        本地训练执行器类，负责在联邦学习中管理单个客户端的训练过程。

        主要职责包括：
        1. 本地数据管理：根据分配的索引构建本地 DataLoader。
        2. 算法逻辑封装：实现了包括 FedAvg, FedProx, MOON, FedProto 以及 FedMPS 在内的多种联邦学习更新机制。
        3. 梯度计算与优化：管理本地模型在指定设备（GPU/CPU）上的反向传播、权重更新及指标计算。
        4. 结果反馈：向服务器返回更新后的模型状态字典（state_dict）及相关的算法元数据（如原型 Prototypes）。
    """
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(self.device)   # Negative Log Likelihood Loss; nn.CrossEntropyLoss = nn.LogSoftmax + nn.NLLLoss
        self.ntd_criterion = NTD_Loss(args.num_classes, args.ntd_tau, args.ntd_beta)
        self.gkd_criterion = nn.CrossEntropyLoss(reduction="mean")
        
        # ABBL: 统计当前 Client 数据集的每个类别样本数量
        # 遍历 trainloader 收集所有标签
        all_labels = []
        for _, labels in self.trainloader:
            all_labels.append(labels)
        # 拼接所有标签并统计每个类别的样本数
        if len(all_labels) > 0:
            all_labels_tensor = torch.cat(all_labels, dim=0)
            real_sample_per_class = all_labels_tensor.bincount(minlength=args.num_classes)
        else:
            real_sample_per_class = torch.zeros(args.num_classes, dtype=torch.long)
        
        # 计算平滑后的类别分布 π（用于 L_ACE 和 L_A-SCL）
        beta_pi = getattr(args, 'beta_pi', 0.5)
        self.pi_sample_per_class = self.make_pi_sample_per_class(real_sample_per_class, beta_pi)
        # 移动到设备
        self.pi_sample_per_class = self.pi_sample_per_class.to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(1 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)

        return trainloader

    def update_weights_fedavg(self, idx,model):
        # Set mode to train model
        model.train()

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            correct=0
            total=0
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                logits, log_probs, high_protos, low_protos, projected_features = model(images)
                loss = self.criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total +=labels.size(0)

                batch_loss.append(loss.item())
            epoch_loss=sum(batch_loss)/len(batch_loss)
            train_acc = correct / total
            print(' User: %d Epoch: %d  Loss: %f ||  train_acc: %f ' % ( idx, iter, epoch_loss, train_acc))

        return model.state_dict()

    def update_weights_prox(self, args, idx, model, global_round):
        '''
        Based on https://github.com/litian96/FedProx
        '''
        # Set mode to train model
        model.train()
        epoch_loss = []
        global_weight_collector = copy.deepcopy(list(model.to(args.device).parameters()))

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            correct = 0
            total = 0
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                logits, log_probs, high_protos, low_protos, projected_features = model(images)
                loss = self.criterion(log_probs, labels)

                fed_prox_reg = 0.0
                # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
                for param_index, param in enumerate(model.parameters()):
                    fed_prox_reg += ((args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total += labels.size(0)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            acc_last_epoch = correct / total
            print(' User: %d Epoch: %d  Loss: %f ||  train_acc: %f ' % (
            idx, iter, sum(batch_loss) / len(batch_loss), acc_last_epoch))
        epoch_loss = sum(epoch_loss) / len(epoch_loss)

        return model.state_dict(), epoch_loss, acc_val.item(), acc_last_epoch



    def update_weights_moon(self,args, idx, model, global_model,previous_models,global_round):
        '''
        Based on https://github.com/Xtra-Computing/MOON
        '''
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        cos = torch.nn.CosineSimilarity(dim=-1)
        criterion = nn.CrossEntropyLoss().to(args.device)
        for iter in range(self.args.train_ep):
            correct=0
            total=0
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)
                model.zero_grad()
                logits, log_probs, high_protos, low_protos, projected_features = model(images)
                protos = high_protos  # 使用 high_protos 作为原型

                loss1 = self.criterion(log_probs, labels)

                _, _, pro2_high, _, _ = global_model(images)
                pro2 = pro2_high  # 使用 high_protos 作为原型
                posi = cos(protos, pro2)
                logits = posi.reshape(-1, 1)

                for previous_model in previous_models:
                    previous_model.to(args.device)
                    _, _, pro3_high, _, _ = previous_model(images)
                    pro3 = pro3_high  # 使用 high_protos 作为原型
                    nega = cos(protos, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= args.temperature
                labels_m = torch.zeros(images.size(0)).to(args.device).long()

                loss2 = args.mu * criterion(logits, labels_m)

                loss = loss1+loss2
                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total +=labels.size(0)
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            acc_last_epoch = correct / total


        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item(),acc_last_epoch

    def update_weights_fedntd(self, args, idx, model):
        '''
        Based on https://github.com/Lee-Gihun/FedNTD
        '''
        # keep global
        dg_model = copy.deepcopy(model)
        dg_model.to(args.device)

        for params in dg_model.parameters():
            params.requires_grad = False

        # Set mode to train model
        model.train()

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            correct = 0
            total = 0
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                logits, _, _, _ = model(images)
                with torch.no_grad():
                    dg_logits, _, _, _ = dg_model(images)

                loss = self.ntd_criterion(logits, labels, dg_logits)

                loss.backward()
                optimizer.step()

                _, y_hat = logits.max(1)
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total += labels.size(0)

                batch_loss.append(loss.item())
            epoch_loss = sum(batch_loss) / len(batch_loss)
            train_acc = correct / total

            print(' User: %d Epoch: %d  Loss: %f ||  train_acc: %f ' % (idx, iter, epoch_loss, train_acc))

        return model.state_dict()

    def update_weights_gkd(self, args, idx, model, global_round, avg_teacher):
        '''
        Based on https://github.com/CGCL-codes/FedGKD
        '''
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            correct = 0
            total = 0
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                student_logits, _, _, _ = model(images)

                loss1 = self.gkd_criterion(student_logits, labels)

                if avg_teacher is None:
                    loss = loss1
                else:
                    avg_teacher = self._turn_off_grad(avg_teacher.to(args.device))
                    with torch.no_grad():
                        teacher_logits, _, _, _ = avg_teacher(images)

                    loss2 = self._divergence(args,
                                             student_logits=student_logits / args.gkd_temperature,
                                             teacher_logits=teacher_logits / args.gkd_temperature,
                                             )
                    loss2 = args.distillation_coefficient * loss2

                    loss = loss1 + loss2

                loss.backward()
                optimizer.step()

                _, y_hat = student_logits.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total += labels.size(0)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            acc_last_epoch = correct / total
            print(' User: %d Epoch: %d  Loss: %f ||  train_acc: %f ' % (
            idx, iter, sum(batch_loss) / len(batch_loss), acc_last_epoch))
        epoch_loss = sum(epoch_loss) / len(epoch_loss)

        return model.state_dict(), epoch_loss, acc_val.item(), acc_last_epoch

    def _turn_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        return model

    def _divergence(self, args, student_logits, teacher_logits):
        divergence = args.temperature * args.temperature * F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )  # forward KL
        return divergence

    def update_weights_fedproc(self, args, idx, model, global_protos, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total': [], '1': [], '2': [], '3': []}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        if global_round < 100:
            alpha = global_round / 100
        else:
            alpha = 1

        for iter in range(self.args.train_ep):
            correct = 0
            total = 0
            batch_loss = {'total': [], '1': [], '2': [], '3': []}
            agg_protos_label = {}
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                logits, log_probs, high_protos, low_protos, projected_features = model(images)
                protos = high_protos  # 使用 high_protos 作为原型
                loss1 = self.criterion(log_probs, labels)

                loss_mysupcon = MySupConLoss(temperature=0.5)
                if len(global_protos) == 0:
                    loss2 = 0 * loss1
                else:
                    global_h_input, global_h_labels = self.hcall(global_protos)
                    global_h_input = global_h_input.to(self.device)
                    global_h_labels = global_h_labels.to(self.device)
                    # 统一归一化：本地特征和全局原型都在对比学习前归一化
                    local_h_input = F.normalize(protos, dim=1)  # (bs,50)
                    local_h_labels = labels  # (bs,)
                    global_h_input = F.normalize(global_h_input, dim=1)
                    loss2 = loss_mysupcon.forward(feature_i=local_h_input, feature_j=global_h_input,
                                                  label_i=local_h_labels,
                                                  label_j=global_h_labels)
                if global_round == 0:
                    loss = loss1
                else:
                    loss = alpha * loss1 + (1 - alpha) * loss2

                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if labels[i].item() in agg_protos_label:
                        agg_protos_label[labels[i].item()].append(protos[i, :])
                    else:
                        agg_protos_label[labels[i].item()] = [protos[i, :]]

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total += labels.size(0)
                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        '| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader),
                            loss.item(),
                            acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            acc_last_epoch = correct / total

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label, acc_last_epoch


    def update_weights_fedproto(self, args, idx, global_protos,model, global_round=round):
        '''
        Based on https://github.com/yuetan031/FedProto
        '''
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}
        epoch_acc=[]

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            correct = 0
            total = 0
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                # loss1: cross-entrophy loss, loss2: proto distance loss
                model.zero_grad()
                logits, log_probs, high_protos, low_protos, projected_features = model(images)
                protos = high_protos  # 使用 high_protos 作为原型
                loss1 = self.criterion(log_probs, labels)

                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0*loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 + loss2 * args.ld
                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i,:]]

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)

                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total += labels.size(0)

                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            acc_last_epoch = correct / total
            epoch_acc.append(acc_last_epoch)

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), agg_protos_label

    def compute_mmd_loss(self, local_features, local_labels, global_rf_means, rf_model):
        """
        计算基于随机傅里叶特征 (RFF) 的最大均值差异 (MMD) 损失函数。
        
        该方法通过对齐本地和全局的 RFF 均值来实现分布对齐。对于每个类别，
        计算本地 RFF 特征的均值，然后与对应的全局 RFF 均值计算 MSE 损失
        （这在数学上等价于最小化 MMD）。
        
        参数:
            local_features: 当前 batch 的特征，形状为 (batch_size, feature_dim)
            local_labels: 当前 batch 的标签，形状为 (batch_size,)
            global_rf_means: 全局 RFF 均值字典，键为类别索引，值为对应的 RFF 均值张量
            rf_model: 对应的 RFF 映射模型（RFF 类的实例）
        
        返回:
            float: 所有有效类别 MMD 损失的平均值。如果当前 batch 没有有效类别匹配，返回 0。
        """
        # 获取当前 batch 中存在的唯一类别
        unique_labels = torch.unique(local_labels)
        
        if len(unique_labels) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 将 rf_model 移动到正确的设备
        # 注意：rf_model 的权重是固定的（buffer），不需要梯度，但我们需要计算输入特征的梯度
        rf_model = rf_model.to(self.device)
        
        # 用于存储每个类别的 MMD 损失
        mmd_losses = []
        
        # 遍历每个类别
        for label in unique_labels:
            label_item = label.item()
            
            # 检查全局统计量中是否存在该类别的 RFF 均值
            if label_item not in global_rf_means or global_rf_means[label_item] is None:
                continue
            
            # 提取该类别的本地特征
            class_mask = (local_labels == label)
            if class_mask.sum() == 0:
                continue
            
            local_class_features = local_features[class_mask]  # shape: (n_c, feature_dim)
            
            # 使用 rf_model 将特征映射到 RFF 空间
            local_rf_features = rf_model(local_class_features)  # shape: (n_c, rf_dim)
            
            # 计算本地 RFF 特征的均值
            local_rf_mean = local_rf_features.mean(dim=0)  # shape: (rf_dim,)
            
            # 获取对应的全局 RFF 均值，确保在正确的 device 上
            global_rf_mean = global_rf_means[label_item].to(self.device)  # shape: (rf_dim,)
            
            # 计算两者之间的 MSE 损失（这在数学上等价于最小化 MMD）
            mmd_loss = torch.nn.functional.mse_loss(local_rf_mean, global_rf_mean)
            mmd_losses.append(mmd_loss)
        
        # 返回所有有效类别 MMD 损失的平均值
        if len(mmd_losses) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            return torch.stack(mmd_losses).mean()

    def update_weights_fedmps(self, args, idx, global_high_protos, global_low_protos, global_logits, model, global_round=round, total_rounds=None, rf_models=None, global_stats=None):
        """
                执行 FedMPS 算法的本地更新过程（集成 ABBL）。

                该函数通过结合多级原型对比学习（Multi-Level Prototype-Based Contrastive Learning）、
                软标签生成（Soft Label Generation）以及 ABBL (Adaptive Bi-Branch Learning) 损失函数
                来应对联邦学习中的数据异构性和长尾分布问题。

                参数:
                    args: 全局参数配置对象，包含 alph, beta, gama, a_ce_gamma, scl_weight_start, scl_weight_end, scl_temperature 等损失权重系数。
                    idx: 当前客户端的索引 ID。
                    global_high_protos: 从服务器获取的全局高级特征原型。
                    global_low_protos: 从服务器获取的全局低级特征原型。
                    global_logits: 由全局模型生成的类别软标签预测，用于知识蒸馏。
                    model: 需要训练的本地模型（包含 projector）。
                    global_round: 当前的全局通信轮次。
                    total_rounds: 总通信轮次数，用于计算余弦退火的 scl_weight。如果为 None，则使用 args.rounds。
                    rf_models: 字典，包含不同层级对应的 RFF 模型实例，例如 
                               {'high': rff_high, 'low': rff_low}。如果为 None，则不计算 MMD 损失。
                    global_stats: 字典，包含全局统计量，结构为：
                                  {'high': {'class_rf_means': [...]}, 'low': {'class_rf_means': [...]}}。
                                  如果为 None，则不计算 MMD 损失。

                损失函数组成（ABBL 集成版本）:
                    loss_ace (L_ACE): 自适应交叉熵损失，使用 logit_adjustment_ce 计算。
                    loss_scl (L_A-SCL): 自适应监督对比学习损失，使用 a_scl_loss 计算。
                    loss_proto_high: 本地高层特征与全局高层原型之间的对比学习损失，用于对齐高级语义。
                    loss_proto_low: 本地底层特征与全局底层原型之间的对比学习损失，用于对齐基础表征。
                    loss_soft: 本地预测概率与全局模型预测（软标签）之间的 KL 散度，用于引入全局知识。

                总损失公式（ABBL 集成版本）:
                    loss = loss_ace + scl_weight * loss_scl + args.alph * loss_proto_high + args.beta * loss_proto_low + args.gama * loss_soft
                    其中 scl_weight 通过余弦退火从 args.scl_weight_start 降到 args.scl_weight_end。

                返回:
                    model.state_dict(): 训练后的模型权重参数。
                    epoch_loss: 包含各类损失分量的字典（total, ace, scl, proto_high, proto_low, soft）。
                    acc_val.item(): 最后一个 batch 的准确率。
                    agg_high_protos_label: 收集到的本地高层原型字典，按标签分类。
                    agg_low_protos_label: 收集到的本地底层原型字典，按标签分类。
                    acc_last_epoch: 整个训练 epoch 的平均准确率。
        """
        # Set mode to train model
        model.train()
        epoch_loss = {'total': [], 'ace': [], 'scl': [], 'proto_high': [], 'proto_low': [], 'soft': []}

        # Set optimizer for the local updates (确保包含 projector 参数)
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        # ABBL: 计算动态 scl_weight（余弦退火）
        if total_rounds is None:
            total_rounds = getattr(args, 'rounds', 100)
        scl_weight_start = getattr(args, 'scl_weight_start', 1.0)
        scl_weight_end = getattr(args, 'scl_weight_end', 0.0)
        # 余弦退火公式: 0.5 * (start - end) * (1 + cos(pi * current / total)) + end
        if total_rounds > 0:
            scl_weight = 0.5 * (scl_weight_start - scl_weight_end) * (1 + math.cos(math.pi * global_round / total_rounds)) + scl_weight_end
        else:
            scl_weight = scl_weight_start

        for iter in range(self.args.train_ep):
            correct = 0
            total = 0
            batch_loss = {'total': [], 'ace': [], 'scl': [], 'proto_high': [], 'proto_low': [], 'soft': []}
            agg_high_protos_label = {}
            agg_low_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                model.zero_grad()
                # ABBL: 获取 5 个返回值（包括 projected_features）
                logits, log_probs, high_protos, low_protos, projected_features = model(images)

                # Loss 1 (ABBL): L_ACE - 自适应交叉熵损失
                # 关键：传入原始 logits，而非 log_probs
                a_ce_gamma = getattr(args, 'a_ce_gamma', 0.1)
                loss_ace = self.logit_adjustment_ce(
                    logits=logits,  # 原始 logits
                    target=labels,
                    sample_per_class=self.pi_sample_per_class,
                    gamma=a_ce_gamma
                )

                # Loss 2 (ABBL): L_A-SCL - 自适应监督对比学习损失（使用修正后的方法）
                loss_scl = self.compute_adaptive_supervised_contrastive_loss(
                    projected_features=projected_features,
                    labels=labels,
                    n_k=self.pi_sample_per_class,
                    args=args
                )

                # Loss 3 (FedMPS): high-level contrastive learning loss between local features and global prototypes
                # Loss 4 (FedMPS): low-level contrastive learning loss between local features and global prototypes
                loss_mysupcon = MySupConLoss(temperature=0.5)
                if len(global_high_protos) == 0:
                    loss_proto_high = 0 * loss_ace
                    loss_proto_low = 0 * loss_ace
                else:
                    global_h_input, global_h_labels = self.hcfit(global_high_protos, high_protos, labels)
                    global_l_input, global_l_labels = self.hcfit(global_low_protos, low_protos, labels)
                    
                    # 统一归一化：本地特征和全局原型都在对比学习前归一化
                    # 这样保证对比学习时输入归一化状态一致
                    local_h_input = F.normalize(high_protos, dim=1)
                    local_h_labels = labels
                    local_l_input = F.normalize(low_protos, dim=1)
                    local_l_labels = labels
                    
                    # 全局原型也需要归一化（因为它们是未归一化特征的均值）
                    global_h_input = F.normalize(global_h_input, dim=1)
                    global_l_input = F.normalize(global_l_input, dim=1)

                    loss_proto_low = loss_mysupcon.forward(feature_i=local_l_input, feature_j=global_l_input,
                                                          label_i=local_l_labels, label_j=global_l_labels)
                    loss_proto_high = loss_mysupcon.forward(feature_i=local_h_input, feature_j=global_h_input,
                                                           label_i=local_h_labels, label_j=global_h_labels)

                # Loss 5 (FedMPS): distillation loss between local soft labels and global soft labels
                soft_loss = nn.KLDivLoss(reduction="batchmean")
                T = args.T
                if len(global_logits) == 0:
                    loss_soft = 0 * loss_ace
                else:
                    # 从全局 logits 字典中获取对应类别的 logits
                    global_logits_tensor = []
                    for l in labels:
                        class_logits = global_logits[l.item()]  # 获取该类别的全局 logits
                        global_logits_tensor.append(class_logits)
                    global_logits_tensor = torch.stack(global_logits_tensor)
                    # 使用 logits 计算 softmax（而非 probs）
                    loss_soft = soft_loss(F.log_softmax(logits / T, dim=1), F.softmax(global_logits_tensor / T, dim=1))

                # ABBL 集成版本的总损失公式
                loss = loss_ace + scl_weight * loss_scl + args.alph * loss_proto_high + args.beta * loss_proto_low + args.gama * loss_soft


                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if labels[i].item() in agg_high_protos_label:
                        agg_high_protos_label[labels[i].item()].append(high_protos[i, :])
                        agg_low_protos_label[labels[i].item()].append(low_protos[i, :])
                    else:
                        agg_high_protos_label[labels[i].item()] = [high_protos[i, :]]
                        agg_low_protos_label[labels[i].item()] = [low_protos[i, :]]

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                correct += torch.eq(y_hat, labels.squeeze()).int().sum().item()
                total += labels.size(0)
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader),
                            loss.item(),
                            acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['ace'].append(loss_ace.item())
                batch_loss['scl'].append(loss_scl.item())
                batch_loss['proto_high'].append(loss_proto_high.item())
                batch_loss['proto_low'].append(loss_proto_low.item())
                batch_loss['soft'].append(loss_soft.item())
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['ace'].append(sum(batch_loss['ace']) / len(batch_loss['ace']))
            epoch_loss['scl'].append(sum(batch_loss['scl']) / len(batch_loss['scl']))
            epoch_loss['proto_high'].append(sum(batch_loss['proto_high']) / len(batch_loss['proto_high']))
            epoch_loss['proto_low'].append(sum(batch_loss['proto_low']) / len(batch_loss['proto_low']))
            epoch_loss['soft'].append(sum(batch_loss['soft']) / len(batch_loss['soft']))
            acc_last_epoch = correct / total

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['ace'] = sum(epoch_loss['ace']) / len(epoch_loss['ace'])
        epoch_loss['scl'] = sum(epoch_loss['scl']) / len(epoch_loss['scl'])
        epoch_loss['proto_high'] = sum(epoch_loss['proto_high']) / len(epoch_loss['proto_high'])
        epoch_loss['proto_low'] = sum(epoch_loss['proto_low']) / len(epoch_loss['proto_low'])
        epoch_loss['soft'] = sum(epoch_loss['soft']) / len(epoch_loss['soft'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_high_protos_label, agg_low_protos_label, acc_last_epoch


    def hcall(self,global_protos):
        global_labels = []
        global_labels.extend(global_protos)
        global_labels = torch.tensor(global_labels)
        global_input = torch.ones((global_labels.shape[0], global_protos.get(next(iter(global_protos)))[0].shape[-1]))
        i = 0
        for label in global_labels:
            global_input[i, :] = global_protos[label.item()][0].data
            i += 1
        return global_input, global_labels

    def hcfit(self,global_protos, local_input, local_labels):# Align global_input with local feature labels
        global_input = copy.deepcopy(local_input.data)
        i = 0
        for label in local_labels:
            if label.item() in global_protos.keys():
                global_input[i, :] = global_protos[label.item()][0].data
            i += 1
        global_labels = local_labels
        return global_input, global_labels

    def hcfitnorepeat(self,global_protos, local_input, local_labels):
        labels = torch.unique(local_labels)
        global_input = torch.zeros((len(labels),local_input.size(-1)))
        i = 0
        for label in labels:
            if label.item() in global_protos.keys():
                global_input[i, :] = global_protos[label.item()][0].data
            i += 1
        global_labels = labels
        return global_input, global_labels


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

    @staticmethod
    def make_pi_sample_per_class(real_sample_per_class, beta_pi):
        """
        计算平滑后的类别分布 π，用于解决 Non-IID 本地类缺失问题。
        
        参数:
            real_sample_per_class: torch.Tensor, shape (num_classes,)
                每个类别的真实样本数量。对于缺失的类别，该值为 0。
            beta_pi: float
                平滑系数，用于设置缺失类别的虚拟样本数。通常取值为 0.1 到 1.0 之间。
        
        返回:
            pi_sample_per_class: torch.Tensor, shape (num_classes,)
                平滑处理后的类别分布，所有类别（包括缺失类别）都有非零的样本数。
        """
        class_size_min = real_sample_per_class[real_sample_per_class > 0].min()
        pi_sample_per_class = real_sample_per_class.clone()
        pi_sample_per_class[pi_sample_per_class == 0] = class_size_min * beta_pi
        return pi_sample_per_class

    @staticmethod
    def logit_adjustment_ce(logits, target, sample_per_class, gamma):
        """
        实现 L_ACE (Adaptive Cross-Entropy) 损失函数。
        
        **重要提示**：此函数的第一个参数 `logits` 必须是**原始 Logits**（未经过 Softmax/LogSoftmax），
        函数内部会执行 `logits + gamma * log(pi)` 然后再做 CrossEntropy。
        
        参数:
            logits: torch.Tensor, shape (batch_size, num_classes)
                **原始 logits**（未经过 Softmax/LogSoftmax），直接从分类器输出获得。
            target: torch.Tensor, shape (batch_size,)
                真实标签，每个元素是类别索引。
            sample_per_class: torch.Tensor, shape (num_classes,)
                每个类别的样本数（通常使用 make_pi_sample_per_class 处理后的平滑分布 π）。
            gamma: float
                调整系数，控制类别不平衡调整的强度。通常取值为 0.1 到 1.0 之间。
        
        返回:
            loss: torch.Tensor
                计算得到的 L_ACE 损失值。
        """
        import torch
        import torch.nn.functional as F
        # 将 sample_per_class 扩展为与 logits 相同的形状，以便进行广播运算
        sample_per_class = (sample_per_class
            .type_as(logits)
            .unsqueeze(dim=0)
            .expand(logits.shape[0], -1))
        
        # 对 logits 进行自适应调整：logits + gamma * log(π)
        adjusted_logits = logits + gamma * torch.log(sample_per_class + 1e-12)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(adjusted_logits, target, reduction='mean')
        return loss

    def compute_adaptive_supervised_contrastive_loss(self, projected_features, labels, n_k, args):
        """
        实现修正后的 L_A-SCL (Adaptive Supervised Contrastive Loss) 损失函数。
        
        该损失函数是监督对比学习（Supervised Contrastive Learning）的自适应版本。
        通过基于类别频率调整负样本对的相似度矩阵，增强对尾部类别的特征学习。
        
        **关键修正**：
        1. 维度修正：delta_negative 形状为 (1, B)，用于调整负样本（列）的相似度
        2. 符号修正：使用加法 (+) 增加头部类负样本的相似度，迫使 Loss 变大
        
        参数:
            projected_features: torch.Tensor, shape (batch_size, feature_dim)
                投影后的特征向量（已在模型 forward 中归一化），通常来自 projector 的输出。
                注意：该函数不再进行归一化，直接使用已归一化的特征。
            labels: torch.Tensor, shape (batch_size,)
                样本对应的标签，每个元素是类别索引。
            n_k: torch.Tensor, shape (num_classes,)
                每个类别的样本数或频率比值（通常使用 make_pi_sample_per_class 处理后的平滑分布 π）。
            args: 参数对象
                包含 scl_temperature 等超参数。
        
        返回:
            loss: torch.Tensor
                计算得到的 L_A-SCL 损失值（标量）。
        """
        import torch
        import torch.nn.functional as F
        
        # 获取温度参数
        temperature = getattr(args, 'scl_temperature', 0.07)
        
        # 1. projected_features 已经在模型中被归一化，直接使用
        z = projected_features  # shape: (batch_size, feature_dim)
        
        # 2. 计算相似度矩阵 (z_i · z_j / temperature)
        similarity_matrix = (z @ z.T) / temperature  # shape: (batch_size, batch_size)
        
        # 3. 获取正样本对掩码矩阵（相同标签的位置为 1）
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # shape: (batch_size, batch_size)
        # positive_mask = labels_equal.float()  # 正样本掩码
        # 确保对比学习只拉近不同的同类样本，排除自身对比
        positive_mask = labels_equal.float() * (1 - torch.eye(z.shape[0], device=z.device))

        # 4. 计算 Delta (基于类别频率)
        # delta_k 是每个类别的频率比值或 log 计数
        # 这里使用 log(n_k) 作为调整项
        delta_k = torch.log(n_k.float() + 1e-12)  # shape: (num_classes,)
        
        # 【关键修正 1：维度】
        # 我们要调整的是"负样本"(列) 的相似度。
        # labels 形状 (B,) -> delta_negative 形状应为 (1, B)
        delta_negative = delta_k[labels].unsqueeze(0)  # shape: (1, batch_size)
        
        # 5. 生成负样本掩码（不同标签的位置为 1）
        negative_mask = 1 - labels_equal.float()  # shape: (batch_size, batch_size)
        
        # 【关键修正 2：符号】
        # 我们要增加头部类负样本的相似度（增加区分难度），迫使 Loss 变大。
        # 使用加法 (+)
        adjusted_similarity = similarity_matrix + negative_mask * delta_negative * 1.0
        
        # 6. 数值稳定性处理：减去最大值以防止数值溢出
        logits_max = torch.max(adjusted_similarity, dim=1, keepdim=True).values
        logits = adjusted_similarity - logits_max.detach()
        
        # 7. 排除自比较（对角线元素），计算 softmax 分母
        eye_mask = 1 - torch.eye(z.shape[0], device=z.device)  # 排除对角线
        exp_logits = torch.exp(logits) * eye_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
        # 8. 计算每个样本的正样本对平均对数概率
        mask_sum = positive_mask.sum(dim=1)  # 每个样本的正样本数量
        # 处理没有正样本的情况（避免除零）
        mask_sum = torch.clamp(mask_sum, min=1.0)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / mask_sum
        
        # 9. 返回最终的监督对比学习损失（取负号并求平均）
        loss = -mean_log_prob_pos.mean()
        return loss

    def get_local_statistics(self, model, rf_models, args, stats_level='high'):
        """
        计算本地统计量，用于 SFD 算法的统计量聚合阶段。

        该方法参考了 SFD/src/flexp/sfd/stat_agg.py 中的 compute_local_stats 函数。
        通过冻结的模型提取 high-level 和 low-level 特征，并使用对应的 RFF 模型计算随机特征，
        然后分别计算每个类别的均值、外积和随机特征均值等统计量。

        参数:
            model: 完整的本地模型（如 FedMPS 模型），其 forward 返回包含 
                   (logits, log_probs, high_features, low_features) 的元组
            rf_models: 字典，包含不同层级对应的 RFF 模型实例，例如 
                       {'high': rff_high, 'low': rff_low}
            args: 全局参数对象，包含 num_classes 等信息
            stats_level: 统计量计算层级选择，可选值：
                        - 'high': 仅计算 high-level 统计量（默认）
                        - 'low': 仅计算 low-level 统计量
                        - 'both': 计算 high-level 和 low-level 统计量

        返回:
            dict: 包含以下结构的字典
                - 'high': 包含 high-level 统计量的字典（如果 stats_level 为 'high' 或 'both'）
                    - 'class_means': 每个类别的特征均值列表
                    - 'class_outers': 每个类别的特征外积均值列表
                    - 'class_rf_means': 每个类别的随机特征均值列表
                - 'low': 包含 low-level 统计量的字典（如果 stats_level 为 'low' 或 'both'，结构同上）
                - 'sample_per_class': 每个类别的样本数（两个层级共享）
        """
        # 确定需要计算的层级
        compute_high = (stats_level == 'high' or stats_level == 'both')
        compute_low = (stats_level == 'low' or stats_level == 'both')
        
        # 准备工作：将 model 和 rf_models 设置为 eval 模式并移动到设备
        model.eval()
        model = model.to(self.device)
        
        # 将需要计算的层级的 RFF 模型设置为 eval 模式并移动到设备
        levels_to_compute = []
        if compute_high:
            levels_to_compute.append('high')
        if compute_low:
            levels_to_compute.append('low')
        
        for level in levels_to_compute:
            if level in rf_models:
                rf_models[level].eval()
                rf_models[level] = rf_models[level].to(self.device)

        # 初始化列表来存储不同层级的特征、标签和随机特征
        zs_high = []  # high-level 特征列表
        zs_low = []   # low-level 特征列表
        ys = []       # 标签列表
        rfs_high = [] # high-level 随机特征列表
        rfs_low = []  # low-level 随机特征列表

        # 特征提取循环
        with torch.no_grad():  # 重要：避免计算梯度
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 调用模型获取输出
                # 根据 FedMPS 模型结构：return logits, log_probs, high_protos, low_protos
                output = model(images)
                
                # 提取 high-level 和 low-level 特征
                # 索引 2 对应 high_protos，索引 3 对应 low_protos
                if isinstance(output, tuple) and len(output) >= 4:
                    high_features = output[2]  # high-level features
                    low_features = output[3]   # low-level features
                else:
                    raise ValueError(f"模型输出格式不符合预期，期望包含至少4个元素的元组，实际得到: {type(output)}")

                # 确保标签是一维的
                labels_1d = labels.squeeze() if labels.dim() > 1 else labels
                ys.append(labels_1d.cpu())

                # 根据 stats_level 选择性地计算和存储特征
                # 注意：RFF模型期望归一化的输入（如果之前是在归一化特征上训练的）
                # 为了保持一致性，对特征进行归一化后再计算RFF
                if compute_high:
                    high_features_norm = F.normalize(high_features, dim=1)
                    rf_high = rf_models['high'](high_features_norm)
                    zs_high.append(high_features_norm.cpu())  # 存储归一化版本用于统计量计算
                    rfs_high.append(rf_high.cpu())
                
                if compute_low:
                    low_features_norm = F.normalize(low_features, dim=1)
                    rf_low = rf_models['low'](low_features_norm)
                    zs_low.append(low_features_norm.cpu())  # 存储归一化版本用于统计量计算
                    rfs_low.append(rf_low.cpu())

        # 将列表拼接成大张量
        y = torch.cat(ys, dim=0)  # shape: (total_samples,)
        
        if compute_high:
            z_high = torch.cat(zs_high, dim=0)  # shape: (total_samples, high_feature_dim)
            rf_high = torch.cat(rfs_high, dim=0)  # shape: (total_samples, high_rf_dim)
        
        if compute_low:
            z_low = torch.cat(zs_low, dim=0)    # shape: (total_samples, low_feature_dim)
            rf_low = torch.cat(rfs_low, dim=0)    # shape: (total_samples, low_rf_dim)

        # 计算每个类别的样本数（两个层级共享）
        num_classes = args.num_classes
        sample_per_class = y.bincount(minlength=num_classes)

        # 定义辅助函数：计算单个层级的统计量
        def compute_level_stats(z, rf, level_name):
            """
            计算单个层级的统计量
            
            参数:
                z: 特征张量，shape (total_samples, feature_dim)
                rf: 随机特征张量，shape (total_samples, rf_dim)
                level_name: 层级名称（用于错误提示）
            
            返回:
                dict: 包含 class_means, class_outers, class_rf_means 的字典
            """
            class_means = []
            class_outers = []
            class_rf_means = []
            
            # 获取特征维度（从实际特征中获取，不假设）
            feature_dim = z.shape[1]
            rf_dim = rf.shape[1]
            
            # 遍历所有类别
            for c in range(num_classes):
                n_c = sample_per_class[c].item()
                
                if n_c == 0:
                    # 如果该类别样本数为 0，则对应的均值和外积设为 0 张量
                    class_mean = torch.zeros(feature_dim)
                    class_outer = torch.zeros(feature_dim, feature_dim)
                    class_rf_mean = torch.zeros(rf_dim)
                else:
                    # 筛选出属于类别 c 的特征和随机特征
                    class_indices = (y == c)
                    z_c = z[class_indices]  # shape: (n_c, feature_dim)
                    rf_c = rf[class_indices]  # shape: (n_c, rf_dim)
                    
                    # 类均值 (mean)
                    mu_c = z_c.mean(dim=0)  # shape: (feature_dim,)
                    
                    # 类外积 (outer product): 将特征转为 float64 精度计算
                    z_c_f64 = z_c.to(torch.float64)
                    # outer = (z_c.T @ z_c) / n_c
                    outer = torch.matmul(z_c_f64.t(), z_c_f64) / n_c  # shape: (feature_dim, feature_dim)
                    # 转回原始精度
                    outer = outer.to(z_c.dtype)
                    
                    # 类 RFF 均值
                    rf_mean_c = rf_c.mean(dim=0)  # shape: (rf_dim,)
                    
                    class_mean = mu_c
                    class_outer = outer
                    class_rf_mean = rf_mean_c
                
                # 将统计量存入列表
                class_means.append(class_mean)
                class_outers.append(class_outer)
                class_rf_means.append(class_rf_mean)
            
            return {
                'class_means': class_means,
                'class_outers': class_outers,
                'class_rf_means': class_rf_means
            }
        
        # 根据 stats_level 选择性地计算统计量
        result = {
            'sample_per_class': sample_per_class
        }
        
        if compute_high:
            high_stats = compute_level_stats(z_high, rf_high, 'high')
            result['high'] = high_stats
        
        if compute_low:
            low_stats = compute_level_stats(z_low, rf_low, 'low')
            result['low'] = low_stats
        
        # 将 model 和 rf_models 移回 CPU 以释放显存
        model.cpu()
        for level in levels_to_compute:
            if level in rf_models:
                rf_models[level].cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 返回按层级组织的统计量字典
        return result


def test_inference(args, model, test_dataset,user_groups_gt,idx):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        logits, log_probs, high_protos, low_protos, projected_features = model(images)
        outputs = log_probs  # 使用 log_probs 作为输出
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss= loss/len(testloader)
    return accuracy, loss

def test_inference_new(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            logits, log_probs, high_protos, low_protos, projected_features = model(images)
            probs = log_probs  # 使用 log_probs 作为概率
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        cnt = np.zeros(10)
        for i in range(10):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i]+=1
        for i in range(10):
            if cnt[i]!=0:
                outputs[:, i] = outputs[:,i]/cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)


    acc = correct/total

    return loss, acc



def test_inference_new_het(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        protos_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            logits, log_probs, high_protos, low_protos, projected_features = model(images)
            protos = high_protos  # 使用 high_protos 作为原型
            protos_list.append(protos)

        ensem_proto = torch.zeros(size=(images.shape[0], protos.shape[1])).to(device)
        # protos ensemble
        for protos in protos_list:
            ensem_proto += protos
        ensem_proto /= len(protos_list)
        
        # 归一化后再计算距离，保证距离计算的一致性
        ensem_proto_norm = F.normalize(ensem_proto, dim=1)

        a_large_num = 100
        outputs = a_large_num * torch.ones(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(10):
                if j in global_protos.keys():
                    global_proto_norm = F.normalize(global_protos[j][0].unsqueeze(0), dim=1).squeeze(0)
                    dist = loss_mse(ensem_proto_norm[i,:], global_proto_norm)
                    outputs[i,j] = dist

        # Prediction
        _, pred_labels = torch.min(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    acc = correct/total

    return acc

def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):
    """ Returns the test accuracy and loss.
    """

    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []#w
    acc_list_l = []#wo
    loss_list_l=[]#wo
    loss_list = []#w
    loss_return_list=[]#w
    for idx in range(args.num_users):
        correct_wo,total_wo,loss_wo=0.0,0.0,0.0
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        # test (local model)
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            logits, log_probs, high_protos, low_protos, projected_features = model(images)
            outputs = log_probs  # 使用 log_probs 作为输出

            batch_loss = criterion(outputs, labels)
            loss_wo += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct_wo += torch.sum(torch.eq(pred_labels, labels)).item()
            total_wo += len(labels)

        acc_wo = correct_wo / total_wo
        loss_re=loss_wo/len(testloader)
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc_wo))
        acc_list_l.append(acc_wo)
        loss_list_l.append(loss_re)

        # test (use global proto)
        correct_w, total_w = 0.0, 0.0
        if global_protos!=[]:
            loss_return=[]
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                logits, log_probs, high_protos, low_protos, projected_features = model(images)
                outputs = log_probs  # 使用 log_probs 作为输出
                protos = high_protos  # 使用 high_protos 作为原型

                # compute the dist between protos and global_protos
                # 归一化后再计算距离，保证距离计算的一致性
                protos_norm = F.normalize(protos, dim=1)
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            global_proto_norm = F.normalize(global_protos[j][0].unsqueeze(0), dim=1).squeeze(0)
                            d = loss_mse(protos_norm[i, :], global_proto_norm)
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct_w += torch.sum(torch.eq(pred_labels, labels)).item()
                total_w += len(labels)

                # compute loss
                # 归一化后再计算MSE损失，保证一致性
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                proto_new_norm = F.normalize(proto_new, dim=1)
                protos_norm_for_loss = F.normalize(protos, dim=1)
                loss2 = loss_mse(proto_new_norm, protos_norm_for_loss)
                # loss1 = loss_function(probs, labels)
                loss1 = criterion(outputs, labels)
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                else:
                    loss2 = loss2.detach().numpy()
                loss_return.append((loss1+loss2*args.ld).item())
            acc_w = correct_w / total_w
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc_w))
            acc_list_g.append(acc_w)
            loss_list.append(loss2)
            loss_return_list.append(sum(loss_return)/len(loss_return))

    return acc_list_l,loss_list_l, acc_list_g, loss_list,loss_return_list

def test_inference_fedproto(args,logger, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []#w
    acc_list_l = []#wo
    loss_list = []
    for idx in range(args.num_users):
        correct_wo,total_wo=0,0
        batch_eval_loss = {'total': [], '1': [], '2': [], '3': []}
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        # test (local model)
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            logits, log_probs, high_protos, low_protos, projected_features = model(images)
            outputs = log_probs  # 使用 log_probs 作为输出
            protos = high_protos  # 使用 high_protos 作为原型

            batch_loss1 = criterion(outputs, labels)

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct_wo += torch.sum(torch.eq(pred_labels, labels)).item()
            total_wo += len(labels)
            batch_eval_loss['1'].append(batch_loss1.item())

        acc_wo = correct_wo / total_wo
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc_wo))
        logger.info('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc_wo))
        acc_list_l.append(acc_wo)

        # test (use global proto)
        correct_w, total_w = 0, 0
        if global_protos!=[]:
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                logits, log_probs, high_protos, low_protos, projected_features = model(images)
                outputs = log_probs  # 使用 log_probs 作为输出
                protos = high_protos  # 使用 high_protos 作为原型

                # compute the dist between protos and global_protos
                # 归一化后再计算距离，保证距离计算的一致性
                protos_norm = F.normalize(protos, dim=1)
                a_large_num = 10000
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            global_proto_norm = F.normalize(global_protos[j][0].unsqueeze(0), dim=1).squeeze(0)
                            d = loss_mse(protos_norm[i, :], global_proto_norm)
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct_w += torch.sum(torch.eq(pred_labels, labels)).item()
                total_w += len(labels)

                # compute loss
                # 归一化后再计算MSE损失，保证一致性
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                proto_new_norm = F.normalize(proto_new, dim=1)
                protos_norm_for_loss = F.normalize(protos, dim=1)
                batch_loss2 = loss_mse(proto_new_norm, protos_norm_for_loss)
                batch_eval_loss['2'].append(batch_loss2.item())
                batch_loss=batch_loss1+batch_loss2
                batch_eval_loss['total'].append(batch_loss.item())

            acc_w = correct_w / total_w
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc_w))
            logger.info('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc_w))
            acc_list_g.append(acc_w)

            loss_list.append(sum(batch_eval_loss['total']) / len(batch_eval_loss['total']))

    return acc_list_l, acc_list_g


def test_inference_fedavg(args,round, local_model_list, test_dataset, user_groups_gt,logger,summary_writer):
    """ Returns the test accuracy and loss.
    """
    acc_list_l = []#wo
    loss_list_l=[]
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(args.device)
        test_acc, test_loss = test_inference(args, model, test_dataset, user_groups_gt, idx)

        print(' User: %d  Loss: %f ||  test_acc: %f ' % (idx ,test_loss,test_acc))
        logger.info(' User: %d  Loss: %f ||  test_acc: %f ' % (idx ,test_loss,test_acc))
        summary_writer.add_scalar('scalar/net_id%d_Test_Accuracy' % (idx), test_acc, round)
        acc_list_l.append(test_acc)
        loss_list_l.append(test_loss)

    return acc_list_l,loss_list_l




def save_protos(args, local_model_list, test_dataset, user_groups_gt):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)

    agg_protos_label = {}
    for idx in range(args.num_users):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            outputs, protos,_ = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label[idx]:
                    agg_protos_label[idx][labels[i].item()].append(protos[i, :])
                else:
                    agg_protos_label[idx][labels[i].item()] = [protos[i, :]]

    x = []
    y = []
    d = []
    for i in range(args.num_users):
        for label in agg_protos_label[i].keys():
            for proto in agg_protos_label[i][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                d.append(i)

    x = np.array(x)
    y = np.array(y)
    d = np.array(d)
    np.save('./' + args.alg + '_protos.npy', x)
    np.save('./' + args.alg + '_labels.npy', y)
    np.save('./' + args.alg + '_idx.npy', d)

    print("Save protos and labels successfully.")



from torch.autograd import Variable
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


def train_global_proto_model(global_model,train_dataloder):
    epochs = 6
    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(global_model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        global_model.train()
        num_correct = 0
        num_samples = 0
        for image, label in train_dataloder:
            image, label = image.to(device), label.to(device)
            optim.zero_grad()
            out = global_model.forward(image.detach())
            pre = out.max(1).indices
            num_correct += (pre == label).sum()
            num_samples += pre.size(0)

            loss = loss_function(out, label)
            loss.backward()
            optim.step()
        acc = num_correct.item() / num_samples
        print("global epoch:{},train acc={}".format(epoch, acc))

    global_model.eval()
    logits=[]
    la=[]
    with torch.no_grad():
        for im, l in train_dataloder:
            im, l = im.to(device), l.to(device)
            output = global_model(im)
            logits.extend(output)
            la.extend(l)
    logits=torch.stack(logits)
    la=torch.tensor(la)
    class_logits={}
    for i in range(len(la)):
        lo=logits[i]
        lo_label=la[i]
        if lo_label.item() in class_logits:
            class_logits[lo_label.item()].append(lo)
        else:
            class_logits[lo_label.item()] = [lo]
    # global logits
    class_logits=agg_func(class_logits)
    return class_logits


def fine_tune_global_model_safs(args, global_model, synthetic_data_list, global_protos, summary_writer=None, logger=None, round=None):
    """
    利用 SAFS 合成特征微调全局模型，并生成全局 logits。
    
    该函数用于利用合成特征微调全局模型，并生成全局 logits。
    
    Args:
        args: 参数对象
        global_model: GlobalFedmps 实例
        synthetic_data_list: feature_synthesis 返回的列表，包含 {'class_index', 'synthetic_features'}
        global_protos: 全局原型字典 {class_index: [proto_tensor]}，用于生成最终的 global_logits
        summary_writer: TensorBoard SummaryWriter 实例，用于记录指标（可选）
        logger: Logger 实例，用于记录日志（可选）
        round: 当前轮次，用于记录指标（可选）
    
    Returns:
        global_logits: 字典 {class_index: logit_tensor}，用于客户端蒸馏
    """
    device = args.device
    global_model = global_model.to(device)
    
    # ========== 数据准备 ==========
    # 遍历 synthetic_data_list，提取所有 synthetic_features (作为 X) 和对应的 class_index (作为 Y)
    synthetic_features_list = []
    synthetic_labels_list = []
    
    for syn_data in synthetic_data_list:
        class_index = syn_data['class_index']
        synthetic_features = syn_data['synthetic_features']  # shape: (syn_num, feature_dim)
        
        # 为每个合成特征创建对应的标签
        num_syn = synthetic_features.shape[0]
        labels = torch.full((num_syn,), class_index, dtype=torch.long)
        
        synthetic_features_list.append(synthetic_features)
        synthetic_labels_list.append(labels)
    
    # 拼接所有特征和标签
    if len(synthetic_features_list) == 0:
        raise ValueError("synthetic_data_list 为空，无法进行微调")
    
    X_all = torch.cat(synthetic_features_list, dim=0)  # shape: (total_syn_num, feature_dim)
    Y_all = torch.cat(synthetic_labels_list, dim=0)    # shape: (total_syn_num,)
    
    # 封装成 TensorDataset 和 DataLoader
    dataset = TensorDataset(X_all, Y_all)
    batch_size = getattr(args, 'safs_finetune_batch_size', 32)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # ========== 模型训练 (Fine-tuning) ==========
    # 获取训练轮数，默认 5-10 个 epoch
    epochs = getattr(args, 'safs_finetune_epochs', 5)
    
    # 使用 SGD 优化器，lr=0.01, momentum=0.9 (根据 SFD 论文)
    optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    
    global_model.train()
    for epoch in range(epochs):
        num_correct = 0
        num_samples = 0
        epoch_loss = 0.0
        num_batches = 0
        
        for features, labels in train_dataloader:
            features = features.to(device)  # shape: (batch_size, feature_dim)
            labels = labels.to(device)       # shape: (batch_size,)
            
            optimizer.zero_grad()
            
            # 将特征输入到 global_model (注意：GlobalFedmps 接受 x1 作为输入)
            out = global_model.forward(features)
            
            # 计算损失
            loss = loss_function(out, labels)
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            pred = out.max(1).indices
            num_correct += (pred == labels).sum().item()
            num_samples += labels.size(0)
            epoch_loss += loss.item()
            num_batches += 1
        
        acc = num_correct / num_samples if num_samples > 0 else 0.0
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"SAFS fine-tuning epoch {epoch+1}/{epochs}, train acc={acc:.4f}, loss={avg_loss:.4f}")
        
        # 记录 SAFS 微调指标（如果提供了 summary_writer 和 round）
        if summary_writer is not None and round is not None:
            summary_writer.add_scalar('scalar/SAFS_FineTune_Loss', avg_loss, round * epochs + epoch)
            summary_writer.add_scalar('scalar/SAFS_FineTune_Acc', acc, round * epochs + epoch)
        
        if logger is not None:
            logger.info(f"SAFS fine-tuning epoch {epoch+1}/{epochs}, train acc={acc:.4f}, loss={avg_loss:.4f}")
    
    # ========== 生成 Global Logits ==========
    # 微调完成后，将模型设为 eval() 模式
    global_model.eval()
    
    global_logits = {}
    
    with torch.no_grad():
        # 遍历 global_protos（这是本轮聚合后的全局原型）
        for class_index, proto_list in global_protos.items():
            # 如果一个类别有多个原型，取平均
            if len(proto_list) > 0:
                # 将原型列表转换为张量
                proto_tensor = torch.stack(proto_list) if isinstance(proto_list, list) else proto_list
                
                # 如果有多个原型，取平均
                if proto_tensor.dim() > 1 and proto_tensor.shape[0] > 1:
                    proto_tensor = proto_tensor.mean(dim=0, keepdim=True)
                elif proto_tensor.dim() == 1:
                    proto_tensor = proto_tensor.unsqueeze(0)
                
                # 移动到设备
                proto_tensor = proto_tensor.to(device)
                
                # 将全局原型输入到微调后的 global_model 中，获取输出 logits
                logit = global_model.forward(proto_tensor)  # shape: (1, num_classes)
                
                # 取第一个（也是唯一的）logit
                logit = logit.squeeze(0)  # shape: (num_classes,)
                
                global_logits[class_index] = logit
    
    print(f"Generated global logits for {len(global_logits)} classes using SAFS fine-tuned model")
    
    return global_logits

