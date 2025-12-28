#!/usr/bin/env python
# -*- coding: utf-8 -*-hello
# Python version: 3.6

import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
from pathlib import Path
from torch.utils.data import TensorDataset
import datetime
import logging
import pickle
import random
import numpy as np
import math

# 将项目根目录添加到 sys.path
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from lib.options import *
from lib.update import *
from lib.models.models import *
from lib.utils import *
from lib.sfd_utils import RFF, aggregate_global_statistics
from lib.safs import MeanCovAligner, feature_synthesis, make_syn_nums

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# Record console output
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
	    pass

def Fedavg(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, summary_writer,logger,logdir):

    idxs_users = np.arange(args.num_users)
    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses= [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w= local_model.update_weights_fedavg( idx=idx,model=copy.deepcopy(local_model_list[idx]))
            local_weights.append(copy.deepcopy(w))

        # aggregate local weights
        w_avg = copy.deepcopy(local_weights[0])
        for k in w_avg.keys():
            for i in range(1, len(local_weights)):
                w_avg[k] += local_weights[i][k]
            w_avg[k] = torch.div(w_avg[k], len(local_weights))

        # Update each local model with the globally averaged parameters
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        # test
        acc_list_l, loss_list_l= test_inference_fedavg(args,round, local_model_list, test_dataset, user_groups_lt,logger,summary_writer)
        print('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        logger.info('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
            net = copy.deepcopy(local_model_list[0])
            torch.save(net.state_dict(), logdir + '/localmodel0.pth')

    print('best results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round,best_acc,best_std))
    logger.info('best results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round,best_acc,best_std))


def Fedprox(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,logdir):

    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses= [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train=[]
        loss_list_train=[]
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, idx_acc = local_model.update_weights_prox(args,idx, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss)
            local_weights.append(copy.deepcopy(w))

        # update global weights
        local_weights_list = local_weights
        w_avg = copy.deepcopy(local_weights_list[0])
        for k in w_avg.keys():
            for i in range(1, len(local_weights_list)):
                w_avg[k] += local_weights_list[i][k]
            w_avg[k] = torch.div(w_avg[k], len(local_weights_list))

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        # test
        acc_list_l, loss_list_l = test_inference_fedavg(args, round, local_model_list, test_dataset,user_groups_lt, logger, summary_writer)
        print('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        logger.info('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        summary_writer.add_scalars('scalar/Total_Avg_Accuracy', {'train':np.mean(acc_list_train),'test':np.mean(acc_list_l)}, round)

        logger.info('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        summary_writer.add_scalars('scalar/Total_Avg_Loss',{'train': np.mean(loss_list_train), 'test': np.mean(loss_list_l)}, round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
            net = copy.deepcopy(local_model_list[0])
            torch.save(net.state_dict(), logdir + '/localmodel0.pth')

        print('best results:')
        print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))
        logger.info('best results:')
        logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))

def Moon(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,global_model,logger,summary_writer,logdir):

    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0
    old_nets_pool=[]#1

    if len(old_nets_pool) < 1:
        old_nets = copy.deepcopy(local_model_list)
        for net in old_nets:
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

    party_list_this_round = [i for i in range(args.num_users)]
    for round in tqdm(range(args.rounds)):

        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False

        local_weights, local_losses= [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train=[]
        loss_list_train=[]

        for idx in idxs_users:
            prev_models = []
            for i in range(len(old_nets_pool)):
                prev_models.append(old_nets_pool[i][idx])
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, idx_acc = local_model.update_weights_moon(args,idx, model=copy.deepcopy(local_model_list[idx]),global_model=global_model,previous_models=prev_models, global_round=round)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        total_data_points = sum([len(user_groups[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(user_groups[r]) / total_data_points for r in party_list_this_round]

        local_weights_list = local_weights
        w_avg = copy.deepcopy(local_weights_list[0])
        for key, value in w_avg.items():
            w_avg[key] = value * fed_avg_freqs[0]
        for k in w_avg.keys():
            for i in range(1, len(local_weights_list)):
                w_avg[k] += local_weights_list[i][k]*fed_avg_freqs[i]

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        global_model.load_state_dict(w_avg)

        if len(old_nets_pool) < args.model_buffer_size:
            old_nets = copy.deepcopy(local_model_list)
            for  net in old_nets:
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            old_nets_pool.append(old_nets)
        elif args.pool_option == 'FIFO':
            old_nets = copy.deepcopy(local_model_list)
            for net in old_nets:
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            for i in range(args.model_buffer_size - 2, -1, -1):
                old_nets_pool[i] = old_nets_pool[i + 1]
            old_nets_pool[args.model_buffer_size - 1] = old_nets

        acc_list_l, loss_list_l,acc_list_g, loss_list,loss_total_list = test_inference_new_het_lt(args, local_model_list, test_dataset,classes_list, user_groups_lt)

        print('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_l), np.std(acc_list_l)))
        logger.info('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_l), np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round

    print('best results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))
    logger.info('best results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))

def fedntd(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, summary_writer,logger,logdir):

    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses= [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w= local_model.update_weights_fedntd(args, idx=idx,model=copy.deepcopy(local_model_list[idx]))
            local_weights.append(copy.deepcopy(w))

        # update global weights
        w_avg = copy.deepcopy(local_weights[0])
        for k in w_avg.keys():
            for i in range(1, len(local_weights)):
                w_avg[k] += local_weights[i][k]
            w_avg[k] = torch.div(w_avg[k], len(local_weights))

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        # test
        acc_list_l, loss_list_l= test_inference_fedavg(args,round, local_model_list, test_dataset, user_groups_lt,logger,summary_writer)
        print('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        logger.info('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round

    print('best results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round,best_acc,best_std))
    logger.info('best results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round,best_acc,best_std))

def fedgkd(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, logdir):
    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0

    models_buffer = []
    ensemble_model = None

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train = []
        loss_list_train = []
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, idx_acc = local_model.update_weights_gkd(args, idx, model=copy.deepcopy(local_model_list[idx]), global_round=round, avg_teacher=ensemble_model)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss)
            local_weights.append(copy.deepcopy(w))

        # update global avg weights for this round
        local_weights_list = local_weights
        w_avg = copy.deepcopy(local_weights_list[0])
        for k in w_avg.keys():
            for i in range(1, len(local_weights_list)):
                w_avg[k] += local_weights_list[i][k]
            w_avg[k] = torch.div(w_avg[k], len(local_weights_list))

        # update global ensemble weights
        if len(models_buffer) >= args.buffer_length:
            models_buffer.pop(0)
        models_buffer.append(copy.deepcopy(w_avg))

        ensemble_w = copy.deepcopy(models_buffer[0])
        for k in ensemble_w.keys():
            for i in range(1, len(models_buffer)):
                ensemble_w[k] += models_buffer[i][k]
            ensemble_w[k] = torch.div(ensemble_w[k], len(models_buffer))

        if ensemble_model is None:
            ensemble_model = copy.deepcopy(local_model_list[0])
        ensemble_model.load_state_dict(ensemble_w, strict=True)

        # provide the client with the average model, not the ensemble model
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model

        # test
        acc_list_l, loss_list_l = test_inference_fedavg(args, round, local_model_list, test_dataset, user_groups_lt, logger, summary_writer)
        print('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean( acc_list_l), np.std( acc_list_l)))
        logger.info('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean( acc_list_l), np.std( acc_list_l)))
        summary_writer.add_scalars('scalar/Total_Avg_Accuracy',{'train': np.mean(acc_list_train), 'test': np.mean(acc_list_l)}, round)

        logger.info('| ROUND: {} | For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean( acc_list_l), np.std( acc_list_l)))
        summary_writer.add_scalars('scalar/Total_Avg_Loss',{'train': np.mean(loss_list_train), 'test': np.mean(loss_list_l)}, round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round

        print('best results:')
        print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))
        logger.info('best results:')
        logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))
def Fedproc(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):

    idxs_users = np.arange(args.num_users)
    party_list_this_round = [i for i in range(args.num_users)]
    global_protos = []

    best_acc = -float('inf')
    best_std = -float('inf')
    best_round = 0
    for round in tqdm(range(args.rounds)):

        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train=[]
        loss_list_train=[]

        for idx in idxs_users:

            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos, idx_acc = local_model.update_weights_fedproc(args,idx, model=copy.deepcopy(local_model_list[idx]),global_protos=global_protos, global_round=round)
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss['1'])
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_protos[idx] = agg_protos

        # update global protos
        global_protos = proto_aggregation(local_protos)
        # update global weights
        total_data_points = sum([len(user_groups[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(user_groups[r]) / total_data_points for r in party_list_this_round]

        local_weights_list = local_weights
        w_avg = copy.deepcopy(local_weights_list[0])
        for key, value in w_avg.items():
            w_avg[key] = value * fed_avg_freqs[0]
        for k in w_avg.keys():
            for i in range(1, len(local_weights_list)):
                w_avg[k] += local_weights_list[i][k]*fed_avg_freqs[i]

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(w_avg, strict=True)
            local_model_list[idx] = local_model


        acc_list_l, loss_list_l,acc_list_g, loss_list,loss_total_list = test_inference_new_het_lt(args, local_model_list, test_dataset,classes_list, user_groups_lt)
        print('| ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
            round, np.mean(acc_list_l), np.std(acc_list_l)))
        logger.info(
            '| ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
                round, np.mean(acc_list_l), np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round

    print('best results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
        best_round, best_acc, best_std))
    logger.info('best results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(
        best_round, best_acc, best_std))


def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, summary_writer,logger,logdir):

    global_protos = []
    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf')
    best_std = -float('inf')
    best_acc_w=-float('inf')
    best_std_w=-float('inf')
    best_round = 0
    best_round_w=0
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w,protos = local_model.update_weights_fedproto(args, idx, global_protos,model=copy.deepcopy(local_model_list[idx]),global_round=round)
            agg_protos = agg_func(protos)
            local_weights.append(copy.deepcopy(w))
            local_protos[idx] = agg_protos

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global weights
        global_protos = proto_aggregation(local_protos)

        # test
        acc_list_l, acc_list_g= test_inference_fedproto(args,logger, local_model_list, test_dataset,classes_list, user_groups_lt, global_protos)

        print('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l), np.std(acc_list_l)))
        logger.info('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_l),np.std(acc_list_l)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        print('| ROUND: {} | For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round,np.mean(acc_list_g), np.std(acc_list_g)))
        logger.info('| ROUND: {} | For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_g), np.std(acc_list_g)))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy_wp', np.mean(acc_list_g), round)

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
        if np.mean(acc_list_g) > best_acc_w:
            best_acc_w = np.mean(acc_list_g)
            best_std_w = np.std(acc_list_g)
            best_round_w = round

    print('best wo results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))
    logger.info('best wo results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))

    print('best w results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round_w , best_acc_w ,best_std_w ))
    logger.info('best w results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round_w ,best_acc_w ,best_std_w ))


def FedMPS(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,summary_writer,logger,logdir):
    """
    FedMPS 主训练函数（集成 ABBL）
    
    注意：确保 local_model_list 中的模型已经在主程序中正确移动到 GPU（通常在模型构建时完成）。
    """
    
    # ABBL: 初始化检查 - 确保所有本地模型已正确移动到设备
    for idx, model in enumerate(local_model_list):
        model_device = next(model.parameters()).device
        expected_device = torch.device(args.device)
        if model_device != expected_device:
            print(f'Warning: Model {idx} is on {model_device}, expected {expected_device}, moving it now...')
            logger.warning(f'Model {idx} is on {model_device}, expected {expected_device}, moving it now...')
            model.to(args.device)
    
    # global model: shares the same structure as the output layer of each local model
    global_model = GlobalFedmps(args)
    global_model.to(args.device)
    global_model.train()

    global_high_protos = {}
    global_low_protos = {}
    global_logits = {}
    idxs_users = np.arange(args.num_users)

    best_acc = -float('inf') # best results wo protos
    best_std = -float('inf')
    best_round = 0
    best_acc_w = -float('inf')  # best results w protos
    best_std_w = -float('inf')
    best_round_w = 0
    
    # ========== Initialize RFF models for SFD Statistics Aggregation ==========
    # This initialization only needs to be done once before the main loop
    print('\n' + '='*60)
    print('Initializing SFD Statistics Aggregation Components')
    print('='*60)
    logger.info('Initializing SFD Statistics Aggregation Components')
    
    # Step 1: Get feature dimensions by running a dummy forward pass
    # Use the first client's model to determine feature dimensions
    dummy_model = copy.deepcopy(local_model_list[0])
    dummy_model.eval()
    dummy_model = dummy_model.to(args.device)
    
    # Create a dummy input to get feature dimensions (adjust size based on dataset)
    if args.dataset == 'mnist' or args.dataset == 'femnist' or args.dataset == 'fashion':
        dummy_input = torch.randn(1, 1, 28, 28).to(args.device)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'realwaste' or args.dataset == 'flowers' or args.dataset == 'defungi':
        dummy_input = torch.randn(1, 3, 32, 32).to(args.device)
    elif args.dataset == 'tinyimagenet':
        dummy_input = torch.randn(1, 3, 64, 64).to(args.device)
    elif args.dataset == 'imagenet':
        dummy_input = torch.randn(1, 3, 224, 224).to(args.device)
    else:
        # Default to CIFAR-10 size
        dummy_input = torch.randn(1, 3, 32, 32).to(args.device)
    
    with torch.no_grad():
        dummy_output = dummy_model(dummy_input)
        if isinstance(dummy_output, tuple) and len(dummy_output) >= 5:
            # 返回值顺序: logits, log_probs, high_level_features, low_level_features, projected_features
            high_feature_dim = dummy_output[2].shape[1]  # high-level feature dimension (索引2)
            low_feature_dim = dummy_output[3].shape[1]   # low-level feature dimension (索引3)
        else:
            raise ValueError(f"模型输出格式不符合预期，期望5个返回值，实际得到{len(dummy_output) if isinstance(dummy_output, tuple) else '非元组'}")
    
    print(f'Detected feature dimensions: high={high_feature_dim}, low={low_feature_dim}')
    logger.info(f'Detected feature dimensions: high={high_feature_dim}, low={low_feature_dim}')
    
    # Step 2: Initialize RFF models for high and low levels
    # Set random seed for reproducibility
    backup_rng_state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state()
    }
    
    # Set deterministic seed for RFF initialization
    random.seed(args.rf_seed)
    np.random.seed(args.rf_seed)
    torch.manual_seed(args.rf_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.rf_seed)
    
    # Initialize RFF models
    rf_model_high = RFF(
        d=high_feature_dim,
        D=args.rf_dim_high,
        gamma=args.rbf_gamma_high,
        device=args.device,
        rf_type=args.rf_type
    )
    
    rf_model_low = RFF(
        d=low_feature_dim,
        D=args.rf_dim_low,
        gamma=args.rbf_gamma_low,
        device=args.device,
        rf_type=args.rf_type
    )
    
    rf_models = {
        'high': rf_model_high,
        'low': rf_model_low
    }
    
    # Restore random state
    random.setstate(backup_rng_state['python'])
    np.random.set_state(backup_rng_state['numpy'])
    torch.set_rng_state(backup_rng_state['torch'])
    
    print(f'Initialized RFF models: high (d={high_feature_dim}, D={args.rf_dim_high}, gamma={args.rbf_gamma_high}), '
          f'low (d={low_feature_dim}, D={args.rf_dim_low}, gamma={args.rbf_gamma_low})')
    logger.info(f'Initialized RFF models: high (d={high_feature_dim}, D={args.rf_dim_high}, gamma={args.rbf_gamma_high}), '
                f'low (d={low_feature_dim}, D={args.rf_dim_low}, gamma={args.rbf_gamma_low})')
    print('='*60 + '\n')
    
    # Initialize global_stats to store the last round's statistics
    global_stats = None
    
    # ========== 保存数据分布元数据 (Metadata) ==========
    # 在训练开始前,收集并保存每个客户端的 pi_sample_per_class 和 classes_list
    print('[Before Training] Saving data distribution metadata...')
    logger.info('[Before Training] Saving data distribution metadata...')
    
    # 收集每个客户端的 pi_sample_per_class
    client_pi_samples = {}
    for idx in idxs_users:
        local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
        # 将 pi_sample_per_class 转换为 CPU 并脱离计算图
        pi_cpu = local_model.pi_sample_per_class.cpu().detach().clone()
        client_pi_samples[idx] = pi_cpu.numpy()  # 转换为 numpy 数组便于保存
    
    # 保存元数据到 pickle 文件
    metadata_dict = {
        'client_pi_sample_per_class': client_pi_samples,  # 每个客户端的平滑类别分布
        'classes_list': classes_list,  # 客户端 ID 与其拥有类别的映射表
        'num_users': args.num_users,
        'num_classes': args.num_classes,
        'beta_pi': getattr(args, 'beta_pi', 0.5)
    }
    
    metadata_path = os.path.join(logdir, 'data_distribution_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata_dict, f)
    print(f'Saved data distribution metadata to {metadata_path}')
    logger.info(f'Saved data distribution metadata to {metadata_path}')
    
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_high_protos, local_low_protos = [], [], {}, {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        acc_list_train = []
        loss_list_train = []
        loss_ace_list = []  # ABBL: Loss_ACE (L_ACE)
        loss_scl_list = []  # ABBL: Loss_SCL (L_A-SCL)
        loss_proto_high_list = []  # FedMPS: Loss_proto_high (L_proto_high)
        loss_proto_low_list = []   # FedMPS: Loss_proto_low (L_proto_low)
        loss_soft_list = []        # FedMPS: Loss_soft (L_soft)
        
        # ABBL: 计算当前轮的 scl_weight（用于日志记录）
        scl_weight_start = getattr(args, 'scl_weight_start', 1.0)
        scl_weight_end = getattr(args, 'scl_weight_end', 0.0)
        if args.rounds > 0:
            scl_weight = 0.5 * (scl_weight_start - scl_weight_end) * (1 + math.cos(math.pi * round / args.rounds)) + scl_weight_end
        else:
            scl_weight = scl_weight_start
        for idx in idxs_users:
            # local model updating
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            # ABBL: 传递 total_rounds 参数用于计算余弦退火权重
            w, loss, acc, high_protos, low_protos, idx_acc = local_model.update_weights_fedmps(
                args, idx, global_high_protos, global_low_protos, global_logits, 
                model=copy.deepcopy(local_model_list[idx]), global_round=round,
                total_rounds=args.rounds,  # ABBL: 传递总轮数用于余弦退火
                rf_models=rf_models, global_stats=global_stats
            )
            acc_list_train.append(idx_acc)
            loss_list_train.append(loss['total'])
            # ABBL: 记录所有损失分量
            loss_ace_list.append(loss['ace'])
            loss_scl_list.append(loss['scl'])
            loss_proto_high_list.append(loss['proto_high'])
            loss_proto_low_list.append(loss['proto_low'])
            loss_soft_list.append(loss['soft'])
            agg_high_protos = agg_func(high_protos)
            agg_low_protos = agg_func(low_protos)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_high_protos[idx] = agg_high_protos
            local_low_protos[idx] = agg_low_protos

        # aggregate local multi-level prototypes instead of local weights
        local_weights_list = local_weights
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        global_high_protos = proto_aggregation(local_high_protos)
        global_low_protos = proto_aggregation(local_low_protos)

        # ========== SFD Statistics Aggregation Stage ==========
        # Collect local statistics from all clients
        print(f'[Round {round+1}] Collecting local statistics from all clients...')
        logger.info(f'[Round {round+1}] Collecting local statistics from all clients...')
        
        # 获取统计量计算层级（从 args 中获取，默认为 'high'）
        stats_level = getattr(args, 'stats_level', 'high')
        
        client_responses = []
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            local_stats = local_model.get_local_statistics(
                model=copy.deepcopy(local_model_list[idx]),
                rf_models=rf_models,
                args=args,
                stats_level=stats_level
            )
            client_responses.append(local_stats)
        
        # Aggregate global statistics
        print(f'[Round {round+1}] Aggregating global statistics...')
        logger.info(f'[Round {round+1}] Aggregating global statistics...')
        
        global_stats = aggregate_global_statistics(
            client_responses=client_responses,
            class_num=args.num_classes,
            stats_level=stats_level
        )
        
        print(f'[Round {round+1}] Global statistics aggregation completed')
        logger.info(f'[Round {round+1}] Global statistics aggregation completed')
        
        # ========== SFD SAFS Feature Synthesis Stage ==========
        # 如果启用了 SAFS，执行特征合成
        if getattr(args, 'enable_safs', 0) == 1:
            print(f'[Round {round+1}] Starting SAFS feature synthesis...')
            logger.info(f'[Round {round+1}] Starting SAFS feature synthesis...')
            
            # 确定使用的层级（与统计量聚合层级一致）
            level_to_use = stats_level if stats_level in ['high', 'low'] else 'high'
            
            # 获取对应层级的全局统计量和 RFF 模型
            if level_to_use == 'high':
                global_stats_level = global_stats['high']
                rf_model = rf_models['high']
                feature_dim = high_feature_dim
            else:  # level_to_use == 'low'
                global_stats_level = global_stats['low']
                rf_model = rf_models['low']
                feature_dim = low_feature_dim
            
            # 提取全局统计量
            class_means = global_stats_level['class_means']
            class_covs = global_stats_level['class_covs']
            class_rf_means = global_stats_level['class_rf_means']
            sample_per_class = global_stats['sample_per_class']
            
            # 计算每个类别的合成特征数量
            syn_nums = make_syn_nums(
                class_sizes=sample_per_class.tolist(),
                max_num=getattr(args, 'safs_max_syn_num', 2000),
                min_num=getattr(args, 'safs_min_syn_num', 600)
            )
            
            # 验证合成特征数量是否足够（必须大于特征维度）
            assert min(syn_nums) > feature_dim, \
                f'最小合成特征数量 {min(syn_nums)} 必须大于特征维度 {feature_dim}'
            
            print(f'[Round {round+1}] Synthetic feature numbers per class: {syn_nums}')
            logger.info(f'[Round {round+1}] Synthetic feature numbers per class: {syn_nums}')
            
            # 为每个类别创建 MeanCov Aligner
            aligners = []
            for c in range(args.num_classes):
                aligner = MeanCovAligner(
                    target_mean=class_means[c],
                    target_cov=class_covs[c],
                    target_cov_eps=getattr(args, 'safs_target_cov_eps', 1e-5)
                )
                aligners.append(aligner)
            
            # 执行特征合成
            class_syn_datasets = feature_synthesis(
                feature_dim=feature_dim,
                class_num=args.num_classes,
                device=args.device,
                aligners=aligners,
                rf_model=rf_model,
                class_rf_means=class_rf_means,
                steps=getattr(args, 'safs_steps', 1000),
                lr=getattr(args, 'safs_lr', 0.1),
                syn_num_per_class=syn_nums,
                input_cov_eps=getattr(args, 'safs_input_cov_eps', 1e-5),
            )
            
            print(f'[Round {round+1}] SAFS feature synthesis completed')
            logger.info(f'[Round {round+1}] SAFS feature synthesis completed')
            print(f'[Round {round+1}] Generated synthetic features for {len(class_syn_datasets)} classes')
            logger.info(f'[Round {round+1}] Generated synthetic features for {len(class_syn_datasets)} classes')
            
            # 可选：保存合成特征数据集（用于后续的分类器微调）
            # 这里可以根据需要保存到文件或传递给分类器微调阶段
            # 例如：torch.save(class_syn_datasets, f'{logdir}/synthetic_features_round_{round+1}.pt')
        else:
            class_syn_datasets = None
        
        # ========== Global Model Training / Fine-tuning ==========
        # 根据是否启用 SAFS 选择不同的全局模型训练方式
        if getattr(args, 'enable_safs', 0) == 1 and class_syn_datasets is not None and len(class_syn_datasets) > 0:
            # 使用 SAFS 合成特征微调全局模型
            print(f'[Round {round+1}] Fine-tuning global model using SAFS synthetic features...')
            logger.info(f'[Round {round+1}] Fine-tuning global model using SAFS synthetic features...')
            
            global_logits = fine_tune_global_model_safs(
                args,
                global_model,
                class_syn_datasets,
                global_high_protos,  # 注意：FedMPS主要使用 high_protos 进行分类层训练
                summary_writer=summary_writer,
                logger=logger,
                round=round
            )
            
            print(f'[Round {round+1}] Global model fine-tuned using SAFS synthetic features.')
            logger.info(f'[Round {round+1}] Global model fine-tuned using SAFS synthetic features.')
        else:
            # 使用原来的方法：基于本地原型训练全局模型
            print(f'[Round {round+1}] Training global model using local prototypes...')
            logger.info(f'[Round {round+1}] Training global model using local prototypes...')
            
            # create inputs: local high-level prototypes
            global_data, global_label = get_global_input(local_high_protos)
            dataset = TensorDataset(global_data, global_label)
            train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            # begin training and output global logits
            global_logits = train_global_proto_model(global_model, train_dataloader)

        # ABBL: 记录训练损失（包括所有损失分量）以及权重退火
        print('| ROUND: {} | Train Loss - Total: {:.5f}, L_ACE: {:.5f}, L_A-SCL: {:.5f}, L_proto_high: {:.5f}, L_proto_low: {:.5f}, L_soft: {:.5f}, SCL_Weight: {:.5f}'.format(
            round, np.mean(loss_list_train), np.mean(loss_ace_list), np.mean(loss_scl_list), 
            np.mean(loss_proto_high_list), np.mean(loss_proto_low_list), np.mean(loss_soft_list), scl_weight))
        logger.info('| ROUND: {} | Train Loss - Total: {:.5f}, L_ACE: {:.5f}, L_A-SCL: {:.5f}, L_proto_high: {:.5f}, L_proto_low: {:.5f}, L_soft: {:.5f}, SCL_Weight: {:.5f}'.format(
            round, np.mean(loss_list_train), np.mean(loss_ace_list), np.mean(loss_scl_list),
            np.mean(loss_proto_high_list), np.mean(loss_proto_low_list), np.mean(loss_soft_list), scl_weight))
        summary_writer.add_scalar('scalar/Train_Total_Loss', np.mean(loss_list_train), round)
        summary_writer.add_scalar('scalar/Train_Loss_ACE', np.mean(loss_ace_list), round)
        summary_writer.add_scalar('scalar/Train_Loss_SCL', np.mean(loss_scl_list), round)
        summary_writer.add_scalar('scalar/Train_Loss_Proto_High', np.mean(loss_proto_high_list), round)
        summary_writer.add_scalar('scalar/Train_Loss_Proto_Low', np.mean(loss_proto_low_list), round)
        summary_writer.add_scalar('scalar/Train_Loss_Soft', np.mean(loss_soft_list), round)
        summary_writer.add_scalar('scalar/SCL_Weight', scl_weight, round)  # ABBL: 记录权重退火变化

        # test
        acc_list_l, loss_list_l, acc_list_g, loss_list, loss_total_list = test_inference_new_het_lt(args,local_model_list,test_dataset,classes_list,user_groups_lt,global_high_protos)

        # 记录每个客户端的准确率（细粒度性能记录）
        for idx in range(args.num_users):
            summary_writer.add_scalar(f'scalar/Client_{idx}_Test_Acc_wo_Protos', acc_list_l[idx], round)
            if idx < len(acc_list_g):
                summary_writer.add_scalar(f'scalar/Client_{idx}_Test_Acc_w_Protos', acc_list_g[idx], round)

        # 计算并记录标准差（用于评估公平性）
        std_acc_wo = np.std(acc_list_l)
        std_acc_w = np.std(acc_list_g) if len(acc_list_g) > 0 else 0.0
        summary_writer.add_scalar('scalar/Total_Test_Std_Accuracy_wo_Protos', std_acc_wo, round)
        summary_writer.add_scalar('scalar/Total_Test_Std_Accuracy_w_Protos', std_acc_w, round)

        print('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_l), std_acc_wo))
        logger.info('| ROUND: {} | For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_l), std_acc_wo))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy', np.mean(acc_list_l), round)

        print('| ROUND: {} | For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_g), std_acc_w))
        logger.info('| ROUND: {} | For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(round, np.mean(acc_list_g), std_acc_w))
        summary_writer.add_scalar('scalar/Total_Test_Avg_Accuracy_wp', np.mean(acc_list_g), round)

        # ========== 原型稳定性分析 (Prototype Data) ==========
        # 每隔 10 个 Round,保存当前的 global_high_protos 和 global_low_protos
        if (round + 1) % 10 == 0:
            print(f'[Round {round+1}] Saving prototype data for stability analysis...')
            logger.info(f'[Round {round+1}] Saving prototype data for stability analysis...')
            
            # 将原型转换为 CPU 并脱离计算图
            proto_data = {
                'round': round + 1,
                'global_high_protos': {},
                'global_low_protos': {}
            }
            
            # 处理 global_high_protos
            # 注意: proto_aggregation 返回的格式是 {class_idx: [proto_tensor]}, 列表只包含一个张量
            for class_idx, proto_list in global_high_protos.items():
                if isinstance(proto_list, list) and len(proto_list) > 0:
                    # 列表通常只包含一个张量,直接取第一个元素
                    proto_tensor = proto_list[0] if isinstance(proto_list[0], torch.Tensor) else torch.tensor(proto_list[0])
                    proto_data['global_high_protos'][class_idx] = proto_tensor.cpu().detach().numpy()
                elif isinstance(proto_list, torch.Tensor):
                    proto_data['global_high_protos'][class_idx] = proto_list.cpu().detach().numpy()
            
            # 处理 global_low_protos
            for class_idx, proto_list in global_low_protos.items():
                if isinstance(proto_list, list) and len(proto_list) > 0:
                    proto_tensor = proto_list[0] if isinstance(proto_list[0], torch.Tensor) else torch.tensor(proto_list[0])
                    proto_data['global_low_protos'][class_idx] = proto_tensor.cpu().detach().numpy()
                elif isinstance(proto_list, torch.Tensor):
                    proto_data['global_low_protos'][class_idx] = proto_list.cpu().detach().numpy()
            
            # 保存到 pickle 文件
            proto_save_path = os.path.join(logdir, f'prototypes_round_{round+1}.pkl')
            with open(proto_save_path, 'wb') as f:
                pickle.dump(proto_data, f)
            print(f'Saved prototype data to {proto_save_path}')
            logger.info(f'Saved prototype data to {proto_save_path}')

        if np.mean(acc_list_l) > best_acc:
            best_acc = np.mean(acc_list_l)
            best_std = np.std(acc_list_l)
            best_round = round
        if np.mean(acc_list_g) > best_acc_w:
            best_acc_w = np.mean(acc_list_g)
            best_std_w = np.std(acc_list_g)
            best_round_w = round

    print('best wo results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))
    logger.info('best wo results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round, best_acc, best_std))

    print('best w results:')
    print('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round_w, best_acc_w, best_std_w))
    logger.info('best w results:')
    logger.info('| BEST ROUND: {} | For all users , mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(best_round_w, best_acc_w, best_std_w))
    
    # Save final SFD statistics (from the last round)
    # Use the same logdir pattern as the main function
    save_dir = os.path.join('../newresults', args.alg, str(datetime.datetime.now().strftime("%Y-%m-%d/%H.%M.%S"))+'_'+args.dataset+'_n'+str(args.ways)+'_sfd_stats')
    mkdirs(save_dir)
    
    # Save final global statistics from the last round
    if global_stats is not None:
        stats_save_path = os.path.join(save_dir, 'global_stats_final.pkl')
        with open(stats_save_path, 'wb') as f:
            pickle.dump(global_stats, f)
        print(f'Saved final global statistics to {stats_save_path}')
        logger.info(f'Saved final global statistics to {stats_save_path}')
    
    # Save RFF models state dict
    rf_models_save_path = os.path.join(save_dir, 'rf_models.pkl')
    rf_models_state = {
        'high': rf_model_high.state_dict(),
        'low': rf_model_low.state_dict()
    }
    with open(rf_models_save_path, 'wb') as f:
        pickle.dump(rf_models_state, f)
    print(f'Saved RFF models to {rf_models_save_path}')
    logger.info(f'Saved RFF models to {rf_models_save_path}')
    
    # Save metadata
    metadata = {
        'high_feature_dim': high_feature_dim,
        'low_feature_dim': low_feature_dim,
        'rf_dim_high': args.rf_dim_high,
        'rf_dim_low': args.rf_dim_low,
        'rbf_gamma_high': args.rbf_gamma_high,
        'rbf_gamma_low': args.rbf_gamma_low,
        'rf_type': args.rf_type,
        'rf_seed': args.rf_seed,
        'num_classes': args.num_classes,
        'num_clients': len(idxs_users)
    }
    metadata_save_path = os.path.join(save_dir, 'metadata.pkl')
    with open(metadata_save_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f'Saved metadata to {metadata_save_path}')
    logger.info(f'Saved metadata to {metadata_save_path}')



if __name__ == '__main__':
    args = args_parser()

    import secrets
    # 如果种子为默认值，自动生成随机种子
    if args.seed == 1234:  # 默认值
        args.seed = secrets.randbelow(2**31)
    if args.rf_seed == 42:  # 默认值
        args.rf_seed = secrets.randbelow(2**31)
        
    exp_details(args)
    # 如果用户指定了自定义 logdir,使用它;否则自动生成
    if args.log_dir is not None:
        logdir = args.log_dir
    else:
        logdir = os.path.join('../newresults', args.alg, str(datetime.datetime.now().strftime("%Y-%m-%d/%H.%M.%S"))+'_'+args.dataset+'_n'+str(args.ways))
    mkdirs(logdir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(logdir, 'log.log'),
        format='[%(levelname)s](%(asctime)s) %(message)s',
        datefmt='%Y/%m/%d/ %I:%M:%S %p', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    print("**Basic Setting...")
    logger.info("**Basic Setting...")
    print('  ', args)
    logging.info(args)

    summary_writer = SummaryWriter(logdir)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)# Minimum 2 classes; cannot exceed the total number of classes  List of the number of classes owned by each client: [,,,]
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset =='cifar100':
        k_list = np.random.randint(args.shots- args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset=='tinyimagenet':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'realwaste':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'flowers':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'defungi' or args.dataset == 'fashion':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'imagenet':  # The number of samples in the category with the fewest samples in the training set is 732, and the number of samples in the category with the fewest samples in the test set is 50.
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)

    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)
    # user_groups: dictionary where
    #   - key = client ID
    #   - value = ndarray of selected sample IDs for the client’s chosen classes (class IDs sorted in ascending order)
    # user_groups_lt: test set sample ID dictionary
    # classes_list: list of lists representing the classes assigned to each client

    # Build models
    local_model_list = []
    for i in range(args.num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20

            local_model = CNNMnist(args=args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i<7:
                    args.out_channels = 18
                elif i>=7 and i<14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar10' or args.dataset=='cifar100' or args.dataset == 'flowers'  or args.dataset == 'defungi' :
            local_model = CNNCifar(args=args)
        elif args.dataset=='tinyimagenet':
            args.num_classes = 200
            local_model = ModelCT(out_dim=256, n_classes=args.num_classes)
        elif args.dataset=='realwaste':
            local_model = CNNCifar(args=args)
        elif args.dataset=='fashion':
            local_model=CNNFashion_Mnist(args=args)
        elif args.dataset == 'imagenet':
            local_model = ResNetWithFeatures(base='resnet18', num_classes=args.num_classes)

        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)


    if args.alg=='fedavg':
        Fedavg(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, summary_writer,logger,logdir)
    elif args.alg=='fedprox':
        Fedprox(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,logdir)
    elif args.alg=='moon':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'femnist':
            global_model = CNNFemnist(args=args)
        elif args.dataset == 'cifar10' or args.dataset=='realwaste' or args.dataset == 'flowers' or args.dataset == 'defungi':
            global_model = CNNCifar(args=args)
        elif args.dataset=='cifar100':
            args.num_classes = 100
            local_model = ModelCT( out_dim=256, n_classes=args.num_classes)
        elif args.dataset=='tinyimagenet':
            args.num_classes = 200
            local_model = ModelCT(out_dim=256, n_classes=args.num_classes)
        elif args.dataset=='fashion':
            global_model=CNNFashion_Mnist(args=args)
        elif args.dataset == 'imagenet':
            global_model = ResNetWithFeatures(base='resnet18')
        global_model.to(args.device)
        global_model.train()
        Moon(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,global_model,logger,summary_writer,logdir)
    elif args.alg == 'fedntd':
        fedntd(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, summary_writer, logger, logdir)
    elif args.alg == 'fedgkd':
        fedgkd(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, logdir)
    elif args.alg=='fedproc':
        Fedproc(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    elif args.alg=='fedproto':
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list, summary_writer,logger,logdir)
    elif args.alg=='ours':#FedMPS
        FedMPS(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,summary_writer,logger,logdir)



