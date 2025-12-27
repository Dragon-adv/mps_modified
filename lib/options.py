#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.04,
                        help='the fraction of clients: C')
    parser.add_argument('--train_ep', type=int, default=1,
                        help="the number of local episodes: E")
    parser.add_argument('--local_bs', type=int, default=8,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--alg', type=str, default='fedproto', help="algorithms")
    parser.add_argument('--mode', type=str, default='task_heter', help="mode")
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--data_dir', type=str, default='../data/', help="directory of dataset")
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--test_ep', type=int, default=10, help="num of test episodes for evaluation")

    # Local arguments
    parser.add_argument('--ways', type=int, default=3, help="num of classes")
    parser.add_argument('--shots', type=int, default=100, help="num of shots")
    parser.add_argument('--train_shots_max', type=int, default=110, help="num of shots")
    parser.add_argument('--test_shots', type=int, default=15, help="num of shots")
    parser.add_argument('--stdev', type=int, default=2, help="stdev of ways")
    parser.add_argument('--ld', type=float, default=1, help="fedproto: weight of proto loss")
    parser.add_argument('--ft_round', type=int, default=10, help="round of fine tuning")
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--model_buffer_size', type=int, default=1,help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--T', type=float, default=5, help='the temperature parameter for soft label')
    parser.add_argument('--alph', type=float, default=1, help='weight of high proto contrastive loss')# Note that the weights of the various losses here differ from those in the article
    parser.add_argument('--beta', type=float, default=1, help='weight of low proto contrastive loss')
    parser.add_argument('--gama', type=float, default=0, help='weight of logits loss')

    parser.add_argument('--Dbeta', type=float, default=0.5, help="diri distribution parameter")

    # for fedgkd
    parser.add_argument("--gkd_temperature", default=1, type=float)
    parser.add_argument("--distillation_coefficient", default=0.1, type=float)
    parser.add_argument("--buffer_length", default=5, type=int)

    # for fedntd
    parser.add_argument('--ntd_tau', type=float, default=1, help='for fedntd, the temperature of distilling')
    parser.add_argument('--ntd_beta', type=float, default=1, help='for fedntd, the wight of ntdloss')

    # for SFD statistics aggregation (RFF parameters)
    parser.add_argument('--rf_seed', type=int, default=42, help='random seed for RFF model initialization')
    parser.add_argument('--rf_dim_high', type=int, default=3000, help='RFF dimension for high-level features')
    parser.add_argument('--rf_dim_low', type=int, default=3000, help='RFF dimension for low-level features')
    parser.add_argument('--rbf_gamma_high', type=float, default=0.01, help='RBF gamma parameter for high-level RFF model')
    parser.add_argument('--rbf_gamma_low', type=float, default=0.01, help='RBF gamma parameter for low-level RFF model')
    parser.add_argument('--rf_type', type=str, default='orf', help='RFF type: orf or iid')
    parser.add_argument('--stats_level', type=str, default='high', 
                        choices=['high', 'low', 'both'],
                        help='Statistics computation level: high (only high-level), low (only low-level), or both (default: high)')

    args = parser.parse_args()
    return args
