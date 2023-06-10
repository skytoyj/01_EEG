'''
Author: Yin Jin
Date: 2022-03-08 19:38:55
LastEditTime: 2023-06-10 17:58:41
LastEditors: JinYin
'''

import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--noise_type', type=str, default='EMG')
    parser.add_argument('--EEG_path', type=str, default='./data/EEG.npy')
    parser.add_argument('--NOS_path', type=str, default='./data/EOG.npy')
    parser.add_argument('--denoise_network', type=str, default='FCNN')
    parser.add_argument('--save_path', type=str, default='./result/')
    parser.add_argument('--depth', type=float, default=6)
    parser.add_argument('--feature_num', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_result', type=bool, default=True)
    opts = parser.parse_args()
    return opts
