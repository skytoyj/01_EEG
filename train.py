'''
Author: Yin Jin
Date: 2022-03-08 19:50:50
LastEditTime: 2023-06-10 18:08:10
LastEditors: JinYin
'''
import torch
import torch.optim as optim
import numpy as np
from tqdm import trange
from opts import get_opts
from audtorch.metrics.functional import pearsonr

import os
from models.CTNet import *
from loss import denoise_loss_mse
from torch.utils.tensorboard import SummaryWriter
import torch

from preprocess.DenoisenetPreprocess import *
from tools import pick_models

def cal_SNR(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    PS = np.sum(np.square(truth), axis=-1)  # power of signal
    PN = np.sum(np.square((predict - truth)), axis=-1)  # power of noise
    ratio = PS / PN
    return torch.from_numpy(10 * np.log10(ratio))
       
def train(opts, model, train_log_dir, val_log_dir, data_save_path, fold):
    EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data = load_data(opts.EEG_path, opts.NOS_path, fold)
    train_data = EEGwithNoise(EEG_train_data, NOS_train_data, opts.batch_size)
    val_data = EEGwithNoise(EEG_val_data, NOS_val_data, opts.batch_size)
    test_data = EEGwithNoise(EEG_test_data, NOS_test_data, opts.batch_size)
    
    if opts.denoise_network == 'FCNN':
        learning_rate = 0.0001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9), eps=1e-8)
    elif opts.denoise_network == 'SimpleCNN':
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.9), eps=1e-8)
    elif opts.denoise_network == 'ResCNN':
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.9), eps=1e-8)
    elif opts.denoise_network == 'NovelCNN':
        optimizer = optim.RMSprop(model.parameters(), lr=5e-5, alpha=0.99, eps=1e-8)
    elif opts.denoise_network == 'EEGDNet':
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.9), eps=1e-8)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.9), eps=1e-8)
    
    best_mse = 100
    if opts.save_result:
        train_summary_writer = SummaryWriter(train_log_dir)
        val_summary_writer = SummaryWriter(val_log_dir)
        f = open(data_save_path + "result.txt", "a+")
    
    for epoch in range(opts.epochs):
        model.train()
        losses = []
        for batch_id in trange(train_data.len()):
            x_t, y_t = train_data.get_batch(batch_id)
            x_t, y_t = torch.Tensor(x_t).to(opts.device).unsqueeze(dim=1), torch.Tensor(y_t).to(opts.device)
            p_t = model(x_t).view(x_t.shape[0], -1)
                
            loss = denoise_loss_mse(p_t, y_t)
    
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        train_data.shuffle()
        train_loss = torch.stack(losses).mean().item()
        if opts.save_result:
            train_summary_writer.add_scalar("Train loss", train_loss, epoch)
        
        # 验证集验证
        model.eval()
        losses = []
        for batch_id in range(val_data.len()):
            x_t, y_t = val_data.get_batch(batch_id)
            x_t, y_t = torch.Tensor(x_t).to(opts.device).unsqueeze(dim=1), torch.Tensor(y_t).to(opts.device)
            
            with torch.no_grad():
                p_t = model(x_t).view(x_t.shape[0], -1)
                loss = ((p_t - y_t) ** 2).mean(dim=-1).sqrt().detach()
                losses.append(loss)
        val_mse = torch.cat(losses, dim=0).mean().item()
        val_summary_writer.add_scalar("Val loss", val_mse, epoch)
        
        # 测试，返回rrmse
        model.eval()
        losses = []
        single_acc, single_snr = [], []
        clean_data, output_data, input_data = [], [], []
        
        for batch_id in range(test_data.len()):
            x_t, y_t = test_data.get_batch(batch_id)
            x_t, y_t = torch.Tensor(x_t).to(opts.device).unsqueeze(dim=1), torch.Tensor(y_t).to(opts.device)

            with torch.no_grad():
                p_t = model(x_t).view(x_t.shape[0], -1)
                loss = (((p_t - y_t) ** 2).mean(dim=-1).sqrt() / (y_t ** 2).mean(dim=-1).sqrt()).detach()
                losses.append(loss)
                single_acc.append(pearsonr(p_t, y_t))
                single_snr.append(cal_SNR(p_t, y_t))

            output_data.append(p_t.cpu().numpy()), clean_data.append(y_t.cpu().numpy()), input_data.append(x_t.cpu().numpy())
        test_rrmse = torch.cat(losses, dim=0).mean().item()
        sum_acc = torch.cat(single_acc, dim=0).mean().item()
        sum_snr = torch.cat(single_snr, dim=0).mean().item()
        val_summary_writer.add_scalar("test rrmse", test_rrmse, epoch)
        
        # 保存最好的结果
        if val_mse < best_mse:
            best_mse = val_mse
            
            best_acc = sum_acc
            best_snr = sum_snr
            best_rrmse = test_rrmse
            print("Save best result")
            f.write("Save best result \n")
            val_summary_writer.add_scalar("best rrmse", best_mse, epoch)
            if opts.save_result:
                np.save(f"{data_save_path}/best_input_data.npy", np.array(input_data))
                np.save(f"{data_save_path}/best_output_data.npy", np.array(output_data))
                np.save(f"{data_save_path}/best_clean_data.npy", np.array(clean_data))
                torch.save(model, f"{data_save_path}/best_{opts.denoise_network}.pth")

        print('epoch:{:3d}, train_loss:{:.4f}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(epoch, train_loss, test_rrmse, sum_acc, sum_snr))
        f.write('epoch: {:3d}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(epoch, test_rrmse, sum_acc, sum_snr) + "\n")

    with open(os.path.join('./json_file/Denoisenet/{}/{}.log'.format(opts.noise_type, opts.denoise_network)), 'a+') as fp:
        fp.write('fold:{}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(fold, best_rrmse, best_acc, best_snr) + "\n")
    
    if opts.save_result:
        np.save(f"{data_save_path}/last_input_data.npy", test_data.EEG_data)
        np.save(f"{data_save_path}/last_output_data.npy", np.array(output_data))
        np.save(f"{data_save_path}/last_clean_data.npy", np.array(clean_data))
        torch.save(model, f"{data_save_path}/last_{opts.denoise_network}.pth")

if __name__ == '__main__':
    opts = get_opts()
    np.random.seed(0)
    opts.epochs = 200
    opts.depth = 6
    
    opts.batch_size = 128
    opts.noise_type = 'EMG'      # FCNN(200), SimpleCNN(50), ResCNN(50), NovelCNN(100), CTNet(200)
    opts.denoise_network = 'CTNet'    
    
    # opts.EEG_path = "./04_Data/01_Dataset/01_EEGdenoisenet/EEG_all_epochs_512hz.npy"
    # opts.NOS_path = "./04_Data/01_Dataset/01_EEGdenoisenet/EMG_all_epochs_512hz.npy"
    opts.EEG_path = "./04_Data/01_Dataset/01_EEGdenoisenet/EEG_all_epochs.npy"
    opts.NOS_path = "./04_Data/01_Dataset/01_EEGdenoisenet/EMG_all_epochs.npy"
    opts.save_path = "./04_Data/2_Result/03_EMG/{}/".format(opts.denoise_network)
    
    print(opts)
    for fold in range(10):
        print(f"fold:{fold}")
        model = pick_models(opts, data_num=512)
        print(model)
        
        foldername = '{}_{}_{}_{}'.format(opts.denoise_network, opts.noise_type, opts.epochs, fold)
            
        train_log_dir = opts.save_path +'/'+foldername +'/'+ '/train'
        val_log_dir = opts.save_path +'/'+foldername +'/'+ '/test'
        data_save_path = opts.save_path +'/'+foldername +'/'
        
        if not os.path.exists(train_log_dir):
            os.makedirs(train_log_dir)
        
        if not os.path.exists(val_log_dir):
            os.makedirs(val_log_dir)
        
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)

        train(opts, model, train_log_dir, val_log_dir, data_save_path, fold)


