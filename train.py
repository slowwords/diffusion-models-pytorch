import os
import numpy as np
import torchvision
import torch
from torch.optim import Adam
from utils import *
from datasets import DMDataset
from noise_predict_models import Unet
from trainer import DDPM_Trainer
from schedulers import DDPM
from omegaconf import OmegaConf
import argparse

class Trainer:
    def __init__(self, config_path: str) -> None:
        self.configs = OmegaConf.load(config_path)
        
        self.image_size = self.configs.base.image_size
        self.data = DMDataset(self.configs.train.data_path, self.image_size)
        self.in_channels = self.configs.base.in_channels
        self.batch_size = self.configs.train.batch_size
        self.data_loader = torch.utils.data.DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.configs.train.num_workers,
        )
        self.device = self.configs.base.device
        self.dim_mults = self.configs.base.dim_mults
        self.denoise_model = Unet(
            dim_in=self.image_size,
            channels=self.in_channels,
            dim_mults=self.dim_mults
        )
        self.timesteps = self.configs.base.timesteps
        self.schedule_name = self.configs.base.schedule_name
        self.beta_start = self.configs.base.beta_start
        self.beta_end = self.configs.base.beta_end
        self.Net = DDPM(
            schedule_name=self.schedule_name,
            timesteps=self.timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            denoise_model=self.denoise_model
        ).to(self.device)
        if self.configs.train.pre_trained_model is not None:
            self.Net.load_state_dict(torch.load(self.configs.train.pre_trained_model))
        # optimizer
        self.lr = self.configs.train.lr
        self.optimizer = Adam(self.Net.parameters(), lr=self.lr)
        self.epoches = self.configs.train.epoches
        self.start_epoch = self.configs.train.start_epoch

        self.trainer = DDPM_Trainer(
            start_epoch=self.start_epoch,
            epoches=self.epoches,
            train_loader=self.data_loader,
            optimizer=self.optimizer,
            device=self.device,
            timesteps=self.timesteps,
            IFEarlyStopping=self.configs.train.IFEarlyStopping,
            IFadjust_learning_rate=self.configs.train.IFadjust_learning_rate,
            **{'patience': self.configs.train.patience, 
               "types": self.configs.train.lr_adjust_type,
               },
        )

        self.ckpts_path = self.configs.train.ckpts_path
        setting = "imageSize{}_channels{}_dimMults{}_timeSteps{}_scheduleName{}".format(self.image_size, self.in_channels, self.dim_mults, self.timesteps, self.schedule_name)
        self.save_path = os.path.join(self.ckpts_path, setting)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __call__(self):
        self.Net = self.trainer(self.Net, model_save_path=self.save_path, loss_type=self.configs.train.loss_type)
    
def main(args):
    DDPM_Trainer = Trainer(config_path=args.config_path)
    DDPM_Trainer()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPM')
    parser.add_argument('--config_path', type=str, default='./configs/CelebA.yaml')
    parser.add_argument('--out_path', '-o', type=str, default='./outs')
    args = parser.parse_args()
    main(args)
    