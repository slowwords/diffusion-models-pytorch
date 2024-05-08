from noise_predict_models import Unet
from omegaconf import OmegaConf
from schedulers import (
    DDPM,
    DDIM,
)
import torch
import argparse
import os
import torchvision

def load_scheduler(method: str = 'ddpm'):
    match method:
        case 'ddpm':
            return DDPM
        case 'ddim':
            return DDIM
        case _:
            raise KeyError(f'不支持采样方法{method}')
    
class Sampler:
    def __init__(self, config_path: str) -> None:
        self.configs = OmegaConf.load(config_path)
        self.image_size = self.configs.base.image_size
        self.in_channels = self.configs.base.in_channels
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
        self.Net = None
    
    def __call__(self, scheduler_method: str, scheduler_steps: int, batch_size: int, return_list: bool = False) -> list[any]:
        Scheduler = load_scheduler(scheduler_method)
        self.Net = Scheduler(
            schedule_name=self.schedule_name,
            timesteps=self.timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            denoise_model=self.denoise_model
        ).to(self.device)
        if self.configs.inference.best_model is None:
            raise ValueError(f'配置文件{self.configs}中没有提供inference.best_model')
        else:
            self.Net.load_state_dict(torch.load(self.configs.inference.best_model, map_location=self.device))
        samples = self.Net.inference(image_size=self.image_size, batch_size=batch_size, channels=self.in_channels, scheduler_steps=scheduler_steps)
        if return_list:
            return samples
        else:
            return samples[-1]
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion Model Inference')
    parser.add_argument('--config_path', type=str, default='./configs/CelebA.yaml')
    parser.add_argument('--out_path', '-o', type=str, default='./outs/CelebA')
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    sampler = Sampler(config_path=args.config_path)
    # samples = sampler('ddpm', 1000, 4)
    samples = sampler('ddim', 50, 4)
    if isinstance(samples, list):
        for idx, sample in enumerate(samples):
            # sample = torch.from_numpy(sample)
            torchvision.utils.save_image(
                sample,
                f'{args.out_path}/{idx}.jpg',
                normalize=True,
                value_range=(-1, 1),
            )
    else:
        # sample = torch.from_numpy(samples)
        torchvision.utils.save_image(
            samples,
            f'{args.out_path}/result_ddim.jpg',
            normalize=True,
            value_range=(-1, 1),
        )