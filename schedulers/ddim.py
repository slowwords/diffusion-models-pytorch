from .ddpm import DDPM
import numpy as np
from tqdm.auto import tqdm
from utils import *

class DDIM(DDPM):
    def __init__(
        self,
        schedule_name="linear_beta_schedule",
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        denoise_model=None
    ):
        super().__init__(
            schedule_name=schedule_name,
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            denoise_model=denoise_model
        )
            
    def sample(self, image_size, batch_size=16, channels=3, scheduler_steps=20, simple_var=True):
        device = next(self.denoise_model.parameters()).device
        imgs = []
        x = torch.randn((batch_size, channels, image_size, image_size), device=device)
        if simple_var:
            eta = 1

        time_steps_mapping = np.linspace(0, self.timesteps, scheduler_steps+1).astype(np.int32)

        for i in tqdm(reversed(range(1, scheduler_steps+1)), desc='DDIM sampling loop time step', total=scheduler_steps):
            
            cur_t = time_steps_mapping[i] - 1
            prev_t = time_steps_mapping[i-1] - 1

            cur_t_pt = torch.full((batch_size,), cur_t, device=device, dtype=torch.long)
            prev_t_pt = torch.full((batch_size,), prev_t, device=device, dtype=torch.long)
            
            ab_cur = extract(self.alphas_bar, cur_t_pt, x.shape)
            if prev_t >= 0:
                ab_prev = extract(self.alphas_bar_prev, prev_t_pt, x.shape)
            else:
                ab_prev = torch.ones_like(x, dtype=torch.float32).to(device)

            eps = self.denoise_model(x, cur_t_pt)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x)

            first_term = (ab_prev / ab_cur) ** 0.5 * x
            second_term = ((1 - ab_prev - var) ** 0.5 - (ab_prev * (1 - ab_cur) / ab_cur) ** 0.5) * eps

            if simple_var:
                third_term = (1 - ab_cur / ab_prev) ** 0.5 * noise
            else:
                third_term = var ** 0.5 * noise
            
            x = first_term + second_term + third_term
            imgs.append(x.cpu())

        return imgs

    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, **kwargs):
        if "image_size" and "batch_size" and "channels" in kwargs.keys():
            return self.sample(image_size=kwargs["image_size"],
                                batch_size=kwargs["batch_size"],
                                channels=kwargs["channels"],
                                scheduler_steps=kwargs['scheduler_steps'])
        else:
            raise ValueError("扩散模型在生成图片时必须传入image_size, batch_size, channels等三个参数")
