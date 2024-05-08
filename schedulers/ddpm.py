from utils import *
from .variance_schedule import VarianceSchedule

class DDPM(nn.Module):
    def __init__(
        self,
        schedule_name="linear_beta_schedule",
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        denoise_model=None
    ):
        super(DDPM, self).__init__()

        self.denoise_model = denoise_model

        # 加噪过程
        # 方差生成策略，DDPM中采样线性策略
        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name, beta_start=beta_start, beta_end=beta_end)
        self.timesteps = timesteps
        # 生成与加噪步长T对应的方差序列
        self.betas = variance_schedule_func(timesteps)  # torch.Size([1000])
        # 定义alpha序列
        self.alphas = 1. - self.betas   # alpha_t = 1 - beta_t
        # 定义alpha_bar: alpha_bar是alpha_t到alpha_i的叠乘
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        # t=T时，alpha_bar应等于1
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)
        # 定义1/sqrt(alpha)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # x_t  = sqrt(alphas_bar)*x_0 + sqrt(1 - alphas_bar)*z_t
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # 这里用的不是简化后的方差而是算出来的
        self.posterior_variance = self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)

    def sample_forward(self, x_start, t, noise=None):
        # forward diffusion (using the nice property)
        # x_t  = sqrt(alphas_bar)*x_0 + sqrt(1 - alphas_bar)*z_t
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_bar_t = extract(self.sqrt_alphas_bar, t, x_start.shape)
        sqrt_one_minus_alphas_bar_t = extract(
            self.sqrt_one_minus_alphas_bar, t, x_start.shape
        )
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * z_t
        return sqrt_alphas_bar_t * x_start + sqrt_one_minus_alphas_bar_t * noise

    def compute_loss(self, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.sample_forward(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def sample_backward_step(self, x, t, t_index):
        """
        去噪流程：
        1. 在每一个时间步t，利用深度网络根据x_t和t得到随机噪声z_t的预测值z_theta(x_t, t)；
        2. 计算均值mu_theta(x_t, t)和方差beta_theta(x_t, t)：
            mu_theta(x_t, t) = 1 / sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_t) * noise_t)
            beta_theta(x_t, t) = (1 - alpha_bar_{t-1}) / (1 - alpha_bar) * beta_t
        3. 利用重参数技巧得到x_{t-1} = mu_theta(x_t, t) + sqrt(\beta_theta(x_t, t) * noise_t) ~ N(0, I);
        4. 重复1~3步骤得到x_0。
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_bar_t = extract(
            self.sqrt_one_minus_alphas_bar, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_bar_t
        )
        """
        mean = (x_t - (1 - alpha_t) / sqrt(1 - alpha_bar_t))
        """

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_backward(self, shape, scheduler_steps):
        device = next(self.denoise_model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, scheduler_steps)), desc='DDPM sampling loop time step', total=scheduler_steps):
            img = self.sample_backward_step(img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu())
        return imgs

    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3, scheduler_steps=1000):
        return self.sample_backward(shape=(batch_size, channels, image_size, image_size), scheduler_steps=scheduler_steps)

    def train(self, **kwargs):
        # 先判断必须参数
        if "x_start" and "t" in kwargs.keys():
            # 接下来判断一些非必选参数
            if "loss_type" and "noise" in kwargs.keys():
                return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"],
                                            noise=kwargs["noise"], loss_type=kwargs["loss_type"])
            elif "loss_type" in kwargs.keys():
                return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"], loss_type=kwargs["loss_type"])
            elif "noise" in kwargs.keys():
                return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"], noise=kwargs["noise"])
            else:
                return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"])

        else:
            raise ValueError("扩散模型在训练时必须传入参数x_start和t！")
    
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
