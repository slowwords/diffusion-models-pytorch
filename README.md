# Diffusion Models

## Run

### Train
```shell
python train.py
```

### Inference
```shell
python inference.py
```

## Result

steps = 0

![alt text](imgs/celeba_64_step=0.jpg)

steps = 500

![alt text](imgs/celeba_64_step=500.jpg)

steps = 750

![alt text](imgs/celeba_64_step=750.jpg)

steps = 1000

![alt text](imgs/celeba_64_step=1000.jpg)

## Different Scheduler Results

### DDPM

steps = 1000

![alt text](imgs/result_ddpm.jpg)

### DDIM

steps = 50

![alt text](imgs/result_ddim.jpg)

## Model Structure (s sample)

```Markdown
DiffusionModel(
  (denoise_model): Unet(
    (init_conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
    (time_mlp): Sequential(
      (0): SinusoidalPositionEmbeddings()
      (1): Linear(in_features=128, out_features=512, bias=True)
      (2): GELU(approximate='none')
      (3): Linear(in_features=512, out_features=512, bias=True)
    )
    (downs): ModuleList(
      (0): ModuleList(
        (0-1): 2 x ResNetBlock(
          (mlp): Sequential(
            (0): SiLU()
            (1): Linear(in_features=512, out_features=256, bias=True)
          )
          (block1): Block(
            (proj): WeightStandardizedConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 128, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (block2): Block(
            (proj): WeightStandardizedConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 128, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (res_conv): Identity()
        )
        (2): Residual(
          (fn): PreNorm(
            (fn): LinearAttention(
              (to_qkv): Conv2d(128, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (to_out): Sequential(
                (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
                (1): GroupNorm(1, 128, eps=1e-05, affine=True)
              )
            )
            (norm): GroupNorm(1, 128, eps=1e-05, affine=True)
          )
        )
        (3): Sequential(
          (0): Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2)
          (1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (1): ModuleList(
        (0-1): 2 x ResNetBlock(
          (mlp): Sequential(
            (0): SiLU()
            (1): Linear(in_features=512, out_features=512, bias=True)
          )
          (block1): Block(
            (proj): WeightStandardizedConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 256, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (block2): Block(
            (proj): WeightStandardizedConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 256, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (res_conv): Identity()
        )
        (2): Residual(
          (fn): PreNorm(
            (fn): LinearAttention(
              (to_qkv): Conv2d(256, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (to_out): Sequential(
                (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
                (1): GroupNorm(1, 256, eps=1e-05, affine=True)
              )
            )
            (norm): GroupNorm(1, 256, eps=1e-05, affine=True)
          )
        )
        (3): Sequential(
          (0): Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2)
          (1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (2): ModuleList(
        (0-1): 2 x ResNetBlock(
          (mlp): Sequential(
            (0): SiLU()
            (1): Linear(in_features=512, out_features=1024, bias=True)
          )
          (block1): Block(
            (proj): WeightStandardizedConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 512, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (block2): Block(
            (proj): WeightStandardizedConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 512, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (res_conv): Identity()
        )
        (2): Residual(
          (fn): PreNorm(
            (fn): LinearAttention(
              (to_qkv): Conv2d(512, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (to_out): Sequential(
                (0): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
                (1): GroupNorm(1, 512, eps=1e-05, affine=True)
              )
            )
            (norm): GroupNorm(1, 512, eps=1e-05, affine=True)
          )
        )
        (3): Sequential(
          (0): Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2)
          (1): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (3): ModuleList(
        (0-1): 2 x ResNetBlock(
          (mlp): Sequential(
            (0): SiLU()
            (1): Linear(in_features=512, out_features=2048, bias=True)
          )
          (block1): Block(
            (proj): WeightStandardizedConv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 1024, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (block2): Block(
            (proj): WeightStandardizedConv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 1024, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (res_conv): Identity()
        )
        (2): Residual(
          (fn): PreNorm(
            (fn): LinearAttention(
              (to_qkv): Conv2d(1024, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (to_out): Sequential(
                (0): Conv2d(128, 1024, kernel_size=(1, 1), stride=(1, 1))
                (1): GroupNorm(1, 1024, eps=1e-05, affine=True)
              )
            )
            (norm): GroupNorm(1, 1024, eps=1e-05, affine=True)
          )
        )
        (3): Conv2d(1024, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (ups): ModuleList(
      (0): ModuleList(
        (0-1): 2 x ResNetBlock(
          (mlp): Sequential(
            (0): SiLU()
            (1): Linear(in_features=512, out_features=4096, bias=True)
          )
          (block1): Block(
            (proj): WeightStandardizedConv2d(3072, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 2048, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (block2): Block(
            (proj): WeightStandardizedConv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 2048, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (res_conv): Conv2d(3072, 2048, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): Residual(
          (fn): PreNorm(
            (fn): LinearAttention(
              (to_qkv): Conv2d(2048, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (to_out): Sequential(
                (0): Conv2d(128, 2048, kernel_size=(1, 1), stride=(1, 1))
                (1): GroupNorm(1, 2048, eps=1e-05, affine=True)
              )
            )
            (norm): GroupNorm(1, 2048, eps=1e-05, affine=True)
          )
        )
        (3): Sequential(
          (0): Upsample(scale_factor=2.0, mode='nearest')
          (1): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (1): ModuleList(
        (0-1): 2 x ResNetBlock(
          (mlp): Sequential(
            (0): SiLU()
            (1): Linear(in_features=512, out_features=2048, bias=True)
          )
          (block1): Block(
            (proj): WeightStandardizedConv2d(1536, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 1024, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (block2): Block(
            (proj): WeightStandardizedConv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 1024, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (res_conv): Conv2d(1536, 1024, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): Residual(
          (fn): PreNorm(
            (fn): LinearAttention(
              (to_qkv): Conv2d(1024, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (to_out): Sequential(
                (0): Conv2d(128, 1024, kernel_size=(1, 1), stride=(1, 1))
                (1): GroupNorm(1, 1024, eps=1e-05, affine=True)
              )
            )
            (norm): GroupNorm(1, 1024, eps=1e-05, affine=True)
          )
        )
        (3): Sequential(
          (0): Upsample(scale_factor=2.0, mode='nearest')
          (1): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (2): ModuleList(
        (0-1): 2 x ResNetBlock(
          (mlp): Sequential(
            (0): SiLU()
            (1): Linear(in_features=512, out_features=1024, bias=True)
          )
          (block1): Block(
            (proj): WeightStandardizedConv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 512, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (block2): Block(
            (proj): WeightStandardizedConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 512, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (res_conv): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): Residual(
          (fn): PreNorm(
            (fn): LinearAttention(
              (to_qkv): Conv2d(512, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (to_out): Sequential(
                (0): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
                (1): GroupNorm(1, 512, eps=1e-05, affine=True)
              )
            )
            (norm): GroupNorm(1, 512, eps=1e-05, affine=True)
          )
        )
        (3): Sequential(
          (0): Upsample(scale_factor=2.0, mode='nearest')
          (1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (3): ModuleList(
        (0-1): 2 x ResNetBlock(
          (mlp): Sequential(
            (0): SiLU()
            (1): Linear(in_features=512, out_features=512, bias=True)
          )
          (block1): Block(
            (proj): WeightStandardizedConv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 256, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (block2): Block(
            (proj): WeightStandardizedConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm): GroupNorm(4, 256, eps=1e-05, affine=True)
            (act): SiLU()
          )
          (res_conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): Residual(
          (fn): PreNorm(
            (fn): LinearAttention(
              (to_qkv): Conv2d(256, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (to_out): Sequential(
                (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
                (1): GroupNorm(1, 256, eps=1e-05, affine=True)
              )
            )
            (norm): GroupNorm(1, 256, eps=1e-05, affine=True)
          )
        )
        (3): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (mid_block1): ResNetBlock(
      (mlp): Sequential(
        (0): SiLU()
        (1): Linear(in_features=512, out_features=4096, bias=True)
      )
      (block1): Block(
        (proj): WeightStandardizedConv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): GroupNorm(4, 2048, eps=1e-05, affine=True)
        (act): SiLU()
      )
      (block2): Block(
        (proj): WeightStandardizedConv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): GroupNorm(4, 2048, eps=1e-05, affine=True)
        (act): SiLU()
      )
      (res_conv): Identity()
    )
    (mid_attn): Residual(
      (fn): PreNorm(
        (fn): Attention(
          (to_qkv): Conv2d(2048, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (to_out): Conv2d(128, 2048, kernel_size=(1, 1), stride=(1, 1))
        )
        (norm): GroupNorm(1, 2048, eps=1e-05, affine=True)
      )
    )
    (mid_block2): ResNetBlock(
      (mlp): Sequential(
        (0): SiLU()
        (1): Linear(in_features=512, out_features=4096, bias=True)
      )
      (block1): Block(
        (proj): WeightStandardizedConv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): GroupNorm(4, 2048, eps=1e-05, affine=True)
        (act): SiLU()
      )
      (block2): Block(
        (proj): WeightStandardizedConv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): GroupNorm(4, 2048, eps=1e-05, affine=True)
        (act): SiLU()
      )
      (res_conv): Identity()
    )
    (final_res_block): ResNetBlock(
      (mlp): Sequential(
        (0): SiLU()
        (1): Linear(in_features=512, out_features=256, bias=True)
      )
      (block1): Block(
        (proj): WeightStandardizedConv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): GroupNorm(4, 128, eps=1e-05, affine=True)
        (act): SiLU()
      )
      (block2): Block(
        (proj): WeightStandardizedConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): GroupNorm(4, 128, eps=1e-05, affine=True)
        (act): SiLU()
      )
      (res_conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    )
    (final_conv): Conv2d(128, 3, kernel_size=(1, 1), stride=(1, 1))
  )
)
```