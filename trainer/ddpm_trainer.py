from .base_trainer import *

class DDPM_Trainer(TrainerBase):
    def __init__(
            self,
            start_epoch: int = None,
            epoches: int = None,
            train_loader: any = None,
            optimizer: any = None,
            device: torch.device = None,
            IFEarlyStopping: bool = False,
            IFadjust_learning_rate: bool = False,
            **kwargs
        ):
        super().__init__(
            start_epoch,
            epoches, 
            train_loader, 
            optimizer, 
            device, 
            IFEarlyStopping, 
            IFadjust_learning_rate, 
            **kwargs
        )

        if "timesteps" in kwargs.keys():
            self.timesteps = kwargs["timesteps"]
        else:
            raise ValueError("To train DDPM model, timesteps is necessary!")
        
    def forward(self, model, *args, **kwargs):

        for i in range(self.start_epoch, self.epoches):
            losses = []
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for step, (features, labels) in loop:
                features = features.to(self.device)
                batch_size = features.shape[0]
                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

                loss = model.train(x_start=features, t=t, loss_type=kwargs['loss_type'])
                losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update information
                loop.set_description(f'Epoch [{i}/{self.epoches}]')
                loop.set_postfix(loss=loss.item())

            # save checkpoint every epoch
            if "model_save_path" in kwargs.keys():
                self.save_best_model(model=model, path=kwargs["model_save_path"])

        return model