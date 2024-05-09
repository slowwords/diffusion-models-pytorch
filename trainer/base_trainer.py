from utils import *

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoints.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class TrainerBase(nn.Module):
    def __init__(
            self,
            start_epoch,
            epoches,
            train_loader,
            optimizer,
            device,
            IFEarlyStopping,
            IFadjust_learning_rate,
            **kwargs
        ):
        super(TrainerBase, self).__init__()
        self.start_epoch = start_epoch
        self.epoches = epoches
        if self.epoches is None:
            raise ValueError("Please provide total epoches!")

        self.train_loader = train_loader
        if self.train_loader is None:
            raise ValueError("Please provide train_loader!")

        self.optimizer = optimizer
        if self.optimizer is None:
            raise ValueError("Please provide optimizer!")

        self.device = device
        if self.device is None:
            raise ValueError("Please provide device!")

        # Judge by factor as follow if earlystopping was used.
        self.IFEarlyStopping = IFEarlyStopping
        if IFEarlyStopping:
            if "patience" in kwargs.keys():
                self.early_stopping = EarlyStopping(patience=kwargs["patience"], verbose=True)
            else:
                raise ValueError("Parameter {patience=intX} must be provide if early stopping was used!")

            if "val_loader" in kwargs.keys():
                self.val_loader = kwargs["val_loader"]
            else:
                raise ValueError("Val_loader must be provide if early stopping was used!")

        # Judge by factor as follow if adjust lr was used.
        self.IFadjust_learning_rate = IFadjust_learning_rate
        if IFadjust_learning_rate:
            if "types" in kwargs.keys():
                self.types = kwargs["types"]
                if "lr_adjust" in kwargs.keys():
                    self.lr_adjust = kwargs["lr_adjust"]
                else:
                    self.lr_adjust = None
            else:
                raise ValueError("Learning rate adjustment policy parameter types must be choiced from 'type1' or 'type2'")

    def adjust_learning_rate(self, epoch, learning_rate):
        # lr = args.learning_rate * (0.2 ** (epoch // 2))
        if self.types == 'type1':
            lr_adjust = {epoch: learning_rate * (0.1 ** ((epoch - 1) // 10))}  # 每10个epoch,学习率缩小10倍
        elif self.types == 'type2':
            if self.lr_adjust is not None:
                lr_adjust = self.lr_adjust
            else:
                lr_adjust = {
                    5: 1e-4, 10: 5e-5, 20: 1e-5, 25: 5e-6,
                    30: 1e-6, 35: 5e-7, 40: 1e-8
                }
        else:
            raise ValueError("Please provide learning rate adjustment policy parameter types from [type1, type2]")

        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    @staticmethod
    def save_best_model(model, path):
        torch.save(model.state_dict(), path+'/'+'BestModel.pth')
        print(f"The checkpoint has been saved to {str(path)} with type '.pth' successful!")

    def forward(self, model, *args, **kwargs):

        pass