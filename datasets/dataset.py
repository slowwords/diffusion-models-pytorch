from torch.utils.data import Dataset
import os
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

IMG_EXTS = ['jpg', 'png', 'jpeg']

def is_image_file(img_path):
    for img_ext in IMG_EXTS:
        if img_path.lower().endswith(img_ext):
            return True
    return False

class DMDataset(Dataset):
    def __init__(self, path: str, load_size: int = 256, max_dataset_size: float = float("inf")) -> None:
        super().__init__()
        self.images = self.make_dataset(path, max_dataset_size)
        self.number_image = len(self.images)
        self.trans = self.image_transforms(load_size=load_size)

    def make_dataset(self, path: str, max_dataset_size: float = float("inf")) -> list[str]:
        images = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

        images = sorted(images)
        return images[:min(max_dataset_size, len(images))]

    def image_transforms(self, load_size: int, mode: str = "resize", p: float = 0.5) -> any:
        if mode == "resize":
            return transforms.Compose([
                transforms.Resize([load_size, load_size]),
                transforms.RandomHorizontalFlip(p=p),  # 以0.5的概率水平翻转
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif mode == "centercrop":
            return transforms.Compose([
                transforms.CenterCrop(size=(load_size, load_size)),
                transforms.RandomHorizontalFlip(p=p),  # 以0.5的概率水平翻转
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
    def __getitem__(self, index) -> Image.Image:
        
        label = np.array([0])
        image = Image.open(self.images[index % self.number_image]).convert('RGB')
        image = self.trans(image)

        return image, label
    
    def __len__(self):
        
        return self.number_image