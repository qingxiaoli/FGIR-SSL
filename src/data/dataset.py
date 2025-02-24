import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FGIRDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: Root directory containing all images (or subdirectories)
        :param transform: Image pre-processing operations
        """
        self.data_dir = data_dir
        self.image_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return {'image': image}
