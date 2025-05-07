import torch
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from config import VAEConfig

class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        super().__init__()
        self.image_folder = folder_path
        self.images = [img for img in os.listdir(folder_path) if img.lower().endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder, self.images[index])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize(VAEConfig.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(0.2)
])


full_dataset = ImageDataset(VAEConfig.folder_path, transform)

total_size = len(full_dataset)
train_size = int(VAEConfig.train_split * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(
    full_dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42)  
)

train_dataloader = DataLoader(train_dataset, batch_size=VAEConfig.batch, 
                              shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)
test_dataloader = DataLoader(test_dataset, batch_size=VAEConfig.batch, num_workers=3, 
                             pin_memory=True, persistent_workers=True)

if __name__ == "__main__":
    for batch in train_dataloader:
        print(batch[0].max(), batch[0].min())
        break