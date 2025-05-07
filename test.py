import torch
import torch.nn as nn
from config import VAEConfig
from model import VAE
from data import test_dataloader
import random
import matplotlib.pyplot as plt
import os

#setting up the things
os.makedirs("comparison_images", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
config = VAEConfig()
vae = VAE(config=config).to(device)
vae.eval()
vae.load_state_dict(torch.load("checkpoints/model_dict56400.pth", weights_only=True))

random_indices = random.sample(range(len(test_dataloader.dataset)), k=VAEConfig.batch)
random_batch = torch.stack([test_dataloader.dataset[i] for i in random_indices])
batch = random_batch[:5].to(device)

with torch.no_grad():
    x_const, _, _ = vae(batch)

original_images = batch.cpu()
generated_images = x_const.cpu()


fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):

    axes[0, i].imshow(original_images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)
    axes[0, i].set_title(f"Original {i}")
    axes[0, i].axis("off")

    axes[1, i].imshow(generated_images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)
    axes[1, i].set_title(f"Reconstructed {i}")
    axes[1, i].axis("off")


random_idx = random.randint(1,100)
plt.tight_layout()
plt.savefig(f"comparison_images/original_vs_generated{random_idx}.png")
plt.close()

print("Comparison image saved successfully!")
