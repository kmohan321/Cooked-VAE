import torch
import random
import torch.nn as nn
import torch.optim as optimizer
import wandb
import lpips
from dataclasses import asdict 
from torch.amp.grad_scaler import GradScaler
from model import VAE
from config import VAEConfig
from tqdm.auto import tqdm
from data import train_dataloader, test_dataloader

def main():
    #setting up the things
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = VAEConfig()
    scaler = GradScaler(device=device)
    vae = VAE(config).to(device)
    model_parameters = sum([p.numel() for p in vae.parameters()])
    run = wandb.init(
        project="VAE",
        config={**asdict(config), "model_parameters":model_parameters}
    )

    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lr = config.lr
    epochs = config.epochs
    beta = config.beta
    model_optimizer = optimizer.AdamW(vae.parameters(), lr = lr)

    def vae_loss(x: torch.Tensor, x_const: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor):

        reconst_loss = loss_fn(x_const, x)
        kl_loss = - 0.5 * torch.mean(
                1 + torch.log(variance) - mean**2 - variance        
        )
        return reconst_loss, kl_loss

    global_steps = 0
    for epoch in range(epochs):
        vae.train()
        for batch in tqdm(train_dataloader,desc=f"epoch {epoch}/{epochs}"):
            global_steps += 1
            batch = batch.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                x_const, mu, var = vae(batch)
                reconst_loss, kl_loss = vae_loss(batch, x_const, mu, var)
                lpips_loss = lpips_fn(x_const, batch).mean()
                total_loss = reconst_loss + beta * kl_loss + lpips_loss #(-ELBO => -(-loss + kl) => (mse_loss + kl))
                
            run.log({"loss/total_loss": total_loss.item(), "loss/reconstr_loss":reconst_loss.item(), "loss/kl_loss":kl_loss.item()
                        , "lpips_loss": lpips_loss.item()},step=global_steps)

            model_optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(model_optimizer)
            scaler.update()

        if (epoch + 1) % 5 == 0:
            random_indices = random.sample(range(len(test_dataloader.dataset)), k=VAEConfig.batch)
            random_batch = torch.stack([test_dataloader.dataset[i] for i in random_indices])
            batch = random_batch[:5].to(device)
            vae.eval()
            with torch.no_grad():
                x_const, _, _ = vae(batch[:5])
            run.log({
                "constructed_images": wandb.Image(x_const.cpu() * 0.5 + 0.5),
                "original_images": wandb.Image(batch[:5].cpu() * 0.5 + 0.5)
            }, step=global_steps)
            torch.save(vae.state_dict(), f"checkpoints/model_dict{global_steps}.pth")

if __name__ == "__main__":
    main()