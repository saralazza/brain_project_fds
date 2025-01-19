import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import cv2

from utils.checkpoint import save_checkpoint, load_checkpoint_vae

class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            # First Convolutional Layer
            nn.Conv2d(1, 64, 3, 2, 1), #(1, 128, 128) -> (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Second Convolutional Layer
            nn.Conv2d(64, 128, 3, 2, 1),  # (64, 64, 64) -> (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Third Convolutional Layer
            nn.Conv2d(128, 256, 3, 2, 1), # (128, 32, 32) -> (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Fourth Convolutional Layer
            nn.Conv2d(256, 512, 3, 2, 1),  # (256, 16, 16) -> (512, 8, 8)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # Fifth Convolutional Layer
            nn.Conv2d(512, 512, 3, 2, 1),  # (512, 8, 8) -> (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # Sixth Convolutional Layer
            nn.Conv2d(512, 1024, 3, 2, 1),  # (512, 4, 4) -> (1024, 2, 2)
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(1024 * 2 * 2, latent_dim)      # Mean
        self.fc_logvar = nn.Linear(1024 * 2 * 2, latent_dim)  # Variance

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 2 * 2),
            nn.ReLU(True),
            nn.Unflatten(1, (1024, 2, 2))
        )
        self.model = nn.Sequential(
            # First Upsample Layer
            nn.Upsample(scale_factor=2),  # (1024, 2, 2) -> (1024, 4, 4)
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # (1024, 4, 4) -> (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Second Upsample Layer
            nn.Upsample(scale_factor=2),  # (512, 4, 4) -> (512, 8, 8)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512, 8, 8) -> (512, 8, 8)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Third Upsample Layer
            nn.Upsample(scale_factor=2),  # (512, 8, 8) -> (512, 16, 16)
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # (512, 16, 16) -> (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Fourth Upsample Layer
            nn.Upsample(scale_factor=2),  # (256, 16, 16) -> (256, 32, 32)
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # (256, 32, 32) -> (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU() ,
            
            # Fifth Upsample Layer
            nn.Upsample(scale_factor=2),  # (128, 32, 32) -> (128, 64, 64)
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # (128, 64, 64) -> (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final Upsample Layer
            nn.Upsample(scale_factor=2),  # (64, 64, 64) -> (64, 128, 128)
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # (64, 128, 128) -> (1, 128, 128)
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = self.fc(z)
        img = self.model(z)
        return img

class VAE(nn.Module):
    def __init__(self, latent_dim=512):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = BCE + KLD
    return loss

def train_loop_vae(device, data_loader, checkpoint_dir):
    save_interval = 1

    # Initialize the model on the device and the optimazer
    vae = VAE().to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    scheduler.step()
    
    # Loading checkpoints, if present
    start_epoch = load_checkpoint_vae(checkpoint_dir, vae, optimizer, device)

    num_epochs = 2
    for epoch in range(start_epoch, num_epochs):
        vae.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
    
        avg_train_loss = train_loss / len(data_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")
    
        # Plot five generated images every five epochs
        if (epoch + 1) % 5 == 0:
            vae.eval()
            with torch.no_grad():
                z = torch.randn(5, 512).to(device)
                sample = vae.decoder(z).cpu()
                fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                for i, ax in enumerate(axes.flat):
                    img = sample[i].squeeze().numpy()
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                plt.show()
                
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'vae_state_dict': vae.state_dict(),
                    'optimizer_vae_state_dict': optimizer.state_dict(),
                }, epoch + 1, checkpoint_dir)
    return vae

def generate_images(vae, device, save_dir):
    # Generate 1500 synthetic images
    TOTAL = 1500
    vae.eval()
    with torch.no_grad():
        for i in range(TOTAL):
            z = torch.randn(1, 512).to(device)
            sample = vae.decoder(z).cpu()
            img = sample[0].squeeze().numpy()
            img = (img * 255).astype(np.uint8) 
            path = f"{save_dir}{i}.jpeg"
            cv2.imwrite(path, img)