import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import cv2

from utils.checkpoint import save_checkpoint, load_checkpoint_gan

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            # First linear layer to map the latent vector to a higher dimension
            nn.Linear(latent_dim, 128 * 8 * 8),  # Map the latent vector to 128x8x8
            nn.ReLU(True),
            nn.Unflatten(1, (128, 8, 8)),  # Reconstruct the output as an image (128, 8, 8)
        )

        # Convolutional layers to increase image size
        self.model = nn.Sequential(
            # First upsampling layer
            nn.Upsample(scale_factor=2),  # (128, 16, 16) -> (128, 32, 32)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # (128, 32, 32) -> (128, 32, 32)
            nn.BatchNorm2d(128, momentum=0.78), 
            nn.ReLU(), 

            # Second upsampling layer
            nn.Upsample(scale_factor=2),  # (128, 32, 32) -> (128, 64, 64)
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # (128, 64, 64) -> (64, 64, 64)
            nn.BatchNorm2d(64, momentum=0.78),
            nn.ReLU(), 

            # Third upsampling layer
            nn.Upsample(scale_factor=2),  # (64, 64, 64) -> (64, 128, 128)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # (64, 128, 128) -> (32, 128, 128)
            nn.BatchNorm2d(32, momentum=0.78), 
            nn.ReLU(), 

            # Fourth upsampling layer
            nn.Upsample(scale_factor=2),  # (32, 128, 128) -> (32, 256, 256)
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # (32, 256, 256) -> (1, 256, 256)
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z)
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # grayscale image
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(512 * 8 * 8, 1)  # Flattened output
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))  # Probability of real image or not
    

def train_loop_gan(device, data_loader, checkpoint_dir):

    # Initialize Generator and Discriminator
    G = Generator().to(device)
    D = Discriminator().to(device)

    # Initialize optimizers relative to generator and discriminator models
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss Function
    criterion = nn.BCELoss()
    
    # Load the last checkpoint, if present, and obtain the starting epoch for the train loop
    start_epoch = load_checkpoint_gan(checkpoint_dir, G, D, optimizer_G, optimizer_D, device) 
    
    # Epochs and interval to save model checkpoints
    num_epochs = 500 
    save_interval = 50 
    
    # Training 
    for epoch in range(start_epoch, num_epochs): 
        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            
            batch_size = real_images.size(0)

            # Create discriminator labels: 1 for real images and 0 for fake images
            real_labels = torch.ones(batch_size, 1).to(device) 
            fake_labels = torch.zeros(batch_size, 1).to(device) 

            # Train discriminator on real images
            D.zero_grad()
            outputs = D(real_images)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            # Train discriminator on fake images
            z = torch.randn(batch_size, 100).to(device)  # Latent vector
            fake_images = G(z)
            outputs = D(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()

            optimizer_D.step()

            # Train generator
            G.zero_grad()
            outputs = D(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()

            optimizer_G.step()

        # Plot five generated images every five epochs
        if (epoch + 1) % 5 == 0:
            G.eval()
            with torch.no_grad():
                z = torch.randn(5, 100).to(device)
                sample = G(z).cpu()

                # Visualize images
                fig, axes = plt.subplots(8, 8, figsize=(10, 10))
                for i, ax in enumerate(axes.flat):
                    img = sample[i].squeeze().numpy()
                    if img.shape[0] == 3:
                        img = img.transpose(1, 2, 0)
                    ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
                    ax.axis('off')
                plt.show()

        # Save model version every save_interval iterations
        print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss_real.item() + d_loss_fake.item()}, g_loss: {g_loss.item()}")
        if (epoch + 1) % save_interval == 0: 
            save_checkpoint({ 
                'epoch': epoch + 1, 
                'generator_state_dict': G.state_dict(), 
                'discriminator_state_dict': D.state_dict(), 
                'optimizer_G_state_dict': optimizer_G.state_dict(), 
                'optimizer_D_state_dict': optimizer_D.state_dict(), 
            }, epoch + 1)
    return G, D

def generate_images(G, device, save_dir):
    # Generate 1500 synthetic images
    TOTAL = 1500
    G.eval()
    with torch.no_grad():
        for i in range(TOTAL):
            z = torch.randn(1, 100).to(device)
            sample = G(z).cpu()
            img = sample[0].squeeze().numpy()
            img = (img * 255).astype(np.uint8)
            path = f"{save_dir}{i}.jpeg"
            cv2.imwrite(path, img)
