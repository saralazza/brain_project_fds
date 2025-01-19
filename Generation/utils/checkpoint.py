import os
import torch

def save_checkpoint(state, epoch, checkpoint_dir):

    filename = f"checkpoint_epoch_{epoch}.pth.tar" 
    torch.save(state, os.path.join(checkpoint_dir, filename)) 
 
def load_checkpoint_gan(checkpoint_dir, G, D, optimizer_G, optimizer_D, device): 
    # Find checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')] 
    if not checkpoints: 
        return 0  # If there is not previous model version, start from epoch 0
 
    # Sort version based on the epoch
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) 
    latest_checkpoint = checkpoints[-1] 
 
    # Load last checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint) 
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device)) 
    G.load_state_dict(checkpoint['generator_state_dict']) 
    D.load_state_dict(checkpoint['discriminator_state_dict']) 
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict']) 
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict']) 
    return checkpoint['epoch'] 

def load_checkpoint_vae(checkpoint_dir, vae, optimizer, device): 
    # Find the last version of the model
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')] 
    if not checkpoints: 
        return 0 # If no file is found, restart from epoch 0
 
    # Sort checkpoints based on the epoch 
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) 
    latest_checkpoint = checkpoints[-1] 
 
    # Load the last checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint) 
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    vae.load_state_dict(checkpoint['vae_state_dict']) 
    optimizer.load_state_dict(checkpoint['optimizer_vae_state_dict']) 
    return checkpoint['epoch']